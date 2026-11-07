"""Voice training & testing (terminal based).

  ./run.sh --train-voice                 say the wake word + every command 3x
  ./run.sh --train-voice "next desktop"  re-train just some phrases
  ./run.sh --voice-test                  live: what it hears, voice match, decision
  ./run.sh --voice-commands              list phrases and learned variants
  ./run.sh --reset-voice                 forget learned pronunciations + voice ID

What training learns (no audio is stored):
  * pronunciation: how the recogniser hears *you* say each phrase; consistent
    mis-hearings are saved as aliases for that command
  * voice ID: an averaged speaker embedding + a threshold calibrated on you
"""
import difflib
import json
import time
from collections import Counter

import numpy as np

from .config import load_config
from .voice import (PROFILE_FILE, CommandMatcher, Microphone, Recognizers, SpeakerProfile,
                    VoiceController, load_profile, normalize, save_profile, unknown_words)


def _capture(recs, mic, timeout=7.0):
    """Record one utterance with an open-vocabulary recogniser.
    Returns (text, spk_vector, audio_bytes) or None on timeout."""
    while mic.read(timeout=0.01) is not None:  # drop stale audio
        pass
    rec = recs.make()
    audio = bytearray()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        data = mic.read()
        if data is None:
            continue
        audio += data
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if normalize(res.get("text")):
                return normalize(res["text"]), res.get("spk"), bytes(audio)
    res = json.loads(rec.FinalResult())
    if normalize(res.get("text")):
        return normalize(res["text"]), res.get("spk"), bytes(audio)
    return None


def learn_aliases(phrase, heard, all_phrases):
    """Keep consistent or similar mis-hearings of `phrase` as aliases."""
    out = {}
    counts = Counter(h for h in heard if h and h != phrase)
    for h, n in counts.items():
        if h in all_phrases or len(h) < 3:
            continue  # would clash with another command / too short to be safe
        sim = difflib.SequenceMatcher(None, h, phrase).ratio()
        if n >= 2 or sim >= 0.5:
            out[h] = phrase
    return out


def train_voice(only=None, reps=3, config_path=None):
    cfg = load_config(config_path)
    vcfg = cfg["voice"]
    commands = {normalize(p): a for p, a in vcfg["commands"].items()}
    wake = [normalize(w) for w in (vcfg["wake_word"] if isinstance(vcfg["wake_word"], list) else [vcfg["wake_word"]])]

    if only:
        targets = [normalize(p) for p in only]
        missing = [p for p in targets if p not in commands and p not in wake]
        if missing:
            raise SystemExit(f"Not in voice.commands in config.yaml: {', '.join(missing)}")
    else:
        targets = wake + list(commands)

    print("Loading speech models ...")
    recs = Recognizers(vcfg)
    bad = unknown_words(targets, recs.vocab)
    if bad:
        print(f"Warning: the speech model doesn't know these words, change those phrases: {', '.join(bad)}")

    profile = load_profile()
    heard, audio_of, vectors = {}, {}, []
    print(f"\nYou'll say {len(targets)} phrase(s) {reps}x each. Speak normally, at your usual distance.\n"
          "Press Enter to start each phrase, 's' + Enter to skip, Ctrl+C to stop (progress is kept).\n")
    try:
        with Microphone(vcfg.get("device")) as mic:
            for i, phrase in enumerate(targets, 1):
                if input(f"[{i}/{len(targets)}]  \"{phrase}\"  -> Enter ").strip().lower() == "s":
                    continue
                heard[phrase], audio_of[phrase] = [], []
                r = 0
                fails = 0
                while r < reps:
                    print(f"     say it ({r + 1}/{reps}) ...", end="", flush=True)
                    got = _capture(recs, mic)
                    if got is None:
                        fails += 1
                        print(" didn't hear anything" + (", skipping" if fails >= 3 else ", again"))
                        if fails >= 3:
                            break
                        continue
                    text, spk, audio = got
                    print(f" heard \"{text}\"" + ("" if text == phrase else "  (learning this)"))
                    heard[phrase].append(text)
                    audio_of[phrase].append(audio)
                    if spk:
                        vectors.append(spk)
                    r += 1
    except KeyboardInterrupt:
        print("\nStopped - saving what was recorded.")

    if not heard:
        print("Nothing recorded; profile unchanged.")
        return

    # ---- pronunciation aliases
    aliases = {a: p for a, p in profile.get("aliases", {}).items() if p not in heard}
    all_phrases = set(commands) | set(wake)
    wake_aliases = {a: p for a, p in profile.get("wake_aliases", {}).items() if p not in heard}
    for phrase, hs in heard.items():
        learned = learn_aliases(phrase, hs, all_phrases)
        (wake_aliases if phrase in wake else aliases).update(learned)
    profile["aliases"] = aliases
    profile["wake_aliases"] = wake_aliases

    # ---- voice ID
    if vectors:
        new, sims = SpeakerProfile.enroll(vectors)
        old = profile.get("speaker")
        if old and old.get("samples"):
            n_old = old["samples"]
            mixed = (np.asarray(old["embedding"]) * n_old + new.embedding * new.samples) / (n_old + new.samples)
            new.embedding = mixed / np.linalg.norm(mixed)
            new.samples += n_old
            new.threshold = min(new.threshold, old["threshold"])
        profile["speaker"] = {"embedding": new.embedding.tolist(), "threshold": new.threshold,
                              "samples": new.samples}
        print(f"\nVoice ID: {new.samples} samples, your similarity {np.median(sims):.2f} typical, "
              f"threshold {new.threshold:.2f}")

    save_profile(profile)
    print(f"Saved {PROFILE_FILE} (aliases + voice ID numbers; no audio)")

    # ---- accuracy check with the real command grammar
    ctrl = VoiceController(vcfg, profile)
    grammar = ctrl.grammar()
    print("\nChecking recognition with your recordings:")
    weak = []
    for phrase, clips in audio_of.items():
        if not clips:
            continue
        ok = 0
        for clip in clips:
            rec = recs.make(grammar)
            rec.AcceptWaveform(clip)
            text = normalize(json.loads(rec.FinalResult()).get("text"))
            if phrase in wake:
                ok += int(text == phrase or text in wake_aliases)
            else:
                _, rest = ctrl._strip_wake(text)
                ok += int(ctrl.matcher.match(rest)[0] == phrase)
        mark = "" if ok == len(clips) else "   <- re-train or pick a more distinct phrase"
        if ok < len(clips):
            weak.append(phrase)
        print(f"  {phrase:<24} {ok}/{len(clips)}{mark}")
    learned = {a: p for a, p in aliases.items() if p in heard}
    if learned:
        print("\nLearned pronunciations:")
        for a, p in learned.items():
            print(f"  \"{a}\" -> {p}")
    if weak:
        print('\nTip: ./run.sh --train-voice "' + '" "'.join(weak) + '"')


def voice_test(config_path=None):
    """Show what's heard and what would happen, without doing anything."""
    from .voice import VoiceListener
    cfg = load_config(config_path)
    vcfg = cfg["voice"]
    ctrl = VoiceController(vcfg, load_profile())
    print(f"Wake word: {', '.join(ctrl.wake_words)} | activation: {vcfg['activation']} | "
          f"voice ID: {'on' if ctrl.check_speaker else 'off'}")
    print("Speak commands; nothing will be executed. Ctrl+C to stop.\n")
    lis = VoiceListener(vcfg, ctrl.grammar())
    lis.start()
    try:
        while lis.is_alive():
            try:
                res = lis.results.get(timeout=0.3)
            except Exception:
                continue
            t = time.monotonic()
            ev = ctrl.process(res, t)
            sim = ctrl.speaker.similarity(res.get("spk")) if ctrl.speaker.ready else None
            s = f"  voice={sim:.2f}" if sim is not None else ""
            if ev is None:
                print(f"  heard \"{normalize(res.get('text'))}\"{s}  -> ignored (no wake word)")
            elif ev["type"] == "command":
                print(f"  heard \"{ev['text']}\"{s}  -> {ev['phrase']}  =>  {ev['action']}")
            elif ev["type"] == "listening":
                print(f"  heard \"{ev['text']}\"{s}  -> listening for {vcfg['listen_seconds']}s ...")
            else:
                print(f"  heard \"{ev['text']}\"{s}  -> rejected: {ev['reason']}")
    except KeyboardInterrupt:
        pass
    lis.stop()


def list_commands(config_path=None):
    cfg = load_config(config_path)
    vcfg = cfg["voice"]
    profile = load_profile()
    m = CommandMatcher(vcfg["commands"], profile.get("aliases"))
    wake = vcfg["wake_word"]
    print(f"Wake word: {wake}   activation: {vcfg['activation']}\n")
    by_phrase = {}
    for a, p in m.aliases.items():
        by_phrase.setdefault(p, []).append(a)
    for p, action in m.commands.items():
        extra = f"   (also: {', '.join(by_phrase[p])})" if p in by_phrase else ""
        print(f"  {p:<24} {action}{extra}")
    spk = profile.get("speaker")
    print(f"\nVoice ID: {'trained, ' + str(spk['samples']) + ' samples' if spk else 'not trained'}")


def reset_voice():
    if PROFILE_FILE.exists():
        PROFILE_FILE.unlink()
        print("Voice profile deleted (aliases + voice ID).")
    else:
        print("No voice profile to delete.")
