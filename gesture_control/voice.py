"""Voice commands: offline speech recognition with Vosk.

Pieces (only VoiceListener touches the mic / Vosk; the rest is plain logic):
  normalize()       text clean-up
  CommandMatcher    phrase (+ learned aliases) -> action
  SpeakerProfile    "only respond to me" (Vosk x-vector speaker embeddings)
  VoiceController   wake word / listen window / confidence / speaker checks
  VoiceListener     background thread: microphone -> Vosk -> result queue

No audio is ever written to disk. Recognition runs entirely on this Mac.
"""
import difflib
import json
import queue
import re
import subprocess
import threading
import zipfile
from pathlib import Path

import numpy as np

from .config import ROOT

MODELS_DIR = ROOT / "models"
PROFILE_FILE = MODELS_DIR / "voice_profile.json"
MODEL_BASE_URL = "https://alphacephei.com/vosk/models/"
SPK_MODEL = "vosk-model-spk-0.4"
SAMPLE_RATE = 16000


# --------------------------------------------------------------------- text
def normalize(text):
    text = (text or "").lower().replace("[unk]", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------- matching
class CommandMatcher:
    def __init__(self, commands, aliases=None, fuzzy_cutoff=0.82):
        self.commands = {normalize(p): a for p, a in (commands or {}).items() if normalize(p)}
        self.aliases = {}  # alias text -> canonical phrase
        for alias, phrase in (aliases or {}).items():
            a, p = normalize(alias), normalize(phrase)
            if a and p in self.commands and a not in self.commands:
                self.aliases[a] = p
        self.fuzzy_cutoff = fuzzy_cutoff

    def phrases(self):
        return list(self.commands) + list(self.aliases)

    def match(self, text):
        """Return (canonical phrase, action) or (None, None)."""
        t = normalize(text)
        if not t:
            return None, None
        if t in self.commands:
            return t, self.commands[t]
        if t in self.aliases:
            p = self.aliases[t]
            return p, self.commands[p]
        close = difflib.get_close_matches(t, self.phrases(), n=1, cutoff=self.fuzzy_cutoff)
        if close:
            p = self.aliases.get(close[0], close[0])
            return p, self.commands[p]
        return None, None

    def grammar(self, wake_words=()):
        """Phrase list for Vosk: restricting the vocabulary makes command
        recognition far more accurate than open dictation."""
        base = self.phrases()
        g = set(base)
        for w in wake_words:
            g.add(w)
            g.update(f"{w} {p}" for p in base)
        return sorted(g) + ["[unk]"]


# ------------------------------------------------------------------ speaker
def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


class SpeakerProfile:
    def __init__(self, embedding=None, threshold=0.5, samples=0):
        self.embedding = None if embedding is None else _unit(embedding)
        self.threshold = float(threshold)
        self.samples = samples

    @property
    def ready(self):
        return self.embedding is not None

    def similarity(self, vec):
        if not self.ready or vec is None or len(vec) == 0:
            return None
        return float(np.dot(self.embedding, _unit(vec)))

    @classmethod
    def enroll(cls, vectors):
        """Average embedding + a threshold calibrated on your own samples."""
        V = np.stack([_unit(v) for v in vectors])
        mean = _unit(V.mean(axis=0))
        sims = V @ mean
        # just below your own weakest typical sample, within sane bounds
        thr = float(np.clip(np.percentile(sims, 10) - 0.10, 0.30, 0.75))
        return cls(mean, thr, len(V)), sims


# --------------------------------------------------------------- profile io
def load_profile(path=PROFILE_FILE):
    path = Path(path)
    if not path.exists():
        return {"aliases": {}, "speaker": None}
    return json.loads(path.read_text())


def save_profile(data, path=PROFILE_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


# --------------------------------------------------------------- controller
class VoiceController:
    """Decides whether a recognised utterance is a command for us."""

    def __init__(self, vcfg, profile=None):
        self.cfg = vcfg
        profile = profile or {"aliases": {}, "speaker": None}
        self.matcher = CommandMatcher(vcfg["commands"], profile.get("aliases"))
        self.wake_words = [normalize(w) for w in self._as_list(vcfg["wake_word"]) if normalize(w)]
        # learned ways you say the wake word (from --train-voice)
        self.wake_words += [normalize(a) for a in (profile.get("wake_aliases") or {})
                            if normalize(a) and normalize(a) not in self.wake_words]
        # longest first so "hey computer" wins over "computer"
        self.wake_words.sort(key=len, reverse=True)
        mode = vcfg["activation"]
        self.wake_allowed = mode in ("wake_word", "both")
        self.gesture_allowed = mode in ("gesture", "both")
        self.always = mode == "always"
        spk = profile.get("speaker")
        thr = vcfg["speaker_threshold"]
        self.speaker = SpeakerProfile()
        if spk:
            self.speaker = SpeakerProfile(spk["embedding"],
                                          spk["threshold"] if thr == "auto" else float(thr),
                                          spk.get("samples", 0))
        self.check_speaker = bool(vcfg["only_my_voice"]) and self.speaker.ready
        self.listen_until = 0.0
        self.muted = False
        self.last_heard = ""
        self.last_t = -1e9

    @staticmethod
    def _as_list(v):
        return v if isinstance(v, list) else [v]

    def grammar(self):
        return self.matcher.grammar(self.wake_words if self.wake_allowed else ())

    def listening(self, t):
        return self.always or t < self.listen_until

    def open_window(self, t, seconds=None):
        self.listen_until = t + (seconds or self.cfg["listen_seconds"])

    def gesture_listen(self, t):
        if self.gesture_allowed or self.always:
            self.open_window(t)
            return True
        return False

    def _strip_wake(self, text):
        for w in self.wake_words:
            if text == w:
                return True, ""
            if text.startswith(w + " "):
                return True, text[len(w) + 1:]
        return False, text

    def process(self, result, t):
        """result: Vosk result dict. Returns an event dict or None.

        event = {"type": "command"|"listening"|"rejected", "phrase", "action",
                 "reason", "text", "similarity"}
        """
        text = normalize(result.get("text", ""))
        if not text:
            return None
        self.last_heard, self.last_t = text, t
        ev = {"text": text, "phrase": None, "action": None, "reason": "", "similarity": None}

        # 1. speaker check ("only respond to me")
        if self.check_speaker:
            sim = self.speaker.similarity(result.get("spk"))
            ev["similarity"] = sim
            if sim is not None and sim < self.speaker.threshold:
                return {**ev, "type": "rejected", "reason": f"not your voice ({sim:.2f})"}
            if sim is None and not self.listening(t):
                return {**ev, "type": "rejected", "reason": "too short to verify voice"}

        # 2. confidence
        words = result.get("result") or []
        if words:
            conf = sum(w.get("conf", 1.0) for w in words) / len(words)
            if conf < self.cfg["min_confidence"]:
                return {**ev, "type": "rejected", "reason": f"unsure ({conf:.2f})"}

        # 3. activation
        woke, rest = self._strip_wake(text) if self.wake_allowed else (False, text)
        if woke and not rest:
            self.open_window(t)
            return {**ev, "type": "listening", "reason": "wake word"}
        if not woke and not self.listening(t):
            return None  # just conversation: ignore silently

        # 4. command
        phrase, action = self.matcher.match(rest)
        if action is None:
            return {**ev, "type": "rejected", "reason": "unknown command"}
        if self.muted and action != "builtin:voice_on":
            return {**ev, "type": "rejected", "reason": "voice muted (say 'start listening')"}
        self.open_window(t)  # allow quick follow-ups ("scroll down" ... "scroll down")
        return {**ev, "type": "command", "phrase": phrase, "action": action}


# ------------------------------------------------------------ models / audio
def model_path(name):
    return MODELS_DIR / name


def ensure_model(name):
    """Download & unzip a Vosk model into models/ if it's missing."""
    path = model_path(name)
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zpath = MODELS_DIR / f"{name}.zip"
    print(f"Downloading voice model {name} ...")
    subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(zpath), MODEL_BASE_URL + f"{name}.zip"], check=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(MODELS_DIR)
    zpath.unlink()
    return path


def vocabulary(model_dir):
    for rel in ("graph/words.txt", "words.txt"):
        f = Path(model_dir) / rel
        if f.exists():
            return {line.split()[0] for line in f.read_text(errors="ignore").splitlines() if line.strip()}
    return None


def unknown_words(phrases, vocab):
    if not vocab:
        return []
    return sorted({w for p in phrases for w in p.split() if w not in vocab and w != "[unk]"})


class Recognizers:
    """Loads the Vosk speech + speaker models once."""

    def __init__(self, vcfg):
        from vosk import Model, SetLogLevel, SpkModel
        SetLogLevel(-1)
        self.model_dir = ensure_model(vcfg["model"])
        self.model = Model(str(self.model_dir))
        self.spk = SpkModel(str(ensure_model(SPK_MODEL)))
        self.vocab = vocabulary(self.model_dir)

    def make(self, grammar=None):
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(self.model, SAMPLE_RATE, json.dumps(grammar)) if grammar \
            else KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)
        rec.SetSpkModel(self.spk)
        return rec


class Microphone:
    """16 kHz mono int16 audio blocks in a queue."""

    def __init__(self, device=None, block=4000):
        import sounddevice as sd
        self.q = queue.Queue(maxsize=200)
        self.stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=block, device=device,
                                        dtype="int16", channels=1, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def read(self, timeout=0.5):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class VoiceListener(threading.Thread):
    """Background thread: mic -> grammar-restricted Vosk -> results queue."""

    def __init__(self, vcfg, grammar):
        super().__init__(daemon=True)
        self.vcfg = vcfg
        self.grammar = grammar
        self.results = queue.Queue()
        self.partial = ""
        self.error = None
        self._stop_evt = threading.Event()
        self._regrammar = None

    def set_grammar(self, grammar):
        self._regrammar = grammar

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            recs = Recognizers(self.vcfg)
            bad = unknown_words(self.grammar, recs.vocab)
            if bad:
                print(f"[voice] these words aren't in the speech model and will be ignored: {', '.join(bad)}")
            rec = recs.make(self.grammar)
            with Microphone(self.vcfg.get("device")) as mic:
                print("[voice] listening (offline)")
                while not self._stop_evt.is_set():
                    if self._regrammar is not None:
                        self.grammar, self._regrammar = self._regrammar, None
                        rec = recs.make(self.grammar)
                    data = mic.read()
                    if data is None:
                        continue
                    if rec.AcceptWaveform(data):
                        self.partial = ""
                        res = json.loads(rec.Result())
                        if normalize(res.get("text")):
                            self.results.put(res)
                    else:
                        self.partial = normalize(json.loads(rec.PartialResult()).get("partial", ""))
        except Exception as e:  # mic permission, missing model, ...
            self.error = e
            print(f"[voice] disabled: {e}\n  -> check System Settings > Privacy & Security > Microphone "
                  "for your terminal app, and that setup.sh downloaded the voice models.")
