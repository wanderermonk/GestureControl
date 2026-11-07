import numpy as np
import pytest

from gesture_control.actions import ActionRunner, DryRunBackend
from gesture_control.config import load_config
from gesture_control.voice import (CommandMatcher, SpeakerProfile, VoiceController, normalize)
from gesture_control.voice_trainer import learn_aliases

RNG = np.random.default_rng(0)
ME = RNG.normal(size=128)
OTHER = RNG.normal(size=128)


def my_vec(noise=0.3):
    return (ME + RNG.normal(scale=noise * np.linalg.norm(ME) / np.sqrt(128), size=128)).tolist()


def other_vec():
    return (OTHER + RNG.normal(scale=0.3, size=128)).tolist()


def res(text, spk=None, conf=0.95):
    r = {"text": text, "result": [{"word": w, "conf": conf} for w in text.split()]}
    if spk is not None:
        r["spk"] = spk
    return r


def vcfg(**over):
    v = load_config()["voice"]
    v.update(over)
    return v


def profile_with_voice():
    sp, _ = SpeakerProfile.enroll([my_vec() for _ in range(30)])
    return {"aliases": {}, "speaker": {"embedding": sp.embedding.tolist(),
                                        "threshold": sp.threshold, "samples": sp.samples}}


# ------------------------------------------------------------------ matching
def test_normalize():
    assert normalize("  Next, Desktop! [unk] ") == "next desktop"


def test_matcher_exact_alias_fuzzy():
    m = CommandMatcher({"next desktop": "system:space_right", "mute": "media:mute"},
                       aliases={"necks desktop": "next desktop"})
    assert m.match("next desktop") == ("next desktop", "system:space_right")
    assert m.match("necks desktop") == ("next desktop", "system:space_right")
    assert m.match("next desktops") == ("next desktop", "system:space_right")
    assert m.match("open the pod bay doors") == (None, None)


def test_grammar_has_wake_combos_and_unk():
    g = CommandMatcher({"mute": "media:mute"}).grammar(["computer"])
    assert "computer mute" in g and "mute" in g and "computer" in g and g[-1] == "[unk]"


def test_default_commands_are_unique_and_valid():
    cmds = load_config()["voice"]["commands"]
    assert len(cmds) >= 30
    assert len({normalize(p) for p in cmds}) == len(cmds)


# ---------------------------------------------------------------- activation
def test_wake_word_inline_command():
    c = VoiceController(vcfg())
    ev = c.process(res("computer next desktop"), 0.0)
    assert ev["type"] == "command" and ev["action"] == "system:space_right"


def test_conversation_without_wake_word_is_ignored():
    c = VoiceController(vcfg())
    assert c.process(res("next desktop"), 0.0) is None


def test_wake_then_command_within_window():
    c = VoiceController(vcfg())
    assert c.process(res("computer"), 0.0)["type"] == "listening"
    assert c.process(res("volume up"), 2.0)["action"] == "media:volume_up"
    # follow-ups extend the window
    assert c.process(res("volume up"), 6.0)["action"] == "media:volume_up"
    assert c.process(res("volume up"), 20.0) is None


def test_gesture_opens_window():
    c = VoiceController(vcfg(activation="gesture"))
    assert c.process(res("computer mute"), 0.0) is None  # wake word disabled
    assert c.gesture_listen(1.0)
    assert c.process(res("mute"), 2.0)["action"] == "media:mute"


def test_wake_word_mode_ignores_gesture():
    c = VoiceController(vcfg(activation="wake_word"))
    assert not c.gesture_listen(0.0)
    assert c.process(res("mute"), 0.5) is None


def test_always_mode():
    c = VoiceController(vcfg(activation="always"))
    assert c.process(res("mute"), 100.0)["action"] == "media:mute"


def test_low_confidence_rejected():
    c = VoiceController(vcfg())
    ev = c.process(res("computer close window", conf=0.3), 0.0)
    assert ev["type"] == "rejected" and "unsure" in ev["reason"]


def test_unknown_command_after_wake():
    ev = VoiceController(vcfg()).process(res("computer make me a sandwich"), 0.0)
    assert ev["type"] == "rejected" and ev["reason"] == "unknown command"


def test_multiple_and_learned_wake_words():
    c = VoiceController(vcfg(wake_word=["computer", "hey mac"]), {"aliases": {}, "wake_aliases": {"computor": "computer"}})
    assert c.process(res("hey mac mute"), 0)["action"] == "media:mute"
    assert c.process(res("computor mute"), 0)["action"] == "media:mute"


def test_learned_alias_used():
    c = VoiceController(vcfg(), {"aliases": {"necks desktop": "next desktop"}})
    assert c.process(res("computer necks desktop"), 0)["action"] == "system:space_right"


# ------------------------------------------------------------------- speaker
def test_enroll_threshold_separates_me_from_others():
    sp, sims = SpeakerProfile.enroll([my_vec() for _ in range(30)])
    mine = [sp.similarity(my_vec()) for _ in range(50)]
    theirs = [sp.similarity(other_vec()) for _ in range(50)]
    assert np.mean(np.array(mine) >= sp.threshold) > 0.9
    assert np.mean(np.array(theirs) >= sp.threshold) < 0.05


def test_only_my_voice():
    c = VoiceController(vcfg(), profile_with_voice())
    assert c.check_speaker
    assert c.process(res("computer mute", spk=my_vec()), 0)["type"] == "command"
    ev = c.process(res("computer mute", spk=other_vec()), 1)
    assert ev["type"] == "rejected" and "not your voice" in ev["reason"]


def test_voice_id_can_be_disabled():
    c = VoiceController(vcfg(only_my_voice=False), profile_with_voice())
    assert c.process(res("computer mute", spk=other_vec()), 0)["type"] == "command"


def test_unverifiable_short_utterance_needs_open_window():
    c = VoiceController(vcfg(), profile_with_voice())
    assert c.process(res("computer mute"), 0)["type"] == "rejected"  # no spk vector
    c.gesture_listen(1.0)
    assert c.process(res("mute"), 1.5)["type"] == "command"


# -------------------------------------------------------------------- mute
def test_voice_mute_only_accepts_start_listening():
    c = VoiceController(vcfg())
    c.muted = True
    assert c.process(res("computer mute"), 0)["type"] == "rejected"
    assert c.process(res("computer start listening"), 1)["action"] == "builtin:voice_on"


# ------------------------------------------------------------------ aliases
def test_learn_aliases():
    phrases = {"next desktop", "mute", "click"}
    got = learn_aliases("next desktop", ["next desktop", "necks desktop", "necks desktop"], phrases)
    assert got == {"necks desktop": "next desktop"}
    # one-off and very different mis-hearing is not learned
    assert learn_aliases("next desktop", ["banana"], phrases) == {}
    # never steal another command's phrase
    assert learn_aliases("next desktop", ["click", "click"], phrases) == {}


# ------------------------------------------------------------- mouse actions
@pytest.mark.parametrize("cmd,expected", [
    ("click", [("click", 100, 100, "left", 1)]),
    ("right_click", [("click", 100, 100, "right", 1)]),
    ("double_click", [("click", 100, 100, "left", 1), ("click", 100, 100, "left", 2)]),
    ("scroll_down", [("scroll", -400, 0)]),
])
def test_mouse_actions(cmd, expected):
    b = DryRunBackend(log=False)
    ActionRunner(b)._safe("mouse", cmd)
    assert b.calls == expected
