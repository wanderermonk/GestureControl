"""Load config.yaml and merge it over the built-in defaults."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.yaml"

DEFAULTS = {
    "gestures": {},
    "cursor": {"region": [0.2, 0.15, 0.8, 0.7], "min_cutoff": 0.9, "beta": 0.012, "deadzone_px": 2},
    "scroll": {"speed": 2500, "natural": False, "horizontal": False, "min_step": 3},
    "timing": {
        "stable_frames": 3, "click_freeze": 0.12, "drag_delay": 0.35, "double_click": 0.45,
        "hold_time": 0.8, "hold_still": 0.06, "swipe_window": 0.4, "swipe_distance": 0.16,
        "swipe_cooldown": 0.9, "lost_reset": 0.3,
    },
    "camera": {"index": 0, "width": 640, "height": 480, "mirror": True},
    "tracking": {
        "min_detection_confidence": 0.6, "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5, "pinch_on": 0.28, "pinch_off": 0.40,
        "use_trained_model": True, "trained_min_confidence": 0.6,
    },
    "ui": {"preview": True, "menubar": True, "preview_width": 480},
    "voice": {
        "enabled": True, "model": "vosk-model-small-en-us-0.15", "device": None,
        "wake_word": "computer", "activation": "both", "listen_seconds": 5,
        "only_my_voice": True, "speaker_threshold": "auto", "min_confidence": 0.55,
        "commands": {},
    },
}

NO_DEEP_MERGE = {"gestures", "commands"}  # user lists replace the defaults


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in NO_DEEP_MERGE:
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    path = Path(path) if path else DEFAULT_PATH
    data = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, data)
    validate(cfg)
    return cfg


def validate(cfg):
    x0, y0, x1, y1 = cfg["cursor"]["region"]
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("cursor.region must be [x0, y0, x1, y1] with 0 <= x0 < x1 <= 1")
    for name, g in cfg["gestures"].items():
        action = g if isinstance(g, str) else (g or {}).get("action")
        if not action:
            raise ValueError(f"gesture '{name}' has no action")
        if "_hold" not in name and "_swipe_" not in name:
            raise ValueError(f"gesture '{name}' must end in _hold or _swipe_<direction>")
    v = cfg["voice"]
    if v["activation"] not in ("wake_word", "gesture", "both", "always"):
        raise ValueError("voice.activation must be wake_word, gesture, both or always")
    for phrase, action in (v["commands"] or {}).items():
        if not isinstance(action, str) or ":" not in action and action != "none":
            raise ValueError(f"voice command '{phrase}' needs an action like 'system:mission_control'")
