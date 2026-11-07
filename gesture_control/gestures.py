"""Per-frame hand pose classification from 21 MediaPipe hand landmarks.

Landmarks are (x, y) in normalised image coordinates (0..1), already mirrored
so that moving your hand to your right increases x. y grows downwards.

Landmark indices:
  0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
"""
from dataclasses import dataclass, field
import math

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
THUMB_MCP = 2

FINGERS = {  # name: (pip, tip)
    "index": (INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_PIP, RING_TIP),
    "pinky": (PINKY_PIP, PINKY_TIP),
}

# Pose names used throughout the app and in config.yaml
POSES = [
    "none",         # no hand / unrecognised
    "point",        # index up -> cursor
    "pinch",        # thumb+index pinch (with ring & pinky folded) -> click / drag
    "pinch_middle", # thumb+middle pinch -> right click
    "two",          # index+middle up -> scroll
    "three",        # index+middle+ring up
    "palm",         # all four fingers up
    "fist",         # everything folded -> neutral / clutch
    "thumbs_up",
    "thumbs_down",
    "shaka",        # thumb + pinky
    "rock",         # index + pinky
    "ok",           # thumb-index pinch, other three fingers up
]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class HandState:
    pose: str = "none"
    fingers: dict = field(default_factory=dict)  # name -> bool (extended)
    thumb: bool = False
    pinch: float = 9.0          # thumb-index distance / hand size
    pinch_middle: float = 9.0   # thumb-middle distance / hand size
    size: float = 0.0           # wrist -> middle MCP, normalised image units
    anchor: tuple = (0.0, 0.0)  # cursor anchor point (index knuckle)
    palm: tuple = (0.0, 0.0)    # palm centre


class PoseClassifier:
    """Stateless geometry + small hysteresis for pinches."""

    def __init__(self, pinch_on=0.28, pinch_off=0.40, extend_ratio=1.12,
                 model=None, min_confidence=0.6):
        self.model = model              # optional learned.GestureModel
        self.min_confidence = min_confidence
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self.extend_ratio = extend_ratio
        self.min_reach = 0.4
        self.last_model = (None, 0.0)
        self._pinching = False
        self._pinching_mid = False

    def reset(self):
        self._pinching = False
        self._pinching_mid = False

    def finger_extended(self, lm, pip, tip):
        w = lm[WRIST]
        return dist(w, lm[tip]) > dist(w, lm[pip]) * self.extend_ratio

    def thumb_extended(self, lm, size):
        # Tucked thumbs end up near the middle knuckle; extended ones point away.
        far_from_palm = dist(lm[THUMB_TIP], lm[MIDDLE_MCP]) > dist(lm[THUMB_IP], lm[MIDDLE_MCP]) * 1.08
        far_from_index = dist(lm[THUMB_TIP], lm[INDEX_MCP]) > 0.45 * size
        return far_from_palm and far_from_index

    def classify(self, lm):
        if lm is None:
            self.reset()
            return HandState()

        size = max(dist(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)
        f = {n: self.finger_extended(lm, p, t) for n, (p, t) in FINGERS.items()}
        thumb = self.thumb_extended(lm, size)

        p_idx = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / size
        p_mid = dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / size
        # A real pinch keeps the finger reaching out; in a fist the curled
        # fingertip sits on the palm right next to the thumb, which must not count.
        idx_reach = dist(lm[INDEX_TIP], lm[INDEX_MCP]) / size > self.min_reach
        mid_reach = dist(lm[MIDDLE_TIP], lm[MIDDLE_MCP]) / size > self.min_reach
        self._pinching = idx_reach and p_idx < (self.pinch_off if self._pinching else self.pinch_on)
        self._pinching_mid = (not self._pinching) and mid_reach and p_mid < (
            self.pinch_off if self._pinching_mid else self.pinch_on)

        palm_pts = [lm[i] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
        palm = (sum(p[0] for p in palm_pts) / 5, sum(p[1] for p in palm_pts) / 5)

        st = HandState(fingers=f, thumb=thumb, pinch=p_idx, pinch_middle=p_mid,
                       size=size, anchor=(lm[INDEX_MCP][0], lm[INDEX_MCP][1]), palm=palm)
        st.pose = self._pose(lm, f, thumb, size)
        return st

    def _pose(self, lm, f, thumb, size):
        i, m, r, p = f["index"], f["middle"], f["ring"], f["pinky"]

        if self._pinching:
            if m and r and p:
                return "ok"
            if not r and not p:
                return "pinch"
        if self._pinching_mid and not r and not p:
            return "pinch_middle"

        # Your trained poses win over the built-in rules when the model is sure.
        if self.model is not None:
            label, conf = self.model.predict(lm)
            self.last_model = (label, conf)
            if label is not None and conf >= self.min_confidence:
                return label

        return self.rule_pose(lm, f, thumb, size)

    def rule_pose(self, lm, f, thumb, size):
        i, m, r, p = f["index"], f["middle"], f["ring"], f["pinky"]
        if i and m and r and p:
            return "palm"
        if i and m and r and not p:
            return "three"
        if i and m and not r and not p:
            return "two"
        if i and p and not m and not r:
            return "shaka" if thumb else "rock"
        if p and thumb and not i and not m and not r:
            return "shaka"
        if i and not m and not r and not p:
            return "point"
        if not (i or m or r or p):
            if thumb:
                dy = lm[THUMB_TIP][1] - lm[THUMB_MCP][1]
                dx = lm[THUMB_TIP][0] - lm[THUMB_MCP][0]
                if abs(dy) > abs(dx) * 1.2 and abs(dy) > 0.35 * size:
                    return "thumbs_up" if dy < 0 else "thumbs_down"
            return "fist"
        return "none"
