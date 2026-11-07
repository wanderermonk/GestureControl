import pytest

from gesture_control.gestures import PoseClassifier
from synth import make_hand

CASES = [
    (dict(index=True), "point"),
    (dict(index=True, thumb=True), "point"),
    (dict(index=True, middle=True), "two"),
    (dict(index=True, middle=True, ring=True), "three"),
    (dict(index=True, middle=True, ring=True, pinky=True), "palm"),
    (dict(index=True, middle=True, ring=True, pinky=True, thumb=True), "palm"),
    (dict(), "fist"),
    (dict(thumb=True, thumb_dir="up"), "thumbs_up"),
    (dict(thumb=True, thumb_dir="down"), "thumbs_down"),
    (dict(pinky=True, thumb=True), "shaka"),
    (dict(index=True, pinky=True), "rock"),
    (dict(pinch="index"), "pinch"),
    (dict(index=True, pinch="middle"), "pinch_middle"),
    (dict(middle=True, ring=True, pinky=True, pinch="index"), "ok"),
]


@pytest.mark.parametrize("kw,expected", CASES)
def test_poses(kw, expected):
    assert PoseClassifier().classify(make_hand(**kw)).pose == expected


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.5])
def test_scale_invariant(scale):
    c = PoseClassifier()
    assert c.classify(make_hand(index=True, middle=True, scale=scale)).pose == "two"
    assert PoseClassifier().classify(make_hand(pinch="index", scale=scale)).pose == "pinch"


def test_no_hand():
    assert PoseClassifier().classify(None).pose == "none"


def test_pinch_hysteresis():
    c = PoseClassifier(pinch_on=0.28, pinch_off=0.40)
    hand = make_hand(index=True)
    # thumb tip moved towards index tip: distance between on and off thresholds
    size = 0.15
    tip = hand[8]

    def at(d):
        h = list(hand)
        h[4] = (tip[0] - d * size, tip[1])
        return h

    assert c.classify(at(0.35)).pose == "point"   # not yet pinching
    assert c.classify(at(0.20)).pose == "pinch"   # pinch starts
    assert c.classify(at(0.35)).pose == "pinch"   # still held (hysteresis)
    assert c.classify(at(0.50)).pose == "point"   # released
