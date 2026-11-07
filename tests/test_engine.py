from gesture_control.actions import ActionRunner, DryRunBackend
from gesture_control.config import load_config
from gesture_control.engine import GestureEngine
from gesture_control.gestures import PoseClassifier
from synth import make_hand, move

FPS = 30
DT = 1 / FPS


class SyncRunner(ActionRunner):
    """Runs actions synchronously so tests can inspect them."""

    def run(self, spec):
        kind, _, value = spec.partition(":")
        if kind == "builtin":
            return super().run(spec)
        self._safe(kind, value)


class Rig:
    def __init__(self):
        self.cfg = load_config()
        self.backend = DryRunBackend(log=False)
        self.actions = SyncRunner(self.backend, {"toggle_pause": self.toggle})
        self.engine = GestureEngine(self.cfg, self.backend, self.actions)
        self.clf = PoseClassifier()
        self.t = 0.0

    def toggle(self):
        self.engine.set_paused(not self.engine.paused)

    def feed(self, hand, seconds=None, frames=None):
        n = frames if frames is not None else max(1, round(seconds * FPS))
        st = None
        for _ in range(n):
            self.t += DT
            st = self.engine.process(self.clf.classify(hand), self.t)
        return st

    def calls(self, kind):
        return [c for c in self.backend.calls if c[0] == kind]


def test_point_moves_cursor():
    r = Rig()
    r.feed(make_hand(index=True), 0.3)
    r.feed(make_hand(index=True, origin=(0.6, 0.8)), 1.0)
    moves = r.calls("move")
    assert moves, "cursor should move"
    assert moves[-1][1] > moves[0][1], "hand right -> cursor right"


def test_quick_pinch_clicks_once():
    r = Rig()
    r.feed(make_hand(index=True), 0.3)
    r.feed(make_hand(pinch="index"), 0.15)
    r.feed(make_hand(index=True), 0.2)
    clicks = r.calls("click")
    assert len(clicks) == 1 and clicks[0][3] == "left" and clicks[0][4] == 1
    assert not r.calls("mouse_down")


def test_double_click():
    r = Rig()
    r.feed(make_hand(index=True), 0.3)
    for _ in range(2):
        r.feed(make_hand(pinch="index"), 0.1)
        r.feed(make_hand(index=True), 0.1)
    counts = [c[4] for c in r.calls("click")]
    assert counts == [1, 2]


def test_long_pinch_drags():
    r = Rig()
    r.feed(make_hand(index=True), 0.3)
    r.feed(make_hand(pinch="index"), 0.5)
    r.feed(make_hand(pinch="index", origin=(0.6, 0.75)), 0.5)
    r.feed(make_hand(index=True, origin=(0.6, 0.75)), 0.2)
    assert len(r.calls("mouse_down")) == 1
    assert r.calls("drag")
    assert len(r.calls("mouse_up")) == 1
    assert not r.calls("click")


def test_hand_lost_while_dragging_releases():
    r = Rig()
    r.feed(make_hand(index=True), 0.3)
    r.feed(make_hand(pinch="index"), 0.6)
    r.feed(None, 0.5)
    assert len(r.calls("mouse_up")) == 1


def test_right_click():
    r = Rig()
    r.feed(make_hand(index=True), 0.3)
    r.feed(make_hand(index=True, pinch="middle"), 0.5)
    r.feed(make_hand(index=True), 0.2)
    rc = [c for c in r.calls("click") if c[3] == "right"]
    assert len(rc) == 1


def test_scroll():
    r = Rig()
    r.feed(make_hand(index=True, middle=True), 0.2)
    h = make_hand(index=True, middle=True)
    for i in range(15):  # move hand up
        r.feed(move(h, 0, -0.01 * i), frames=1)
    s = r.calls("scroll")
    assert s and all(c[1] > 0 for c in s), "hand up -> scroll up"


def swipe(r, pose_kw, dx, dy, frames=8):
    h = make_hand(**pose_kw)
    r.feed(h, 0.2)
    for i in range(1, frames + 1):
        r.feed(move(h, dx * i / frames, dy * i / frames), frames=1)
    r.feed(move(h, dx, dy), 0.1)


def test_palm_swipes():
    palm = dict(index=True, middle=True, ring=True, pinky=True)
    for (dx, dy), expected in [((0.3, 0), "space_right"), ((-0.3, 0), "space_left"),
                               ((0, -0.3), "mission_control"), ((0, 0.3), "app_expose")]:
        r = Rig()
        swipe(r, palm, dx, dy)
        assert r.calls("system") == [("system", expected)], (dx, dy)


def test_three_swipe_snaps_window():
    r = Rig()
    swipe(r, dict(index=True, middle=True, ring=True), -0.3, 0)
    assert r.calls("window") == [("window", "snap_left")]


def test_swipe_does_not_also_fire_hold():
    r = Rig()
    swipe(r, dict(index=True, middle=True, ring=True), 0.3, 0)
    r.feed(move(make_hand(index=True, middle=True, ring=True), 0.3, 0), 2.0)
    assert r.calls("system") == []  # no screenshot


def test_hold_and_repeat_volume():
    r = Rig()
    r.feed(make_hand(thumb=True, thumb_dir="up"), 1.6)
    n = len(r.calls("media"))
    # fires at ~0.5s+stabilise, then every 0.25s -> about 5
    assert 4 <= n <= 6
    assert all(c == ("media", "volume_up") for c in r.calls("media"))


def test_still_hold_screenshot_once():
    r = Rig()
    r.feed(make_hand(index=True, middle=True, ring=True), 3.0)
    assert r.calls("system") == [("system", "screenshot")]


def test_fist_does_nothing():
    r = Rig()
    r.feed(make_hand(), 3.0)
    assert r.backend.calls == []


def test_pause_and_resume_with_rock():
    r = Rig()
    rock = make_hand(index=True, pinky=True)
    r.feed(rock, 1.3)
    assert r.engine.paused
    r.feed(make_hand(), 0.2)
    # while paused nothing happens
    r.feed(make_hand(index=True), 0.5)
    r.feed(make_hand(pinch="index"), 0.1)
    r.feed(make_hand(index=True), 0.2)
    r.feed(make_hand(thumb=True, thumb_dir="up"), 1.5)
    assert not r.calls("click") and not r.calls("media") and not r.calls("move")
    # resume
    r.feed(rock, 1.3)
    assert not r.engine.paused
