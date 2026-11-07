import numpy as np

from gesture_control.gestures import PoseClassifier
from gesture_control.learned import GestureModel, features
from gesture_control.trainer import TrainingSession
from synth import make_hand, move

RNG = np.random.default_rng(1)


def jitter(pts, s=0.004):
    return [(x + RNG.normal(0, s), y + RNG.normal(0, s)) for x, y in pts]


def vulcan():
    """A custom pose the rules don't know: index+middle together, ring+pinky together."""
    h = make_hand(index=True, middle=True, ring=True, pinky=True, thumb=True)
    h = list(h)
    for tip, dx in ((8, 0.015), (12, -0.005), (16, 0.005), (20, -0.015)):
        h[tip] = (h[tip][0] + dx, h[tip][1])
    for pip, dx in ((6, 0.012), (10, -0.004), (14, 0.004), (18, -0.012)):
        h[pip] = (h[pip][0] + dx, h[pip][1])
    h[16] = (h[16][0] + 0.03, h[16][1])
    h[20] = (h[20][0] + 0.02, h[20][1])
    return h


POSES = {
    "point": dict(index=True),
    "two": dict(index=True, middle=True),
    "palm": dict(index=True, middle=True, ring=True, pinky=True),
    "fist": dict(),
    "thumbs_up": dict(thumb=True, thumb_dir="up"),
    "thumbs_down": dict(thumb=True, thumb_dir="down"),
}


def build_model():
    m = GestureModel()
    for name, kw in POSES.items():
        base = make_hand(**kw)
        m.add(name, np.stack([features(jitter(move(base, RNG.uniform(-.1, .1), RNG.uniform(-.1, .1))))
                              for _ in range(40)]))
    m.fit()
    return m


def test_features_invariant_to_position_and_scale():
    a = features(make_hand(index=True))
    b = features(make_hand(index=True, origin=(0.3, 0.6), scale=1.4))
    assert np.allclose(a, b, atol=1e-4)


def test_features_keep_orientation():
    up = features(make_hand(thumb=True, thumb_dir="up"))
    down = features(make_hand(thumb=True, thumb_dir="down"))
    assert np.linalg.norm(up - down) > 0.5


def test_model_predicts_trained_poses():
    m = build_model()
    for name, kw in POSES.items():
        label, conf = m.predict(jitter(make_hand(**kw, origin=(0.45, 0.75))))
        assert label == name and conf > 0.6, (name, label, conf)


def test_unknown_pose_rejected():
    m = build_model()
    # rock was never trained and is far from everything recorded
    label, _ = m.predict(make_hand(index=True, pinky=True, thumb=True))
    assert label is None


def test_custom_pose_via_classifier():
    m = build_model()
    m.add("vulcan", np.stack([features(jitter(vulcan())) for _ in range(40)]))
    m.fit()
    c = PoseClassifier(model=m)
    assert c.classify(jitter(vulcan())).pose == "vulcan"
    # pinch stays rule-based even with a model loaded
    assert PoseClassifier(model=m).classify(make_hand(pinch="index")).pose == "pinch"


def test_model_falls_back_to_rules_when_unsure():
    m = build_model()
    c = PoseClassifier(model=m)
    assert c.classify(make_hand(index=True, pinky=True)).pose == "rock"  # untrained -> rules


def test_save_load_roundtrip(tmp_path):
    m = build_model()
    p = tmp_path / "g.npz"
    m.save(p)
    m2 = GestureModel.load(p)
    assert m2.counts() == m.counts()
    assert m2.predict(make_hand(index=True))[0] == "point"


def test_evaluate_reports_high_accuracy():
    acc, _ = build_model().evaluate()
    assert min(acc.values()) > 0.9


def test_retrain_replaces_samples():
    m = build_model()
    m.remove("point")
    m.add("point", np.stack([features(make_hand(index=True))] * 10))
    assert m.counts()["point"] == 10


def test_training_session_flow():
    s = TrainingSession(["point", "palm"], record_seconds=1.0, countdown=0.5, min_samples=10)
    t, dt = 0.0, 1 / 30
    hand = make_hand(index=True)

    def run(seconds, lm, key=None):
        nonlocal t
        for i in range(int(seconds / dt)):
            t += dt
            s.step(t, lm, key if i == 0 else None)

    run(0.2, hand)                 # waiting for SPACE: nothing recorded
    assert s.state == "ready" and not s.samples["point"]
    run(0.1, hand, "space")
    run(0.6 + 1.05 + 0.55 + 1.05, hand)  # countdown, round 1, countdown, round 2
    assert s.pose == "palm" and len(s.samples["point"]) >= 50
    run(0.1, None, "skip")
    assert s.state == "done" and s.skipped == ["palm"]


def test_training_session_retries_when_hand_missing():
    s = TrainingSession(["point"], record_seconds=0.5, countdown=0.2, min_samples=10)
    t = 0.0
    s.step(t, None, "space")
    for _ in range(60):
        t += 1 / 30
        s.step(t, None)
    assert s.state == "ready" and "not visible" in s.message
