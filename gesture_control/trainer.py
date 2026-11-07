"""Guided training: record your own hand poses and build a personal model.

  ./run.sh --train                   record all the built-in poses
  ./run.sh --train point palm        re-record just these
  ./run.sh --train vulcan            add a brand-new pose called "vulcan"
  ./run.sh --list-trained            show what's in your model
  ./run.sh --forget vulcan           delete one pose from your model
  ./run.sh --reset-training          delete your model (back to built-in rules)

Only landmark coordinates are recorded, never camera images.
"""
import re
import time

import cv2
import numpy as np

from .config import load_config
from .learned import MODEL_FILE, RULE_ONLY, GestureModel, features
from .overlay import CONNECTIONS
from .tracker import Camera, HandTracker

BUILTIN = [
    ("point", "Index finger up, other fingers folded"),
    ("two", "Index + middle finger up (peace sign)"),
    ("three", "Index + middle + ring finger up"),
    ("palm", "Open hand, all fingers up"),
    ("fist", "Closed fist"),
    ("thumbs_up", "Thumbs up"),
    ("thumbs_down", "Thumbs down"),
    ("shaka", "Thumb + pinky out ('call me')"),
    ("rock", "Index + pinky up ('rock on')"),
]
DESCRIPTIONS = dict(BUILTIN)

ROUNDS = [
    "Hold the pose steady, centred in front of the camera",
    "Keep the pose; slowly move it around and tilt it a little",
]

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


def check_name(name):
    if not NAME_RE.match(name):
        raise SystemExit(f"Pose name '{name}' must be lowercase letters, digits or _ (e.g. vulcan, two_up).")
    if name in RULE_ONLY:
        raise SystemExit(f"'{name}' is detected by fixed rules (pinch distance) and can't be trained.")


class TrainingSession:
    """Frame-driven state machine; no camera or window code, so it's testable."""

    def __init__(self, poses, record_seconds=2.0, countdown=2.0, min_samples=25):
        self.poses = list(poses)
        self.record_seconds = record_seconds
        self.countdown = countdown
        self.min_samples = min_samples
        self.i = 0
        self.round = 0
        self.state = "ready"   # ready -> countdown -> record -> (next round | next pose) ... -> done
        self.t_state = 0.0
        self.samples = {p: [] for p in self.poses}
        self.skipped = []
        self.message = ""

    @property
    def pose(self):
        return self.poses[self.i] if self.i < len(self.poses) else None

    def _goto(self, state, t):
        self.state, self.t_state = state, t

    def _next_pose(self, t):
        self.i += 1
        self.round = 0
        self._goto("done" if self.i >= len(self.poses) else "ready", t)

    def step(self, t, landmarks, key=None):
        """Advance one frame. key: 'space', 'skip', 'redo', or None."""
        if self.state == "done":
            return
        if self.state == "ready":
            if key == "space":
                self._goto("countdown", t)
            elif key == "skip":
                self.skipped.append(self.pose)
                self.samples[self.pose] = []
                self._next_pose(t)
            return
        if key == "redo":  # start this pose over
            self.samples[self.pose] = []
            self.round = 0
            self._goto("ready", t)
            return
        if self.state == "countdown":
            if t - self.t_state >= self.countdown:
                self._goto("record", t)
            return
        if self.state == "record":
            if landmarks is not None:
                self.samples[self.pose].append(features(landmarks))
            if t - self.t_state >= self.record_seconds:
                self.round += 1
                if self.round < len(ROUNDS):
                    self._goto("countdown", t)
                elif len(self.samples[self.pose]) < self.min_samples:
                    # hand wasn't visible enough: try again
                    self.message = f"Only {len(self.samples[self.pose])} samples - hand not visible. Again!"
                    self.samples[self.pose] = []
                    self.round = 0
                    self._goto("ready", t)
                else:
                    self.message = f"Recorded {len(self.samples[self.pose])} samples of '{self.pose}'"
                    self._next_pose(t)

    def progress(self, t):
        if self.state == "countdown":
            return max(0.0, self.countdown - (t - self.t_state))
        if self.state == "record":
            return min(1.0, (t - self.t_state) / self.record_seconds)
        return 0.0


def _draw(frame, landmarks, session, t):
    h, w = frame.shape[:2]
    recording = session.state == "record"
    color = (60, 60, 230) if recording else (80, 200, 120)
    if landmarks:
        pts = [(int(x * w), int(y * h)) for x, y in landmarks]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    def text(s, y, scale=0.6, c=(255, 255, 255), thick=1):
        cv2.putText(frame, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, thick, cv2.LINE_AA)

    if session.state == "done":
        text("Done! Building your model...", 40, 0.8, (80, 200, 120), 2)
        return frame
    pose = session.pose
    n = f"{session.i + 1}/{len(session.poses)}"
    text(f"[{n}]  {pose.upper()}", 32, 0.85, (255, 220, 120), 2)
    text(DESCRIPTIONS.get(pose, "Your custom pose"), 58)
    if session.state == "ready":
        text("SPACE = start    S = skip    Q = quit (saves what's recorded)", 86, 0.5, (180, 180, 180))
        if session.message:
            text(session.message, 104, 0.45, (120, 200, 255))
    elif session.state == "countdown":
        text(ROUNDS[session.round], 86, 0.55, (120, 200, 255))
        secs = session.progress(t)
        cv2.putText(frame, f"{secs:.0f}" if secs >= 0.5 else "GO", (w // 2 - 30, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
    elif session.state == "record":
        text(f"RECORDING  ({ROUNDS[session.round].split(';')[0].lower()})", 86, 0.55, (60, 60, 230), 2)
        cv2.rectangle(frame, (14, 96), (w - 14, 104), (60, 60, 60), -1)
        cv2.rectangle(frame, (14, 96), (14 + int((w - 28) * session.progress(t)), 104), (60, 60, 230), -1)
        if not landmarks:
            text("No hand visible!", h - 20, 0.7, (60, 60, 230), 2)
        cv2.circle(frame, (w - 30, 30), 10, (60, 60, 230), -1)
    return frame


def report(model):
    print("\nYour gesture model:")
    for label, n in model.counts().items():
        print(f"  {label:<14} {n:4d} samples")
    ev = model.evaluate()
    if ev is None:
        print("  (record at least two poses to measure accuracy)")
        return
    acc, conf = ev
    print("\nEstimated accuracy (cross-validated):")
    weak = []
    for label, a in acc.items():
        mark = "" if a >= 0.9 else "   <- often confused, consider re-recording"
        print(f"  {label:<14} {a * 100:5.1f}%{mark}")
        if a < 0.9:
            weak.append(label)
    if conf:
        print("\nMost common mix-ups:")
        for (a, b), n in sorted(conf.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {a} -> {b}: {n}")
    if weak:
        print(f"\nTip: ./run.sh --train {' '.join(weak)}  (make these poses more distinct)")


def run_training(pose_names, config_path=None):
    cfg = load_config(config_path)
    if not pose_names:
        pose_names = [p for p, _ in BUILTIN]
    for p in pose_names:
        check_name(p)

    model = GestureModel.load(MODEL_FILE) or GestureModel()
    cam_cfg = cfg["camera"]
    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    tracker = HandTracker(cfg["tracking"])
    session = TrainingSession(pose_names)
    win = "GestureControl - training"
    print("Training window open. Use the hand you'll control the Mac with.")

    try:
        while session.state != "done":
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if cam_cfg["mirror"]:
                frame = cv2.flip(frame, 1)
            lm = tracker.detect(frame)
            k = cv2.waitKey(1) & 0xFF
            key = {ord(" "): "space", ord("s"): "skip", ord("r"): "redo"}.get(k)
            if k in (ord("q"), 27):
                break
            t = time.monotonic()
            session.step(t, lm, key)
            cv2.imshow(win, _draw(frame, lm, session, t))
    finally:
        cam.close()
        tracker.close()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    recorded = {p: s for p, s in session.samples.items() if len(s) >= session.min_samples}
    if not recorded:
        print("Nothing recorded; your model was not changed.")
        return
    for p, s in recorded.items():
        model.remove(p)  # re-recording a pose replaces it
        model.add(p, np.stack(s))
    model.fit()
    model.save(MODEL_FILE)
    print(f"\nSaved {MODEL_FILE}  (landmark numbers only, no images)")
    report(model)

    custom = [p for p in recorded if p not in DESCRIPTIONS]
    if custom:
        print("\nMap your new pose(s) in config.yaml, e.g.:")
        for p in custom:
            print(f'  {p}_hold: "media:mute"')
            print(f'  {p}_swipe_left: "hotkey:cmd+["')


def list_trained():
    model = GestureModel.load(MODEL_FILE)
    if model is None or not len(model.y):
        print("No personal model yet. Run:  ./run.sh --train")
        return
    report(model)


def forget(names):
    model = GestureModel.load(MODEL_FILE)
    if model is None:
        print("No personal model yet.")
        return
    for n in names:
        if n not in model.labels:
            print(f"'{n}' is not in your model.")
        model.remove(n)
    model.fit()
    model.save(MODEL_FILE)
    print(f"Removed {', '.join(names)}.")


def reset():
    if MODEL_FILE.exists():
        MODEL_FILE.unlink()
        print("Personal model deleted; using the built-in rules.")
    else:
        print("No personal model to delete.")
