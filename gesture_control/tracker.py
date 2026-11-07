"""Webcam capture + MediaPipe HandLandmarker (Tasks API)."""
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .config import ROOT

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    return path


class Camera:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if hasattr(cv2, "CAP_AVFOUNDATION") \
            else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. On macOS, allow camera access for your "
                "terminal app in System Settings > Privacy & Security > Camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class HandTracker:
    def __init__(self, tracking_cfg):
        model = ensure_model()
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=tracking_cfg["min_detection_confidence"],
            min_hand_presence_confidence=tracking_cfg["min_presence_confidence"],
            min_tracking_confidence=tracking_cfg["min_tracking_confidence"],
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ms = -1

    def detect(self, frame_bgr):
        """Return a list of 21 (x, y) tuples in image coords (0..1), or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last_ms:  # timestamps must strictly increase
            ts = self._last_ms + 1
        self._last_ms = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        return [(p.x, p.y) for p in result.hand_landmarks[0]]

    def close(self):
        self.landmarker.close()
