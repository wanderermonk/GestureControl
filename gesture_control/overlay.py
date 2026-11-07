"""Camera preview with the tracked hand and current status drawn on top."""
import cv2

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

MODE_COLORS = {  # BGR
    "cursor": (80, 200, 120), "drag": (60, 140, 255), "scroll": (255, 180, 60),
    "gesture": (230, 120, 230), "idle": (160, 160, 160), "paused": (60, 60, 230),
}

WINDOW = "GestureControl"


def draw(frame, landmarks, status, cfg, fps, hold_progress, now, trained=False, voice=None):
    h, w = frame.shape[:2]
    mode = status["mode"]
    color = MODE_COLORS.get(mode, (200, 200, 200))

    # active cursor region
    x0, y0, x1, y1 = cfg["cursor"]["region"]
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (90, 90, 90), 1)

    if landmarks:
        pts = [(int(x * w), int(y * h)) for x, y in landmarks]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 5 if i in (4, 8, 12) else 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pts[5], 8, color, 2, cv2.LINE_AA)  # cursor anchor

        if hold_progress > 0:
            c = pts[9]
            cv2.ellipse(frame, c, (28, 28), -90, 0, int(360 * hold_progress), (255, 255, 255), 4, cv2.LINE_AA)

    # status panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    label = "PAUSED  (hold 'rock' or press P)" if status["paused"] else f"{mode.upper()}  |  {status['pose']}"
    cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if trained:
        cv2.putText(frame, "trained", (w - 100, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.0f} fps", (w - 100, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if status["last_action"] and now - status["last_action_t"] < 2.0:
        cv2.putText(frame, status["last_action"], (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Q quit  P pause  H hide", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)

    if voice and voice["state"] != "off":
        _draw_voice(frame, voice, now)

    pw = cfg["ui"]["preview_width"]
    if pw and pw != w:
        frame = cv2.resize(frame, (pw, int(h * pw / w)))
    return frame


VOICE_COLORS = {"on": (160, 160, 160), "listening": (80, 220, 255), "muted": (90, 90, 90)}


def _draw_voice(frame, v, now):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    st = v["state"]
    color = VOICE_COLORS.get(st, (200, 200, 200))
    label = "MIC: listening %.0fs" % max(0, v["until"] - now) if st == "listening" else f"MIC: {st}"
    cv2.circle(frame, (16, h - 18), 6, color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (30, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    ev = v["event"]
    msg = ""
    if v.get("partial") and st == "listening":
        msg = v["partial"] + " ..."
    elif ev and now - v["event_t"] < 3.0:
        if ev["type"] == "command":
            msg = f'"{ev["text"]}" -> {ev["action"]}'
        elif ev["type"] == "rejected":
            msg = f'"{ev["text"]}": {ev["reason"]}'
    if msg:
        cv2.putText(frame, msg[:60], (190, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
