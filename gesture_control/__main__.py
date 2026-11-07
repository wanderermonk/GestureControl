"""GestureControl: control your Mac with hand gestures.

Run:  python -m gesture_control [--no-preview] [--no-menubar] [--dry-run] [--config PATH]
Train: python -m gesture_control --train [POSE ...]   (see trainer.py)
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time

import cv2

from . import overlay
from .actions import ActionRunner, make_backend
from .config import DEFAULT_PATH, ROOT, load_config
from .engine import GestureEngine
from .gestures import PoseClassifier
from .learned import GestureModel
from .tracker import Camera, HandTracker


class MenuBarBridge:
    """Starts menubar.py as a child process and relays messages."""

    def __init__(self):
        self.commands = queue.Queue()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "gesture_control.menubar"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=str(ROOT))
        threading.Thread(target=self._read, daemon=True).start()
        self._last = None

    def _read(self):
        for line in self.proc.stdout:
            self.commands.put(line.strip())

    def send(self, state):
        if state == self._last or self.proc.poll() is not None:
            return
        self._last = state
        try:
            self.proc.stdin.write(json.dumps(state) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser(description="Control your Mac with hand gestures.")
    ap.add_argument("--config", default=str(DEFAULT_PATH))
    ap.add_argument("--no-preview", action="store_true", help="start with the preview hidden")
    ap.add_argument("--no-menubar", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print actions instead of performing them")
    ap.add_argument("--start-paused", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="ignore your trained model; use built-in rules")
    tr = ap.add_argument_group("training")
    tr.add_argument("--train", nargs="*", metavar="POSE",
                    help="record your own poses (no names = all built-in poses; new names = new poses)")
    tr.add_argument("--list-trained", action="store_true", help="show your trained poses and accuracy")
    tr.add_argument("--forget", nargs="+", metavar="POSE", help="remove poses from your model")
    tr.add_argument("--reset-training", action="store_true", help="delete your model")
    vo = ap.add_argument_group("voice")
    vo.add_argument("--no-voice", action="store_true", help="run without voice commands")
    vo.add_argument("--train-voice", nargs="*", metavar="PHRASE",
                    help="teach it your voice & pronunciation (no phrases = wake word + all commands)")
    vo.add_argument("--reps", type=int, default=3, help="repetitions per phrase when training voice")
    vo.add_argument("--voice-test", action="store_true", help="show what it hears without acting")
    vo.add_argument("--voice-commands", action="store_true", help="list voice commands")
    vo.add_argument("--reset-voice", action="store_true", help="forget voice training")
    args = ap.parse_args()

    if args.train_voice is not None or args.voice_test or args.voice_commands or args.reset_voice:
        from . import voice_trainer
        if args.reset_voice:
            voice_trainer.reset_voice()
        elif args.voice_commands:
            voice_trainer.list_commands(args.config)
        elif args.voice_test:
            voice_trainer.voice_test(args.config)
        else:
            voice_trainer.train_voice(args.train_voice, args.reps, args.config)
        return

    if args.train is not None or args.list_trained or args.forget or args.reset_training:
        from . import trainer
        if args.reset_training:
            trainer.reset()
        elif args.forget:
            trainer.forget(args.forget)
        elif args.list_trained:
            trainer.list_trained()
        else:
            trainer.run_training(args.train, args.config)
        return

    cfg = load_config(args.config)
    backend = make_backend(args.dry_run)
    on_mac = sys.platform == "darwin"

    if on_mac and not args.dry_run:
        from .backend_macos import accessibility_trusted
        if not accessibility_trusted(prompt=True):
            print("\n!! Accessibility permission is missing, so the cursor and keys can't be controlled.\n"
                  "   System Settings > Privacy & Security > Accessibility -> enable your terminal app,\n"
                  "   then quit and reopen the terminal and run this again.\n")

    state = {"preview": cfg["ui"]["preview"] and not args.no_preview, "running": True}
    engine = None
    voice = {"ctrl": None, "listener": None, "event": None, "event_t": -1e9}

    def toggle_pause():
        engine.set_paused(not engine.paused)
        print("Gestures paused" if engine.paused else "Gestures resumed")

    def set_paused(p):
        engine.set_paused(p)
        print("Gestures paused" if p else "Gestures resumed")

    def voice_listen():
        if voice["ctrl"] and not voice["ctrl"].muted and voice["ctrl"].gesture_listen(time.monotonic()):
            print("[voice] listening ...")

    def voice_mute(m):
        if voice["ctrl"]:
            voice["ctrl"].muted = m
            print("[voice] muted - say 'computer, start listening'" if m else "[voice] on")

    actions = ActionRunner(backend, builtins={
        "toggle_pause": toggle_pause,
        "pause": lambda: set_paused(True),
        "resume": lambda: set_paused(False),
        "voice_listen": voice_listen,
        "voice_off": lambda: voice_mute(True),
        "voice_on": lambda: voice_mute(False),
    })
    engine = GestureEngine(cfg, backend, actions)
    engine.set_paused(args.start_paused)
    def make_classifier():
        model = None
        if cfg["tracking"]["use_trained_model"] and not args.no_model:
            model = GestureModel.load()
            if model is not None and len(model.y):
                print(f"Using your trained poses: {', '.join(model.labels)}")
            else:
                model = None
        return PoseClassifier(cfg["tracking"]["pinch_on"], cfg["tracking"]["pinch_off"],
                              model=model, min_confidence=cfg["tracking"]["trained_min_confidence"])

    classifier = make_classifier()

    cam_cfg = cfg["camera"]
    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    tracker = HandTracker(cfg["tracking"])

    def start_voice():
        if args.no_voice or not cfg["voice"]["enabled"]:
            return
        try:
            from .voice import VoiceController, VoiceListener, load_profile
        except ImportError as e:
            print(f"[voice] unavailable ({e}); run setup.sh again")
            return
        ctrl = VoiceController(cfg["voice"], load_profile())
        voice["ctrl"] = ctrl
        if voice["listener"] is None or not voice["listener"].is_alive():
            voice["listener"] = VoiceListener(cfg["voice"], ctrl.grammar())
            voice["listener"].start()
        else:
            voice["listener"].set_grammar(ctrl.grammar())
        vid = "on" if ctrl.check_speaker else "off (run --train-voice)"
        print(f"[voice] wake word: '{ctrl.wake_words[0] if ctrl.wake_words else '-'}', "
              f"activation: {cfg['voice']['activation']}, only my voice: {vid}")

    start_voice()

    menubar = None
    if on_mac and cfg["ui"]["menubar"] and not args.no_menubar:
        try:
            menubar = MenuBarBridge()
        except Exception as e:
            print(f"Menu bar icon unavailable: {e}")

    def reload_config():
        nonlocal cfg, classifier
        try:
            new = load_config(args.config)
        except Exception as e:
            print(f"Config error, keeping old config: {e}")
            return
        cfg = new
        engine.update_config(cfg)
        classifier = make_classifier()
        start_voice()
        print("Config reloaded")

    def handle(cmd):
        if cmd == "toggle_pause":
            toggle_pause()
        elif cmd == "toggle_preview":
            state["preview"] = not state["preview"]
        elif cmd == "reload":
            reload_config()
        elif cmd == "quit":
            state["running"] = False

    print("GestureControl running. Point with your index finger to move the cursor, pinch to click.\n"
          "Hold the 'rock' sign (index + pinky) for 1s to pause/resume. Ctrl+C or Q in the preview to quit.")

    window_open = False
    fps, last_t = 0.0, time.monotonic()
    try:
        while state["running"]:
            if menubar:
                while not menubar.commands.empty():
                    handle(menubar.commands.get())
                if menubar.proc.poll() is not None:
                    menubar = None

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if cam_cfg["mirror"]:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            landmarks = tracker.detect(frame)
            hand = classifier.classify(landmarks)
            status = engine.process(hand, now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # voice commands heard since last frame
            lis, vctrl = voice["listener"], voice["ctrl"]
            if lis is not None:
                while not lis.results.empty():
                    ev = vctrl.process(lis.results.get(), now)
                    if ev is None:
                        continue
                    voice["event"], voice["event_t"] = ev, now
                    if ev["type"] == "command":
                        print(f"[voice] \"{ev['text']}\" -> {ev['action']}")
                        actions.run(ev["action"])
                    elif ev["type"] == "listening":
                        print("[voice] listening ...")
                    else:
                        print(f"[voice] \"{ev['text']}\" ignored: {ev['reason']}")
                if lis.error is not None:
                    voice["listener"] = None
            vstate = "off"
            if voice["listener"] is not None and vctrl is not None:
                vstate = "muted" if vctrl.muted else ("listening" if vctrl.listening(now) else "on")

            if menubar:
                menubar.send({"paused": engine.paused, "preview": state["preview"], "pose": status["pose"],
                              "voice": vstate})

            if state["preview"]:
                img = overlay.draw(frame, landmarks, status, cfg, fps, engine.hold_progress(now), now,
                                   trained=classifier.model is not None,
                                   voice={"state": vstate, "event": voice["event"], "event_t": voice["event_t"],
                                          "partial": voice["listener"].partial if voice["listener"] else "",
                                          "until": vctrl.listen_until if vctrl else 0})
                cv2.imshow(overlay.WINDOW, img)
                window_open = True
            elif window_open:
                cv2.destroyWindow(overlay.WINDOW)
                window_open = False

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                toggle_pause()
            elif key == ord("h"):
                state["preview"] = False
            elif window_open and cv2.getWindowProperty(overlay.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                state["preview"] = False  # user closed the window: just hide it
                window_open = False
    except KeyboardInterrupt:
        pass
    finally:
        engine.release()
        if voice["listener"]:
            voice["listener"].stop()
        cam.close()
        tracker.close()
        cv2.destroyAllWindows()
        if menubar:
            menubar.close()
        print("GestureControl stopped.")


if __name__ == "__main__":
    main()
