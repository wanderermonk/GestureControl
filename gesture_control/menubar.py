"""Menu bar icon (runs as a child process so it can own its own main thread).

Protocol with the parent (gesture_control.__main__):
  parent -> child  (stdin):  JSON lines  {"paused": bool, "preview": bool, "pose": str}
  child  -> parent (stdout): command lines  toggle_pause | toggle_preview | reload | quit
"""
import json
import subprocess
import sys
import threading

import rumps

from .config import DEFAULT_PATH


class GestureMenuBar(rumps.App):
    def __init__(self):
        super().__init__("GestureControl", title="✋", quit_button=None)
        self.state = {"paused": False, "preview": True, "pose": "-", "voice": "off"}
        self._dirty = True
        self._parent_gone = False

        self.pose_item = rumps.MenuItem("Pose: -")
        self.voice_item = rumps.MenuItem("Voice: off")
        self.enabled_item = rumps.MenuItem("Gesture control enabled", callback=self.on_toggle)
        self.preview_item = rumps.MenuItem("Show camera preview", callback=self.on_preview)
        self.menu = [
            self.pose_item,
            self.voice_item,
            None,
            self.enabled_item,
            self.preview_item,
            None,
            rumps.MenuItem("Edit gestures (config.yaml)", callback=self.on_open_config),
            rumps.MenuItem("Reload config", callback=lambda _: self.send("reload")),
            None,
            rumps.MenuItem("Quit GestureControl", callback=self.on_quit),
        ]
        threading.Thread(target=self._read_parent, daemon=True).start()
        rumps.Timer(self._refresh, 0.2).start()

    # parent communication ---------------------------------------------------
    def send(self, cmd):
        try:
            print(cmd, flush=True)
        except BrokenPipeError:
            self._parent_gone = True

    def _read_parent(self):
        for line in sys.stdin:
            try:
                self.state.update(json.loads(line))
                self._dirty = True
            except ValueError:
                pass
        self._parent_gone = True  # stdin closed: parent exited

    def _refresh(self, _):
        if self._parent_gone:
            rumps.quit_application()
            return
        if not self._dirty:
            return
        self._dirty = False
        paused = self.state.get("paused")
        v = self.state.get("voice", "off")
        self.title = ("⏸" if paused else "✋") + (" 🎤" if v == "listening" else "")
        self.voice_item.title = f"Voice: {v}"
        self.enabled_item.state = 0 if paused else 1
        self.preview_item.state = 1 if self.state.get("preview") else 0
        self.pose_item.title = f"Pose: {self.state.get('pose', '-')}"

    # menu callbacks -----------------------------------------------------------
    def on_toggle(self, _):
        self.send("toggle_pause")

    def on_preview(self, _):
        self.send("toggle_preview")

    def on_open_config(self, _):
        subprocess.Popen(["open", "-t", str(DEFAULT_PATH)])

    def on_quit(self, _):
        self.send("quit")


def main():
    try:  # no Dock icon for the menu bar helper
        from AppKit import NSApplication
        NSApplication.sharedApplication().setActivationPolicy_(1)
    except Exception:
        pass
    GestureMenuBar().run()


if __name__ == "__main__":
    main()
