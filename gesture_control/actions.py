"""Action dispatcher: turns "type:value" strings from config.yaml into backend calls."""
import sys
import threading


class DryRunBackend:
    """Prints instead of acting. Used on non-macOS systems and with --dry-run."""

    def __init__(self, size=(1440, 900), log=True):
        self.size = size
        self.log = log
        self.calls = []

    def _rec(self, *call):
        self.calls.append(call)
        if self.log and call[0] not in ("move", "drag"):
            print("[dry-run]", *call)

    def screen_size(self):
        return self.size

    def move(self, x, y): self._rec("move", x, y)
    def drag(self, x, y): self._rec("drag", x, y)
    def mouse_down(self, x, y): self._rec("mouse_down", x, y)
    def mouse_up(self, x, y): self._rec("mouse_up", x, y)
    def click(self, x, y, button="left", count=1): self._rec("click", x, y, button, count)
    def scroll(self, dy, dx=0): self._rec("scroll", dy, dx)
    def hotkey(self, combo): self._rec("hotkey", combo)
    def media(self, name): self._rec("media", name)
    def window(self, cmd): self._rec("window", cmd)
    def system(self, cmd): self._rec("system", cmd)
    def open_app(self, name): self._rec("app", name)
    def cursor_pos(self): return (100, 100)


def make_backend(dry_run=False):
    if dry_run or sys.platform != "darwin":
        if sys.platform != "darwin" and not dry_run:
            print("Not running on macOS: using dry-run backend (actions are printed only).")
        return DryRunBackend()
    from .backend_macos import MacBackend
    return MacBackend()


class ActionRunner:
    def __init__(self, backend, builtins=None, scroll_step=400):
        self.backend = backend
        self.builtins = builtins or {}
        self.scroll_step = scroll_step

    def run(self, spec: str):
        spec = (spec or "").strip()
        if not spec or spec == "none":
            return
        kind, _, value = spec.partition(":")
        kind, value = kind.strip().lower(), value.strip()
        if kind == "builtin":
            fn = self.builtins.get(value)
            if fn is None:
                print(f"[action] unknown builtin '{value}'")
            else:
                fn()
            return
        # Window/AppleScript calls can take ~100 ms: keep them off the camera loop.
        threading.Thread(target=self._safe, args=(kind, value), daemon=True).start()

    def _mouse(self, cmd):
        x, y = self.backend.cursor_pos()
        if cmd == "click":
            self.backend.click(x, y, "left", 1)
        elif cmd == "double_click":
            self.backend.click(x, y, "left", 1)
            self.backend.click(x, y, "left", 2)
        elif cmd == "right_click":
            self.backend.click(x, y, "right", 1)
        elif cmd == "scroll_up":
            self.backend.scroll(self.scroll_step)
        elif cmd == "scroll_down":
            self.backend.scroll(-self.scroll_step)
        else:
            raise ValueError(f"unknown mouse command '{cmd}'")

    def _safe(self, kind, value):
        try:
            if kind == "hotkey":
                self.backend.hotkey(value)
            elif kind == "media":
                self.backend.media(value)
            elif kind == "window":
                self.backend.window(value)
            elif kind == "system":
                self.backend.system(value)
            elif kind == "app":
                self.backend.open_app(value)
            elif kind == "mouse":
                self._mouse(value)
            else:
                print(f"[action] unknown action type '{kind}' in '{kind}:{value}'")
        except Exception as e:  # never crash the tracking loop
            print(f"[action] {kind}:{value} failed: {e}")
