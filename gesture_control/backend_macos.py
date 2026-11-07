"""macOS input backend: Quartz events for mouse/keyboard, NSEvent for media keys,
System Events (AppleScript) for window placement.

Needs Accessibility permission for the app that runs Python (Terminal, iTerm,
VS Code...): System Settings > Privacy & Security > Accessibility.
"""
import subprocess

import Quartz
from AppKit import NSEvent, NSScreen

# --------------------------------------------------------------------- keys
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "return": 36,
    "enter": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43,
    "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50,
    "delete": 51, "backspace": 51, "escape": 53, "esc": 53, "forwarddelete": 117,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111, "home": 115, "end": 119,
    "pageup": 116, "pagedown": 121, "left": 123, "right": 124, "down": 125, "up": 126,
}
ARROWS = {123, 124, 125, 126}
NAV_KEYS = {115, 116, 117, 119, 121} | ARROWS | {122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111}

MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand, "command": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift,
    "ctrl": Quartz.kCGEventFlagMaskControl, "control": Quartz.kCGEventFlagMaskControl,
    "alt": Quartz.kCGEventFlagMaskAlternate, "opt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "fn": Quartz.kCGEventFlagMaskSecondaryFn,
}
MODIFIER_KEYCODES = {
    Quartz.kCGEventFlagMaskCommand: 55, Quartz.kCGEventFlagMaskShift: 56,
    Quartz.kCGEventFlagMaskAlternate: 58, Quartz.kCGEventFlagMaskControl: 59,
    Quartz.kCGEventFlagMaskSecondaryFn: 63,
}

# NX_KEYTYPE_* values for media keys
MEDIA_KEYS = {
    "volume_up": 0, "volume_down": 1, "brightness_up": 2, "brightness_down": 3,
    "mute": 7, "play_pause": 16, "next": 17, "previous": 18,
}
NSSystemDefined = 14


def _post(ev):
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


class MacBackend:
    # ------------------------------------------------------------- display
    def screen_size(self):
        b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(b.size.width), int(b.size.height)

    # --------------------------------------------------------------- mouse
    def cursor_pos(self):
        loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return int(loc.x), int(loc.y)

    def _mouse(self, etype, x, y, button=Quartz.kCGMouseButtonLeft, clicks=1):
        ev = Quartz.CGEventCreateMouseEvent(None, etype, (x, y), button)
        if clicks > 1:
            Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, clicks)
        _post(ev)

    def move(self, x, y):
        self._mouse(Quartz.kCGEventMouseMoved, x, y)

    def drag(self, x, y):
        self._mouse(Quartz.kCGEventLeftMouseDragged, x, y)

    def mouse_down(self, x, y):
        self._mouse(Quartz.kCGEventLeftMouseDown, x, y)

    def mouse_up(self, x, y):
        self._mouse(Quartz.kCGEventLeftMouseUp, x, y)

    def click(self, x, y, button="left", count=1):
        if button == "right":
            down, up, btn = Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp, Quartz.kCGMouseButtonRight
        else:
            down, up, btn = Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp, Quartz.kCGMouseButtonLeft
        self._mouse(down, x, y, btn, count)
        self._mouse(up, x, y, btn, count)

    def scroll(self, dy, dx=0):
        ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, int(dy), int(dx))
        _post(ev)

    # ------------------------------------------------------------ keyboard
    def hotkey(self, combo: str):
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        if not parts:
            return
        *mods, key = parts
        if key not in KEYCODES:
            raise ValueError(f"Unknown key '{key}' in hotkey '{combo}'")
        code = KEYCODES[key]
        flags = 0
        for m in mods:
            if m not in MODIFIERS:
                raise ValueError(f"Unknown modifier '{m}' in hotkey '{combo}'")
            flags |= MODIFIERS[m]
        if code in NAV_KEYS:
            # Real arrow/function key presses carry these flags; without them
            # shortcuts like ctrl+left (switch Space) are ignored by macOS.
            flags |= Quartz.kCGEventFlagMaskSecondaryFn
            if code in ARROWS:
                flags |= Quartz.kCGEventFlagMaskNumericPad

        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        held = [f for f in MODIFIER_KEYCODES if flags & f]
        acc = 0
        for f in held:  # press modifiers
            acc |= f
            ev = Quartz.CGEventCreateKeyboardEvent(src, MODIFIER_KEYCODES[f], True)
            Quartz.CGEventSetFlags(ev, acc)
            _post(ev)
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(src, code, down)
            Quartz.CGEventSetFlags(ev, flags)
            _post(ev)
        for f in reversed(held):  # release modifiers
            acc &= ~f
            ev = Quartz.CGEventCreateKeyboardEvent(src, MODIFIER_KEYCODES[f], False)
            Quartz.CGEventSetFlags(ev, acc)
            _post(ev)

    def media(self, name: str):
        if name not in MEDIA_KEYS:
            raise ValueError(f"Unknown media key '{name}'")
        key = MEDIA_KEYS[name]
        for down in (True, False):
            state = 0xA if down else 0xB
            ev = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                NSSystemDefined, (0, 0), state << 8, 0, 0, None, 8, (key << 16) | (state << 8), -1)
            _post(ev.CGEvent())

    # -------------------------------------------------------------- windows
    def visible_frame(self):
        """Main screen area minus menu bar & Dock, in top-left-origin points."""
        screens = NSScreen.screens()
        full_h = screens[0].frame().size.height
        vf = NSScreen.mainScreen().visibleFrame()
        x, w, h = vf.origin.x, vf.size.width, vf.size.height
        y = full_h - (vf.origin.y + h)
        return int(x), int(y), int(w), int(h)

    def set_front_window(self, x, y, w, h):
        script = f'''
        tell application "System Events"
            set p to first application process whose frontmost is true
            tell p
                if (count of windows) is 0 then return
                set position of window 1 to {{{x}, {y}}}
                set size of window 1 to {{{w}, {h}}}
            end tell
        end tell'''
        run_osascript(script)

    def window(self, cmd: str):
        x, y, w, h = self.visible_frame()
        if cmd == "snap_left":
            self.set_front_window(x, y, w // 2, h)
        elif cmd == "snap_right":
            self.set_front_window(x + w // 2, y, w - w // 2, h)
        elif cmd == "maximize":
            self.set_front_window(x, y, w, h)
        elif cmd == "center":
            cw, ch = int(w * 0.7), int(h * 0.8)
            self.set_front_window(x + (w - cw) // 2, y + (h - ch) // 2, cw, ch)
        elif cmd == "minimize":
            self.hotkey("cmd+m")
        elif cmd == "close":
            self.hotkey("cmd+w")
        elif cmd == "fullscreen":
            self.hotkey("ctrl+cmd+f")
        else:
            raise ValueError(f"Unknown window command '{cmd}'")

    # --------------------------------------------------------------- system
    def system(self, cmd: str):
        if cmd == "mission_control":
            subprocess.Popen(["open", "-a", "Mission Control"])
        elif cmd == "app_expose":
            self.hotkey("ctrl+down")
        elif cmd == "space_left":
            self.hotkey("ctrl+left")
        elif cmd == "space_right":
            self.hotkey("ctrl+right")
        elif cmd == "show_desktop":
            self.hotkey("f11")
        elif cmd == "launchpad":
            subprocess.Popen(["open", "-a", "Launchpad"])
        elif cmd == "screenshot":
            self.hotkey("cmd+shift+3")
        elif cmd == "screenshot_area":
            self.hotkey("cmd+shift+4")
        elif cmd == "app_switcher":
            self.hotkey("cmd+tab")
        elif cmd == "lock_screen":
            self.hotkey("ctrl+cmd+q")
        else:
            raise ValueError(f"Unknown system command '{cmd}'")

    def open_app(self, name: str):
        subprocess.Popen(["open", "-a", name])


def run_osascript(script):
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        print(f"[window] AppleScript failed: {r.stderr.strip()}\n"
              "  -> allow your terminal under System Settings > Privacy & Security > "
              "Accessibility (and Automation > System Events).")


def accessibility_trusted(prompt=True):
    """True if this process may post input events. Optionally shows the system prompt."""
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))
    except Exception:
        return True  # can't check; assume OK
