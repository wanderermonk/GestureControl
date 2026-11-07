"""Gesture state machine.

Turns a stream of per-frame HandState objects into actions:
  * cursor moves, clicks, double clicks, drags and right clicks (point / pinch)
  * scrolling (two fingers up)
  * swipes  ("<pose>_swipe_<left|right|up|down>" in config)
  * holds   ("<pose>_hold" in config), optionally repeating

The engine never talks to the OS directly. It calls a `mouse` backend and an
`actions` dispatcher, which keeps it testable.
"""
from collections import deque

from .filters import OneEuro2D

CURSOR_POSES = ("point", "pinch", "pinch_middle")
DIRECTIONS = ("left", "right", "up", "down")


class GestureEngine:
    def __init__(self, cfg, mouse, actions, clock=None):
        self.cfg = cfg
        self.mouse = mouse
        self.actions = actions
        self.paused = False
        self.screen_w, self.screen_h = mouse.screen_size()
        self.last_action = ""
        self.last_action_t = -1e9
        self._reset_all()

    # ------------------------------------------------------------------ setup
    def _reset_all(self):
        c = self.cfg["cursor"]
        self.filter = OneEuro2D(min_cutoff=c["min_cutoff"], beta=c["beta"])
        self.stable_pose = "none"
        self.cand_pose = "none"
        self.cand_count = 0
        self.last_seen_t = None
        self.cursor = None
        # pinch / drag
        self.pinch_t = None
        self.pinch_pos = None
        self.dragging = False
        self.freeze_until = 0.0
        self.last_click_t = -1e9
        self.last_click_pos = (0, 0)
        self.click_count = 0
        self.right_fired = False
        # scroll
        self.scroll_prev = None
        self.scroll_acc = [0.0, 0.0]
        # swipe / hold
        self.history = deque()
        self.cooldown_until = 0.0
        self.hold_start = None
        self.hold_origin = None
        self.hold_fired_t = None
        self.hold_blocked = False

    def update_config(self, cfg):
        self.release()
        self.cfg = cfg
        self._reset_all()

    def set_paused(self, paused):
        if paused:
            self.release()
        self.paused = paused

    def release(self):
        """Make sure no mouse button is left held down."""
        if self.dragging and self.cursor is not None:
            self.mouse.mouse_up(*self.cursor)
        self.dragging = False
        self.pinch_t = None

    # --------------------------------------------------------------- helpers
    def _gesture(self, name):
        """Return normalised gesture config: dict(action, hold, repeat) or None."""
        g = self.cfg["gestures"].get(name)
        if not g:
            return None
        if isinstance(g, str):
            g = {"action": g}
        t = self.cfg["timing"]
        return {
            "action": g["action"],
            "hold": float(g.get("hold", t["hold_time"])),
            "repeat": float(g.get("repeat", 0) or 0),
        }

    def _fire(self, name, spec, t):
        if self.paused and spec["action"] != "builtin:toggle_pause":
            return
        self.last_action = f"{name} -> {spec['action']}"
        self.last_action_t = t
        self.actions.run(spec["action"])

    def _to_screen(self, nx, ny):
        x0, y0, x1, y1 = self.cfg["cursor"]["region"]
        u = min(max((nx - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((ny - y0) / (y1 - y0), 0.0), 1.0)
        return u * (self.screen_w - 1), v * (self.screen_h - 1)

    # ------------------------------------------------------------------ main
    def process(self, hand, t):
        """Feed one frame. `hand` is a HandState (pose 'none' when no hand)."""
        timing = self.cfg["timing"]

        # Hand lost for a while -> reset smoothing so the cursor doesn't fly.
        if hand.pose == "none":
            if self.last_seen_t is not None and t - self.last_seen_t > timing["lost_reset"]:
                self.filter.reset()
                self.release()
                self.stable_pose = "none"
        else:
            self.last_seen_t = t

        prev = self.stable_pose
        self._stabilise(hand.pose)
        pose = self.stable_pose

        if pose != prev:
            self._on_pose_change(prev, pose, t)

        if pose in CURSOR_POSES and not self.paused:
            self._cursor(hand, pose, t)
        elif pose == "two" and not self.paused:
            self._scroll(hand, t)

        if pose != "none":
            self._swipe_and_hold(pose, hand, t)

        return self.status(hand)

    def _stabilise(self, pose):
        need = self.cfg["timing"]["stable_frames"]
        # Moving within the cursor family (point <-> pinch) must be instant,
        # otherwise clicks feel laggy.
        if pose in CURSOR_POSES and self.stable_pose in CURSOR_POSES:
            self.stable_pose = pose
            self.cand_pose, self.cand_count = pose, need
            return
        if pose == self.cand_pose:
            self.cand_count += 1
        else:
            self.cand_pose, self.cand_count = pose, 1
        if self.cand_count >= need:
            self.stable_pose = pose

    def _on_pose_change(self, prev, pose, t):
        # Leaving a pinch -> click or end drag
        if prev == "pinch" and pose != "pinch":
            self._end_pinch(t)
        if prev == "pinch_middle":
            self.right_fired = False
        if pose == "pinch":
            self.pinch_t = t
            self.pinch_pos = self.cursor
            self.freeze_until = t + self.cfg["timing"]["click_freeze"]
        if pose not in CURSOR_POSES:
            self.release()
        if pose != "two":
            self.scroll_prev = None
            self.scroll_acc = [0.0, 0.0]
        self.history.clear()
        self.hold_start = None
        self.hold_fired_t = None
        self.hold_blocked = False

    # ---------------------------------------------------------------- cursor
    def _cursor(self, hand, pose, t):
        timing = self.cfg["timing"]
        sx, sy = self._to_screen(*hand.anchor)
        fx, fy = self.filter(sx, sy, t)
        new = (round(fx), round(fy))

        frozen = t < self.freeze_until
        if not frozen and new != self.cursor:
            dz = self.cfg["cursor"]["deadzone_px"]
            if self.cursor is None or abs(new[0] - self.cursor[0]) + abs(new[1] - self.cursor[1]) >= dz:
                self.cursor = new
                if self.dragging:
                    self.mouse.drag(*self.cursor)
                else:
                    self.mouse.move(*self.cursor)

        if self.cursor is None:
            return

        if pose == "pinch" and self.pinch_t is not None and not self.dragging:
            if self.pinch_pos is None:
                self.pinch_pos = self.cursor
            if t - self.pinch_t >= timing["drag_delay"]:
                self.mouse.mouse_down(*self.pinch_pos)
                self.dragging = True
                self.last_action = "drag start"
                self.last_action_t = t

        if pose == "pinch_middle" and not self.right_fired:
            self.right_fired = True
            self.mouse.click(*self.cursor, button="right", count=1)
            self.last_action, self.last_action_t = "right click", t

    def _end_pinch(self, t):
        timing = self.cfg["timing"]
        if self.dragging:
            self.mouse.mouse_up(*self.cursor)
            self.dragging = False
            self.last_action, self.last_action_t = "drag end", t
        elif self.pinch_t is not None and self.pinch_pos is not None and not self.paused:
            pos = self.pinch_pos
            near = abs(pos[0] - self.last_click_pos[0]) + abs(pos[1] - self.last_click_pos[1]) < 40
            if t - self.last_click_t < timing["double_click"] and near:
                self.click_count = min(self.click_count + 1, 3)
            else:
                self.click_count = 1
            self.mouse.click(*pos, button="left", count=self.click_count)
            self.last_click_t, self.last_click_pos = t, pos
            self.last_action = "double click" if self.click_count == 2 else "click"
            self.last_action_t = t
        self.pinch_t = None
        self.pinch_pos = None

    # ---------------------------------------------------------------- scroll
    def _scroll(self, hand, t):
        sc = self.cfg["scroll"]
        if self.scroll_prev is None:
            self.scroll_prev = hand.palm
            return
        dx = hand.palm[0] - self.scroll_prev[0]
        dy = hand.palm[1] - self.scroll_prev[1]
        self.scroll_prev = hand.palm
        sign = -1 if sc["natural"] else 1
        # hand moves up (dy < 0) -> scroll up (positive wheel) unless natural
        self.scroll_acc[1] += -dy * sc["speed"] * sign
        self.scroll_acc[0] += -dx * sc["speed"] * sign if sc["horizontal"] else 0.0
        vy, vx = int(self.scroll_acc[1]), int(self.scroll_acc[0])
        if abs(vy) >= sc["min_step"] or abs(vx) >= sc["min_step"]:
            self.mouse.scroll(vy, vx)
            self.scroll_acc = [self.scroll_acc[0] - vx, self.scroll_acc[1] - vy]

    # ----------------------------------------------------------- swipe/hold
    def _swipe_and_hold(self, pose, hand, t):
        timing = self.cfg["timing"]
        swipes = {d: self._gesture(f"{pose}_swipe_{d}") for d in DIRECTIONS}
        has_swipe = any(swipes.values())
        hold = self._gesture(f"{pose}_hold")

        if has_swipe and t >= self.cooldown_until:
            self.history.append((t, hand.palm))
            while self.history and t - self.history[0][0] > timing["swipe_window"]:
                self.history.popleft()
            t0, p0 = self.history[0]
            dx, dy = hand.palm[0] - p0[0], hand.palm[1] - p0[1]
            d = None
            if abs(dx) > timing["swipe_distance"] and abs(dx) > abs(dy) * 1.6:
                d = "right" if dx > 0 else "left"
            elif abs(dy) > timing["swipe_distance"] and abs(dy) > abs(dx) * 1.6:
                d = "down" if dy > 0 else "up"
            if d and swipes[d]:
                self._fire(f"{pose} swipe {d}", swipes[d], t)
                self.cooldown_until = t + timing["swipe_cooldown"]
                self.history.clear()
                self.hold_blocked = True  # don't also trigger the hold
                return

        if not hold or self.hold_blocked:
            return
        if self.hold_start is None:
            self.hold_start, self.hold_origin = t, hand.palm
            return
        moved = abs(hand.palm[0] - self.hold_origin[0]) + abs(hand.palm[1] - self.hold_origin[1])
        if moved > timing["hold_still"]:
            # hand is moving: restart the hold timer
            self.hold_start, self.hold_origin = t, hand.palm
            self.hold_fired_t = None
            return
        if self.hold_fired_t is None:
            if t - self.hold_start >= hold["hold"]:
                self._fire(f"{pose} hold", hold, t)
                self.hold_fired_t = t
        elif hold["repeat"] > 0 and t - self.hold_fired_t >= hold["repeat"]:
            self._fire(f"{pose} hold", hold, t)
            self.hold_fired_t = t

    # ---------------------------------------------------------------- status
    def hold_progress(self, t):
        hold = self._gesture(f"{self.stable_pose}_hold")
        if not hold or self.hold_start is None or self.hold_fired_t is not None or self.hold_blocked:
            return 0.0
        return min((t - self.hold_start) / hold["hold"], 1.0)

    def status(self, hand):
        pose = self.stable_pose
        if self.paused:
            mode = "paused"
        elif self.dragging:
            mode = "drag"
        elif pose in CURSOR_POSES:
            mode = "cursor"
        elif pose == "two":
            mode = "scroll"
        elif pose == "none":
            mode = "idle"
        else:
            mode = "gesture"
        return {
            "pose": pose,
            "raw_pose": hand.pose,
            "mode": mode,
            "paused": self.paused,
            "cursor": self.cursor,
            "last_action": self.last_action,
            "last_action_t": self.last_action_t,
        }
