"""Signal smoothing: the One Euro filter (Casiez et al., 2012).

Low jitter when the hand is still, low lag when it moves fast.
"""
import math


class LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = LowPass()
        self.dx = LowPass()
        self.t_prev = None
        self.x_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = LowPass()
        self.dx = LowPass()
        self.t_prev = None
        self.x_prev = None

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            self.dx(0.0, 1.0)
            return self.x(x, 1.0)
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        edx = self.dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        out = self.x(x, self._alpha(cutoff, dt))
        self.t_prev, self.x_prev = t, x
        return out


class OneEuro2D:
    def __init__(self, **kw):
        self.fx = OneEuroFilter(**kw)
        self.fy = OneEuroFilter(**kw)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)
