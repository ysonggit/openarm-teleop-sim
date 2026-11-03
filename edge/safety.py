"""Edge-side safety gate.

Runs inside the robot's local control loop. The network only delivers goals;
this gate decides what the joint controllers actually receive:
clamp to joint limits, rate-limit toward the goal, and hold on link loss.
"""

import numpy as np

ACTIVE, HOLD, ESTOP = "active", "hold", "estop"


class SafetyGate:
    def __init__(self, lower, upper, vmax, watchdog_s=0.5):
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.vmax = np.asarray(vmax, float)
        self.watchdog_s = watchdog_s
        self.mode = HOLD
        self.reason = "no controller"
        self.goal = None  # last accepted operator goal, clamped
        self.target = None  # rate-limited setpoint sent to the joint controllers
        self.last_cmd_t = None
        self.events = []  # (mode, reason) transitions, drained by the caller

    def command(self, q, now):
        """Accept an operator goal. Returns 'ok', 'clamped' or 'rejected:<mode>'."""
        if self.mode != ACTIVE:
            return f"rejected:{self.mode}"
        q = np.asarray(q, float)
        if q.shape != self.lower.shape:
            raise ValueError(f"expected {self.lower.size} joints, got {q.size}")
        clamped = np.clip(q, self.lower, self.upper)
        self.goal = clamped
        self.last_cmd_t = now
        return "clamped" if not np.array_equal(clamped, q) else "ok"

    def activate(self, q_meas, now):
        """Leave HOLD/ESTOP and resume from the current setpoint, so nothing jumps."""
        if self.target is None:
            self._freeze(q_meas)
        self.goal = self.target.copy()
        self.last_cmd_t = now
        self._set(ACTIVE, "operator activated")

    def hold(self, reason):
        """Stop at the current setpoint. Resyncing to the measured pose would let
        gravity sag accumulate over repeated hold/activate cycles."""
        if self.target is not None:
            self.goal = self.target.copy()
        self._set(HOLD, reason)

    def estop(self, q_meas, reason="operator e-stop"):
        """Stop at the measured pose: the arm may be blocked or in contact."""
        self._freeze(q_meas)
        self._set(ESTOP, reason)

    def tick(self, q_meas, now, dt):
        """One control-loop step. Returns the setpoint for the joint controllers."""
        if self.target is None:
            self._freeze(q_meas)
        if self.mode == ACTIVE and now - self.last_cmd_t > self.watchdog_s:
            self.hold(f"watchdog: no command for {self.watchdog_s:.1f}s")
        if self.mode == ACTIVE:
            step = self.vmax * dt
            self.target = self.target + np.clip(self.goal - self.target, -step, step)
        return self.target

    def _freeze(self, q_meas):
        self.goal = self.target = np.clip(np.asarray(q_meas, float), self.lower, self.upper)

    def _set(self, mode, reason):
        if (mode, reason) != (self.mode, self.reason):
            self.mode, self.reason = mode, reason
            self.events.append((mode, reason))
