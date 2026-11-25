"""Cartesian teleop: stick velocities in the camera frame -> TCP targets -> joint goals.

The operator sends velocities, not positions, so a lost packet only means a
brief stop. Joint goals still go through the SafetyGate.
"""

import numpy as np

from edge.ik import ArmIK

MAX_SPEED = 0.15  # m/s at full stick
MAX_DT = 0.1  # longest gap integrated per message; longer gaps are not "caught up"
REACH_SLACK = 0.01  # m; beyond this the target snaps back to what the arm reached

# Enactic OpenArm v1 bimanual naming. Another robot needs its own table here.
ARM_JOINTS = {side: [f"{side}_joint{i}_ctrl" for i in range(1, 8)] for side in ("left", "right")}
GRIP = {side: f"{side}_finger1_ctrl" for side in ("left", "right")}
TCP = {side: f"openarm_{side}_hand_tcp" for side in ("left", "right")}


class Arm:
    def __init__(self, model, names, home, side):
        self.idx = [names.index(n) for n in ARM_JOINTS[side]]
        self.grip = names.index(GRIP[side])
        joints = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.ik = ArmIK(model, TCP[side], joints, rest=home[self.idx])


class CartesianTeleop:
    def __init__(self, model, names, home, lower, upper, frame):
        self.arms = {side: Arm(model, names, home, side) for side in ("left", "right")}
        self.lower, self.upper = lower, upper
        self.frame = np.asarray(frame)  # columns: world vectors for (right, forward, up)
        self.targets = {}
        self.last_t = None

    def reset(self, goal, now):
        """Re-anchor the TCP targets at the current joint goal (on activate / home)."""
        self.targets = {s: a.ik.fk(goal[a.idx]) for s, a in self.arms.items()}
        self.last_t = now

    def step(self, goal, side, v, grip, now):
        """Integrate one operator message. Returns the new full joint goal."""
        arm = self.arms[side]
        dt = min(max(now - self.last_t, 0.0), MAX_DT)
        self.last_t = now

        target = self.targets[side] + self.frame @ np.clip(v, -1, 1) * MAX_SPEED * dt
        q, err = arm.ik.solve(goal[arm.idx], target)
        if err > REACH_SLACK:
            target = arm.ik.fk(q)
        self.targets[side] = target

        goal = np.array(goal, float)
        goal[arm.idx] = q
        g = float(np.clip(grip, 0, 1))
        goal[arm.grip] = self.upper[arm.grip] + g * (self.lower[arm.grip] - self.upper[arm.grip])
        return goal
