"""Cartesian teleop: stick velocities in the camera frame -> TCP targets -> joint goals.

The operator sends velocities, not positions, so a lost packet only means a
brief stop. Joint goals still go through the SafetyGate.

On a mobile base the TCP targets live in the base frame, so the arms ride
along when the base drives. The base itself is three more joint goals
(x, y, yaw) and goes through the same gate: rate limit, hold, e-stop.
"""

import numpy as np

from edge.ik import ArmIK
from edge.mujoco_adapter import BASE_JOINTS

MAX_SPEED = 0.15  # m/s at full stick
BASE_SPEED = 0.4  # m/s at full stick
BASE_TURN = 0.8  # rad/s at full stick
MAX_DT = 0.1  # longest gap integrated per message; longer gaps are not "caught up"
REACH_SLACK = 0.01  # m; beyond this the target snaps back to what the arm reached

# Enactic OpenArm v1 bimanual naming. Another robot needs its own table here.
ARM_JOINTS = {side: [f"{side}_joint{i}_ctrl" for i in range(1, 8)] for side in ("left", "right")}
GRIP = {side: f"{side}_finger1_ctrl" for side in ("left", "right")}
TCP = {side: f"openarm_{side}_hand_tcp" for side in ("left", "right")}


def yaw_frame(yaw):
    """Columns (right, forward, up) for a level view heading `yaw` from +x."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([[fwd[1], -fwd[0], 0.0], fwd, [0.0, 0.0, 1.0]])


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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
        self.frame = np.asarray(frame)  # columns: base-frame vectors for (right, forward, up)
        self.base = [names.index(n) for n in BASE_JOINTS] if all(n in names for n in BASE_JOINTS) else None
        self.targets = {}  # TCP targets in the base frame
        self.last_t = None

    def _dt(self, now):
        dt = min(max(now - self.last_t, 0.0), MAX_DT)
        self.last_t = now
        return dt

    def _base_yaw(self, goal):
        return 0.0 if self.base is None else float(goal[self.base[2]])

    def _view(self, goal, yaw):
        """World vectors for the stick axes: the operator's 3D view, else the robot camera."""
        return yaw_frame(yaw) if yaw is not None else rot_z(self._base_yaw(goal)) @ self.frame

    def drive(self, goal, v, now, yaw=None):
        """Integrate one base message: v = (right, forward, turn right) in -1..1."""
        dt = self._dt(now)
        goal = np.array(goal, float)
        if self.base is None:
            return goal
        v = np.clip(v, -1, 1)
        ix, iy, ith = self.base
        goal[[ix, iy]] += (self._view(goal, yaw) @ [v[0], v[1], 0.0])[:2] * BASE_SPEED * dt
        goal[ith] -= v[2] * BASE_TURN * dt
        return goal

    def home(self, goal, home):
        """Home the arms; the base stays where it is."""
        out = np.array(home, float)
        if self.base is not None:
            out[self.base] = goal[self.base]
        return out

    def reset(self, goal, now):
        """Re-anchor the TCP targets at the current joint goal (on activate / home)."""
        self.targets = {s: a.ik.fk(goal[a.idx]) for s, a in self.arms.items()}
        self.last_t = now

    def step(self, goal, side, v, grip, now, yaw=None):
        """Integrate one operator message. Returns the new full joint goal.

        yaw (rad): heading of the operator's view in the world XY plane. When
        given, the stick follows that view instead of the fixed camera.
        """
        arm = self.arms[side]
        dt = self._dt(now)

        to_base = rot_z(self._base_yaw(goal)).T
        target = self.targets[side] + to_base @ self._view(goal, yaw) @ np.clip(v, -1, 1) * MAX_SPEED * dt
        q, err = arm.ik.solve(goal[arm.idx], target)
        if err > REACH_SLACK:
            target = arm.ik.fk(q)
        self.targets[side] = target

        goal = np.array(goal, float)
        goal[arm.idx] = q
        g = float(np.clip(grip, 0, 1))
        goal[arm.grip] = self.upper[arm.grip] + g * (self.lower[arm.grip] - self.upper[arm.grip])
        return goal
