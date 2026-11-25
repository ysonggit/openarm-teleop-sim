"""Position-only IK for one arm: damped least squares with a null-space pull
toward a rest pose. Runs on its own MjData, so it never disturbs the sim.

For a real robot, load that robot's model here; the operator interface only
ever sends end-effector motion.
"""

import mujoco
import numpy as np

DAMPING = 0.05
NULL_GAIN = 0.1


class ArmIK:
    def __init__(self, model, tcp_body, joint_names, rest):
        self.model = model
        self.data = mujoco.MjData(model)
        self.body = model.body(tcp_body).id
        jids = [model.joint(n).id for n in joint_names]
        self.qadr = model.jnt_qposadr[jids]
        self.dofs = model.jnt_dofadr[jids]
        self.lower, self.upper = model.jnt_range[jids].T
        self.rest = np.asarray(rest, float)
        self._jacp = np.zeros((3, model.nv))

    def fk(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.body].copy()

    def solve(self, q0, target, iters=30, tol=1e-4):
        """Return (q, position error in m), starting the search at q0."""
        q = np.clip(np.asarray(q0, float), self.lower, self.upper)
        n = len(q)
        for _ in range(iters):
            err = target - self.fk(q)
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.body)
            J = self._jacp[:, self.dofs]
            J_pinv = J.T @ np.linalg.inv(J @ J.T + DAMPING**2 * np.eye(3))
            dq = J_pinv @ err + (np.eye(n) - J_pinv @ J) @ (NULL_GAIN * (self.rest - q))
            q = np.clip(q + dq, self.lower, self.upper)
        return q, float(np.linalg.norm(target - self.fk(q)))
