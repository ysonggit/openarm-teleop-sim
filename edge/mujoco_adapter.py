"""MuJoCo stand-in for the robot.

Same surface a real adapter (ROS 2 / supplier SDK) must provide:
names, limits, read(), write(target), step(dt), camera frames.
"""

import io

import mujoco
import numpy as np
from PIL import Image

ARM_VMAX = 1.0  # rad/s, conservative for teleop
GRIPPER_VMAX = 0.05  # m/s


class MujocoAdapter:
    def __init__(self, model_path, keyframe="home", width=480, height=360):
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], rgba=[0.55, 0.5, 0.42, 1]
        )
        spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.home = None
        if keyframe:
            key = m.key(keyframe)
            mujoco.mj_resetDataKeyframe(m, self.data, key.id)
            self.home = key.ctrl.copy()
        mujoco.mj_forward(m, self.data)

        self.names = [m.actuator(i).name for i in range(m.nu)]
        self.lower, self.upper = m.actuator_ctrlrange.T.copy()
        self.vmax = np.array([ARM_VMAX if self._is_hinge(i) else GRIPPER_VMAX for i in range(m.nu)])
        if self.home is None:
            self.home = self.read()[0]

        self.renderer = mujoco.Renderer(m, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, 0, 0.45]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.9, 160, -15

    def _is_hinge(self, i):
        m = self.model
        return (
            m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
            and m.jnt_type[m.actuator_trnid[i, 0]] == mujoco.mjtJoint.mjJNT_HINGE
        )

    def read(self):
        """Measured position and velocity in actuator space."""
        return self.data.actuator_length.copy(), self.data.actuator_velocity.copy()

    def write(self, target):
        self.data.ctrl[:] = target

    def step(self, dt):
        for _ in range(max(1, round(dt / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def render_jpeg(self, quality=70):
        self.renderer.update_scene(self.data, camera=self.camera)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
