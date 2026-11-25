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
MARKER_IDLE = [0.6, 0.6, 0.6, 0.35]
MARKER_ACTIVE = [0.85, 0.47, 0.34, 0.9]
MARKER_DONE = [0.3, 0.65, 0.35, 0.5]


class MujocoAdapter:
    def __init__(self, model_path, keyframe="home", width=480, height=360, markers=(), marker_radius=0.02):
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], rgba=[0.55, 0.5, 0.42, 1]
        )
        spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
        for i, pos in enumerate(markers):  # visual only: no contacts
            spec.worldbody.add_geom(
                name=f"marker{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[marker_radius, 0, 0],
                pos=pos, contype=0, conaffinity=0, rgba=MARKER_IDLE,
            )
        self.model = spec.compile()
        self.marker_ids = [self.model.geom(f"marker{i}").id for i in range(len(markers))]
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
        # Steep view from behind the robot, which faces +x. Height is hard to
        # judge from here; that is part of what the reach task measures.
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.45, 0, 0.45]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.3, 0, -65

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

    def camera_frame(self):
        """Columns: world vectors for screen right, screen forward (level), up."""
        self.renderer.update_scene(self.data, camera=self.camera)
        fwd = np.array(self.renderer.scene.camera[0].forward, float)
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        return np.column_stack([np.cross(fwd, up), fwd, up])

    def camera_params(self):
        c = self.camera
        return {"lookat": c.lookat.tolist(), "distance": c.distance, "azimuth": c.azimuth, "elevation": c.elevation}

    def body_positions(self, names):
        return [self.data.xpos[self.model.body(n).id].copy() for n in names]

    def set_marker_colors(self, active, done):
        for i, gid in enumerate(self.marker_ids):
            self.model.geom_rgba[gid] = MARKER_ACTIVE if i == active else MARKER_DONE if i < done else MARKER_IDLE

    def render_jpeg(self, quality=70):
        self.renderer.update_scene(self.data, camera=self.camera)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
