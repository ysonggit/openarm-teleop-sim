"""MuJoCo stand-in for the robot.

Same surface a real adapter (ROS 2 / supplier SDK) must provide:
names, limits, read(), write(target), step(dt), camera frames.
"""

import io
import warnings

import mujoco
import numpy as np
from PIL import Image

ARM_VMAX = 1.0  # rad/s, conservative for teleop
GRIPPER_VMAX = 0.05  # m/s
MARKER_IDLE = [0.6, 0.6, 0.6, 0.35]
MARKER_ACTIVE = [0.85, 0.47, 0.34, 0.9]
MARKER_DONE = [0.3, 0.65, 0.35, 0.5]

# Stanford TidyBot base (MuJoCo Menagerie): planar x, y, yaw joints, no wheel physics.
BASE_JOINTS = ("joint_x", "joint_y", "joint_th")
BASE_VMAX = (0.5, 0.5, 1.0)  # m/s, m/s, rad/s
BASE_FENCE = 1.5  # m; the base may not leave this square around the start (floor is 2 m)
BASE_TOP = 0.335  # m; top of the TidyBot arm plate
COLUMN_SINK = 0.28  # m; the OpenArm column's foot sits inside the base, so the shoulders end up ~0.75 m high
MOUNT_Z = BASE_TOP - COLUMN_SINK  # m; height of the OpenArm origin (column foot) above the floor


class MujocoAdapter:
    def __init__(self, model_path, keyframe="home", width=480, height=360, markers=(), marker_radius=0.02,
                 base_path=None):
        spec = mujoco.MjSpec.from_file(str(model_path))
        self.mount_z = 0.0
        if base_path:
            spec = self._mount_on_base(spec, base_path)
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], rgba=[0.55, 0.5, 0.42, 1]
        )
        spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
        for i, pos in enumerate(markers):  # visual only: no contacts
            spec.worldbody.add_geom(
                name=f"marker{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[marker_radius, 0, 0],
                pos=pos, contype=0, conaffinity=0, rgba=MARKER_IDLE,
            )
        with warnings.catch_warnings():  # attach() with an empty prefix keeps the arm's names
            warnings.simplefilter("ignore")
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
        unlimited = m.actuator_ctrllimited == 0  # e.g. base yaw
        self.lower[unlimited], self.upper[unlimited] = -np.inf, np.inf
        self.vmax = np.array([ARM_VMAX if self._is_hinge(i) else GRIPPER_VMAX for i in range(m.nu)])
        self.base_qadr = None
        if all(n in self.names for n in BASE_JOINTS):
            self.vmax[[self.names.index(n) for n in BASE_JOINTS]] = BASE_VMAX
            self.base_qadr = [m.jnt_qposadr[m.joint(n).id] for n in BASE_JOINTS]
        if self.home is None:
            self.home = self.read()[0]

        self.renderer = mujoco.Renderer(m, height=height, width=width)
        # Steep view from behind the robot, which faces +x. Height is hard to
        # judge from here; that is part of what the reach task measures.
        # On the mobile base it rides along like a chase camera (see _follow_base).
        self.camera_local = {"lookat": [0.45, 0, 0.45 + self.mount_z], "distance": 1.3, "azimuth": 0, "elevation": -65}
        self.camera = mujoco.MjvCamera()
        self._follow_base()

    def _mount_on_base(self, arm, base_path):
        base = mujoco.MjSpec.from_file(str(base_path))
        base.option.timestep = arm.option.timestep
        for name in BASE_JOINTS[:2]:
            act = base.actuator(name)
            act.ctrllimited, act.ctrlrange = True, [-BASE_FENCE, BASE_FENCE]
        mount = base.body("base_link").add_frame(pos=[0, 0, MOUNT_Z])
        base.attach(arm, frame=mount, prefix="")
        self.mount_z = MOUNT_Z
        return base

    def base_pose(self):
        """(x, y, yaw) of the base, or zeros for a fixed robot."""
        return np.zeros(3) if self.base_qadr is None else self.data.qpos[self.base_qadr].copy()

    def _follow_base(self):
        x, y, th = self.base_pose()
        c, s = np.cos(th), np.sin(th)
        lx, ly, lz = self.camera_local["lookat"]
        self.camera.lookat[:] = [x + c * lx - s * ly, y + s * lx + c * ly, lz]
        self.camera.distance = self.camera_local["distance"]
        self.camera.azimuth = self.camera_local["azimuth"] + np.degrees(th)
        self.camera.elevation = self.camera_local["elevation"]

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
        self._follow_base()
        self.renderer.update_scene(self.data, camera=self.camera)
        fwd = np.array(self.renderer.scene.camera[0].forward, float)
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        return np.column_stack([np.cross(fwd, up), fwd, up])

    def camera_params(self):
        """Robot camera in the base frame (the 3D view places it on the base itself)."""
        return dict(self.camera_local)

    def body_positions(self, names):
        return [self.data.xpos[self.model.body(n).id].copy() for n in names]

    def set_marker_colors(self, active, done):
        for i, gid in enumerate(self.marker_ids):
            self.model.geom_rgba[gid] = MARKER_ACTIVE if i == active else MARKER_DONE if i < done else MARKER_IDLE

    def render_jpeg(self, quality=70):
        self._follow_base()
        self.renderer.update_scene(self.data, camera=self.camera)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
