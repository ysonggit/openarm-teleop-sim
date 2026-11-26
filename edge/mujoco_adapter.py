"""MuJoCo stand-in for the robot.

Same surface a real adapter (ROS 2 / supplier SDK) must provide:
names, limits, read(), write(target), step(dt), camera frames.
"""

import io
import warnings

import mujoco
import numpy as np
from PIL import Image

from edge import greenhouse

ARM_VMAX = 1.0  # rad/s, conservative for teleop
GRIPPER_VMAX = 0.05  # m/s
BERRY_IDLE = greenhouse.BERRY
BERRY_ACTIVE = [1.0, 0.55, 0.1, 1]  # the berry to pick next is highlighted amber
BERRY_DONE = [0.3, 0.65, 0.35, 1]
GRASP_DEPTH = 0.012  # m; a berry comes off its stem only if no further than this ahead of the gripper center

# Stanford TidyBot base (MuJoCo Menagerie): planar x, y, yaw joints, no wheel physics.
BASE_JOINTS = ("joint_x", "joint_y", "joint_th")
BASE_VMAX = (0.5, 0.5, 1.0)  # m/s, m/s, rad/s
BASE_FENCE = 1.5  # m; the base may not leave this square around the start (floor is 2 m)
BASE_TOP = 0.335  # m; top of the TidyBot arm plate
BASE_HALF_WIDTH = 0.241  # m; the TidyBot bumper is a 0.48 m square
COLUMN_SINK = 0.18  # m; the OpenArm column's foot sits inside the base, so the shoulders end up ~0.85 m high.
# Arms hanging straight down reach 0.63 m below the shoulder, so below ~1.0 m they would touch
# the base top; teleop keeps the grippers out of that zone instead (see edge/teleop.py).
MOUNT_Z = BASE_TOP - COLUMN_SINK  # m; height of the OpenArm origin (column foot) above the floor


class MujocoAdapter:
    def __init__(self, model_path, keyframe="home", width=480, height=360, base_path=None, greenhouse_rows=False):
        spec = mujoco.MjSpec.from_file(str(model_path))
        for body in spec.bodies:  # the arm's own controller holds it up, as on the real robot;
            body.gravcomp = 1  # without this the grippers sag ~2.5 cm below their targets
        self.mount_z = 0.0
        if base_path:
            spec = self._mount_on_base(spec, base_path)
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], rgba=[0.55, 0.5, 0.42, 1]
        )
        spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
        berries = greenhouse.build(spec, self.mount_z, base_path is not None) if greenhouse_rows else []
        with warnings.catch_warnings():  # attach() with an empty prefix keeps the arm's names
            warnings.simplefilter("ignore")
            self.model = spec.compile()
        self.berry_ids = [self.model.geom(n).id for n in berries]  # visual geoms (colors, 3D view)
        self._berry_contact = [self.model.geom(f"{n}_collision").id for n in berries]
        self.stem_ids = [self.model.equality(f"stem{i}").id for i in range(len(berries))]
        self.berry_start = np.array([self.model.body(n).pos for n in berries]).reshape(-1, 3)
        self.fingers = {side: {self.model.geom(f"openarm_{side}_{f}_finger_collision").id for f in ("left", "right")}
                        for side in ("left", "right")}
        self._tcp = {side: self.model.body(f"openarm_{side}_hand_tcp").id for side in ("left", "right")}
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

        # The renderer and its MjData are made on first use, in the thread that renders
        # (see render_jpeg): GPU readback can take 100s of ms and must not stall control.
        self.size = (height, width)
        self._render = None
        # Steep view from behind the robot, which faces +x. Height is hard to
        # judge from here; that is part of what the pick task measures.
        # On the mobile base it rides along like a chase camera (see _follow_base).
        self.camera_local = {"lookat": [0.45, 0, 0.45 + self.mount_z], "distance": 1.3, "azimuth": 0, "elevation": -65}

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

    def _camera(self, base_pose):
        """The robot camera rides along with the base like a chase camera."""
        x, y, th = base_pose
        c, s = np.cos(th), np.sin(th)
        lx, ly, lz = self.camera_local["lookat"]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [x + c * lx - s * ly, y + s * lx + c * ly, lz]
        cam.distance = self.camera_local["distance"]
        cam.azimuth = self.camera_local["azimuth"] + np.degrees(th)
        cam.elevation = self.camera_local["elevation"]
        return cam

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
        az = np.radians(self._camera(self.base_pose()).azimuth)  # MuJoCo free camera looks along azimuth
        fwd = np.array([np.cos(az), np.sin(az), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        return np.column_stack([np.cross(fwd, up), fwd, up])

    def camera_params(self):
        """Robot camera in the base frame (the 3D view places it on the base itself)."""
        return dict(self.camera_local)

    def body_positions(self, names):
        return [self.data.xpos[self.model.body(n).id].copy() for n in names]

    def berry_positions(self):
        return [self.data.xpos[self.model.geom_bodyid[g]].copy() for g in self.berry_ids]

    def berry_attached(self):
        return [bool(self.data.eq_active[e]) for e in self.stem_ids]

    def detach_grasped(self):
        """Release a berry from its stem once both fingers of one gripper press on it with the
        berry between them (a shallow grasp on its back edge would slip off, so it stays on).
        Returns [(berry index, side)] for berries picked this step."""
        d, picked = self.data, []
        touching = {}
        for c in d.contact[:d.ncon]:
            touching.setdefault(c.geom1, set()).add(c.geom2)
            touching.setdefault(c.geom2, set()).add(c.geom1)
        for i, (g, e) in enumerate(zip(self._berry_contact, self.stem_ids)):
            if not d.eq_active[e]:
                continue
            for side, fingers in self.fingers.items():
                tcp = self._tcp[side]  # its z axis points out of the gripper
                ahead = (d.xpos[self.model.geom_bodyid[g]] - d.xpos[tcp]) @ d.xmat[tcp].reshape(3, 3)[:, 2]
                if ahead < GRASP_DEPTH and fingers <= touching.get(g, set()):
                    d.eq_active[e] = 0
                    picked.append((i, side))
                    break
        return picked

    def reset_berries(self):
        """Hang every berry back on its stem."""
        m, d = self.model, self.data
        for g, e, pos in zip(self.berry_ids, self.stem_ids, self.berry_start):
            j = m.body_jntadr[m.geom_bodyid[g]]
            d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7] = [*pos, 1, 0, 0, 0]
            d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0
            d.eq_active[e] = 1
        mujoco.mj_forward(m, d)

    def set_berry_colors(self, active, done):
        for i, gid in enumerate(self.berry_ids):
            self.model.geom_rgba[gid] = BERRY_ACTIVE if i == active else BERRY_DONE if i < done else BERRY_IDLE

    def render_jpeg(self, qpos=None, quality=70):
        """Robot camera frame for a snapshot of qpos (default: now). Always call it from
        the same thread; the server uses one worker thread so control never waits on it."""
        if self._render is None:
            self._render = mujoco.Renderer(self.model, *self.size), mujoco.MjData(self.model)
        renderer, d = self._render
        d.qpos[:] = self.data.qpos if qpos is None else qpos
        mujoco.mj_kinematics(self.model, d)
        base = np.zeros(3) if self.base_qadr is None else d.qpos[self.base_qadr]
        renderer.update_scene(d, camera=self._camera(base))
        buf = io.BytesIO()
        Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
