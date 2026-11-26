"""Head RGB-D camera on a short mast on top of the OpenArm column.

Simulated as one MuJoCo camera rendered twice: color, then depth. Both images
share one pinhole model (aligned by construction; a real sensor needs its own
depth-to-color registration). Depth is z-depth along the optical axis in m.

Wire format of one frame (one binary WebSocket message on /rgbd):
    uint32 LE n | n bytes JPEG color | zlib(uint16 LE depth in mm, row-major)
Depth outside [NEAR, FAR] is 0 = no return, as on a real sensor.
"""

import io
import struct
import zlib

import mujoco
import numpy as np
from PIL import Image

NAME = "head_rgbd"
WIDTH, HEIGHT = 320, 240
FOVY = 58.0  # deg, vertical
PITCH = 45.0  # deg below horizontal: sees both grippers, the near berries and the aisle ahead
NEAR, FAR = 0.1, 3.0  # m; valid depth range
COLUMN_TOP = 0.773  # m above the column origin: top of the shoulder housing
MAST = 0.10  # m; camera this far above the column top (~1.03 m above the floor on the base)
MOUNT = (0.03, 0.0, COLUMN_TOP + MAST)  # in the column frame; forward of center so the shoulders stay out of view
DARK = [0.15, 0.15, 0.16, 1]


def add_camera(spec, column="openarm_body_link0"):
    """Mast, a camera housing (visual only) and the camera, on the column body."""
    body = spec.body(column)
    x, y, z = MOUNT
    visual = dict(contype=0, conaffinity=0, mass=0, rgba=DARK)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.01, 0, 0], fromto=[x, y, COLUMN_TOP, x, y, z - 0.015],
                  **visual)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0125, 0.045, 0.0125], pos=MOUNT, **visual)
    th = np.radians(PITCH)
    # MuJoCo cameras look along -z with +y up: image right = -y (robot's right), image up = forward-up.
    R = np.column_stack([[0, -1, 0], [np.sin(th), 0, np.cos(th)], [-np.cos(th), 0, np.sin(th)]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())
    body.add_camera(name=NAME, pos=[x + 0.013, y, z], quat=quat, fovy=FOVY)


def intrinsics(width=WIDTH, height=HEIGHT, fovy=FOVY):
    """Pinhole model (OpenCV convention: x right, y down, z forward)."""
    f = height / 2 / np.tan(np.radians(fovy) / 2)
    return {"width": width, "height": height, "fx": f, "fy": f, "cx": width / 2, "cy": height / 2}


def pack(rgb, depth, quality=80):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    jpeg = buf.getvalue()
    mm = np.where((depth >= NEAR) & (depth <= FAR), np.round(depth * 1000), 0).astype("<u2")
    return struct.pack("<I", len(jpeg)) + jpeg + zlib.compress(mm.tobytes(), 1)


def unpack(frame, width=WIDTH, height=HEIGHT):
    """Inverse of pack: (JPEG bytes, depth in m with 0 = no return)."""
    (n,) = struct.unpack_from("<I", frame)
    jpeg = frame[4:4 + n]
    depth = np.frombuffer(zlib.decompress(frame[4 + n:]), "<u2").reshape(height, width) / 1000.0
    return jpeg, depth
