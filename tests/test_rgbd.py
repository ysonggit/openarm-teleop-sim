import io
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from edge import rgbd

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
BASE = ROOT / "vendor/tidybot_base/base.xml"
needs_model = pytest.mark.skipif(not (MODEL.exists() and BASE.exists()), reason="run scripts/fetch_model.sh first")


def test_frame_round_trips_depth_in_mm_and_drops_out_of_range():
    rgb = np.zeros((rgbd.HEIGHT, rgbd.WIDTH, 3), np.uint8)
    depth = np.full((rgbd.HEIGHT, rgbd.WIDTH), 0.8123)
    depth[0, 0], depth[0, 1] = rgbd.NEAR / 2, rgbd.FAR * 2  # too close, no hit
    jpeg, back = rgbd.unpack(rgbd.pack(rgb, depth))
    assert Image.open(io.BytesIO(jpeg)).size == (rgbd.WIDTH, rgbd.HEIGHT)
    assert back[0, 0] == 0 and back[0, 1] == 0
    np.testing.assert_allclose(back[1:], 0.812, atol=0.0005)


@pytest.fixture(scope="module")
def world():
    from edge.mujoco_adapter import MujocoAdapter

    a = MujocoAdapter(MODEL, base_path=BASE, scene="strawberry")
    a.write(a.home)
    a.step(0.5)
    return a


def to_pixel(world, p):
    """World point -> (u, v, z-depth) through the head camera's pinhole model."""
    cam = world.data.cam(rgbd.NAME)
    R = cam.xmat.reshape(3, 3)
    c = R.T @ (np.asarray(p) - cam.xpos)  # MuJoCo camera frame: x right, y up, looks along -z
    x, y, z = c[0], -c[1], -c[2]  # OpenCV: x right, y down, z forward
    k = rgbd.intrinsics()
    return k["fx"] * x / z + k["cx"], k["fy"] * y / z + k["cy"], z


@needs_model
def test_center_pixel_depth_matches_a_raycast(world):
    m, d = world.model, world.data
    _, depth = world.render_rgbd()
    cam = d.cam(rgbd.NAME)
    geomid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, cam.xpos, -cam.xmat.reshape(3, 3)[:, 2], None, 1, -1, geomid)
    k = rgbd.intrinsics()
    center = depth[int(k["cy"]) - 1:int(k["cy"]) + 1, int(k["cx"]) - 1:int(k["cx"]) + 1].mean()
    assert center == pytest.approx(dist, abs=0.01)


@needs_model
def test_camera_sees_both_grippers_and_the_near_berries_at_the_right_depth(world):
    from edge.teleop import TCP

    _, depth = world.render_rgbd()
    for p in world.body_positions(TCP.values()):
        u, v, _ = to_pixel(world, p)
        assert 0 <= u < rgbd.WIDTH and 0 <= v < rgbd.HEIGHT
    for p in world.berry_start[:2]:
        u, v, z = to_pixel(world, p)
        assert 0 <= u < rgbd.WIDTH and 0 <= v < rgbd.HEIGHT
        assert depth[int(v), int(u)] == pytest.approx(z, abs=0.03)  # the berry's front surface


@needs_model
def test_camera_rides_with_the_base(world):
    before = world.data.cam(rgbd.NAME).xpos.copy()
    goal = world.home.copy()
    goal[0] = 0.3
    world.write(goal)
    world.step(1.5)
    np.testing.assert_allclose(world.data.cam(rgbd.NAME).xpos - before, [0.3, 0, 0], atol=0.01)
    world.write(world.home)
    world.step(1.5)
