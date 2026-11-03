from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parent.parent / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
ARM_JOINTS = [*range(7), *range(8, 15)]  # left[0:7] + right[8:15]; 7 and 15 are grippers
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh first")


@pytest.fixture(scope="module")
def adapter():
    from edge.mujoco_adapter import MujocoAdapter

    return MujocoAdapter(MODEL)


def test_exposes_two_arms_of_eight_actuators(adapter):
    assert len(adapter.names) == 16
    assert adapter.names[0].startswith("left_") and adapter.names[8].startswith("right_")
    assert np.all(adapter.lower < adapter.upper)


def test_holds_home_pose(adapter):
    adapter.write(adapter.home)
    adapter.step(0.5)
    q, _ = adapter.read()
    np.testing.assert_allclose(q[ARM_JOINTS], adapter.home[ARM_JOINTS], atol=0.05)


@pytest.mark.parametrize("joint", [0, 8])  # joint1 of each arm
def test_tracks_a_joint_target(adapter, joint):
    target = adapter.home.copy()
    target[joint] = np.clip(target[joint] + 0.3, adapter.lower[joint], adapter.upper[joint])
    adapter.write(target)
    adapter.step(1.0)
    q, _ = adapter.read()
    assert abs(q[joint] - target[joint]) < 0.05


def test_renders_jpeg(adapter):
    assert adapter.render_jpeg()[:2] == b"\xff\xd8"
