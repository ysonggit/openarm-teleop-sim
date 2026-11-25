from pathlib import Path

import mujoco
import numpy as np
import pytest

MODEL = Path(__file__).resolve().parent.parent / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh first")
HOME = np.array([0, 0, 0, 1.570796, 0, 0, 0])


@pytest.fixture(scope="module")
def ik():
    from edge.ik import ArmIK

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joints = [f"openarm_left_joint{i}" for i in range(1, 8)]
    return ArmIK(model, "openarm_left_hand_tcp", joints, rest=HOME)


@pytest.mark.parametrize("offset", [[0.05, 0, 0], [0, 0.08, 0], [0, 0, 0.08], [-0.05, -0.05, -0.05]])
def test_reaches_nearby_targets(ik, offset):
    target = ik.fk(HOME) + offset
    q, err = ik.solve(HOME, target)
    assert err < 2e-3
    np.testing.assert_allclose(ik.fk(q), target, atol=2e-3)


def test_stays_within_joint_limits(ik):
    q, err = ik.solve(HOME, ik.fk(HOME) + [2.0, 0, 0])  # far out of reach
    assert err > 0.5
    assert np.all(q >= ik.lower) and np.all(q <= ik.upper)


def test_small_target_step_gives_small_joint_step(ik):
    q, _ = ik.solve(HOME, ik.fk(HOME) + [0.01, 0, 0])
    assert np.max(np.abs(q - HOME)) < 0.1
