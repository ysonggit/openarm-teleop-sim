from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
BASE = ROOT / "vendor/tidybot_base/base.xml"
pytestmark = pytest.mark.skipif(not (MODEL.exists() and BASE.exists()), reason="run scripts/fetch_model.sh first")
FRAME = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # right = -y, forward = +x, up = +z


@pytest.fixture(scope="module")
def adapter():
    from edge.mujoco_adapter import MujocoAdapter

    return MujocoAdapter(MODEL, base_path=BASE)


@pytest.fixture
def teleop(adapter):
    from edge.teleop import CartesianTeleop

    t = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper, FRAME)
    t.reset(adapter.home, now=0.0)
    return t, adapter.home.copy()


def test_arms_stand_on_the_base(adapter):
    from edge.mujoco_adapter import BASE_FENCE, BASE_JOINTS, BASE_VMAX, MOUNT_Z

    assert adapter.names[:3] == list(BASE_JOINTS) and len(adapter.names) == 19
    np.testing.assert_allclose(adapter.vmax[:3], BASE_VMAX)
    np.testing.assert_allclose(adapter.upper[:2], BASE_FENCE)
    assert np.isinf(adapter.upper[2])  # yaw may turn freely
    import mujoco

    mujoco.mj_forward(adapter.model, adapter.data)
    shoulder = adapter.data.site("arm_origin").xpos
    assert shoulder[2] == pytest.approx(0.697 + MOUNT_Z, abs=1e-6)


def test_forward_stick_drives_the_base_in_the_view_frame(teleop):
    from edge.teleop import BASE_SPEED

    t, goal = teleop
    for i in range(1, 11):
        goal = t.drive(goal, [0, 1, 0], now=i * 0.05)
    np.testing.assert_allclose(goal[:3], [BASE_SPEED * 0.5, 0, 0], atol=1e-9)


def test_right_stick_turns_clockwise(teleop):
    from edge.teleop import BASE_TURN

    t, goal = teleop
    goal = t.drive(goal, [0, 0, 1], now=0.05)
    assert goal[2] == pytest.approx(-BASE_TURN * 0.05)


def test_arm_targets_ride_with_the_base(teleop):
    """With the base turned 90 deg, 'forward' in a fixed view (+x world) is -y in the base frame."""
    t, goal = teleop
    goal[2] = np.pi / 2
    arm = t.arms["left"]
    start = arm.ik.fk(goal[arm.idx])
    for i in range(1, 11):
        goal = t.step(goal, "left", [0, 1, 0], 0, now=i * 0.05, yaw=0.0)
    np.testing.assert_allclose(arm.ik.fk(goal[arm.idx]) - start, [0, -0.075, 0], atol=2e-3)


def test_home_leaves_the_base_where_it_is(teleop, adapter):
    t, goal = teleop
    goal[:3] = [0.4, -0.2, 1.0]
    homed = t.home(goal, adapter.home)
    np.testing.assert_allclose(homed[:3], [0.4, -0.2, 1.0])
    np.testing.assert_allclose(homed[3:], adapter.home[3:])


def test_gate_keeps_the_base_inside_the_fence(adapter):
    from edge.mujoco_adapter import BASE_FENCE
    from edge.safety import SafetyGate

    gate = SafetyGate(adapter.lower, adapter.upper, adapter.vmax)
    gate.activate(adapter.home, now=0.0)
    goal = adapter.home.copy()
    goal[0] = 5.0
    assert gate.command(goal, now=0.0) == "clamped"
    assert gate.goal[0] == BASE_FENCE


def test_driving_moves_the_grippers_in_the_world(adapter):
    from edge.teleop import TCP

    adapter.write(adapter.home)
    adapter.step(0.5)
    before = adapter.body_positions(TCP.values())
    target = adapter.home.copy()
    target[0] = 0.3
    adapter.write(target)
    adapter.step(1.5)
    after = adapter.body_positions(TCP.values())
    assert adapter.base_pose()[0] == pytest.approx(0.3, abs=0.01)
    for b, a in zip(before, after):
        np.testing.assert_allclose(a - b, [0.3, 0, 0], atol=0.02)


def test_hello_and_state_are_strict_json(adapter, tmp_path):
    """Browsers reject Infinity/NaN, and the base yaw has no limits."""
    import json

    from edge.safety import SafetyGate
    from edge.server import Edge, SessionLog
    from edge.task import ReachTask
    from edge.teleop import CartesianTeleop

    teleop = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper, FRAME)
    gate = SafetyGate(adapter.lower, adapter.upper, adapter.vmax)
    gate.tick(adapter.read()[0], now=0.0, dt=0.01)
    edge = Edge(adapter, gate, SessionLog(tmp_path), teleop, ReachTask())
    json.dumps(edge.hello(1), allow_nan=False)
    json.dumps(edge.state(), allow_nan=False)
