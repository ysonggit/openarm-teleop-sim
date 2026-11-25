from pathlib import Path

import numpy as np
import pytest

from edge.task import DWELL, RADIUS, ReachTask

MODEL = Path(__file__).resolve().parent.parent / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh first")
FRAME = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # right = -y, forward = +x, up = +z


@pytest.fixture
def teleop():
    from edge.mujoco_adapter import MujocoAdapter
    from edge.teleop import CartesianTeleop

    a = MujocoAdapter(MODEL)
    t = CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, FRAME)
    t.reset(a.home, now=0.0)
    return t, a.home.copy()


def tcp(t, side, goal):
    arm = t.arms[side]
    return arm.ik.fk(goal[arm.idx])


@needs_model
def test_full_forward_stick_moves_tcp_at_max_speed(teleop):
    t, goal = teleop
    start = tcp(t, "left", goal)
    for i in range(1, 11):  # 10 messages, 0.05 s apart
        goal = t.step(goal, "left", [0, 1, 0], 0, now=i * 0.05)
    np.testing.assert_allclose(tcp(t, "left", goal) - start, [0.075, 0, 0], atol=2e-3)


@needs_model
def test_long_gap_is_not_caught_up(teleop):
    from edge.teleop import MAX_DT, MAX_SPEED

    t, goal = teleop
    start = tcp(t, "right", goal)
    goal = t.step(goal, "right", [0, 0, 1], 0, now=5.0)
    assert tcp(t, "right", goal)[2] - start[2] == pytest.approx(MAX_SPEED * MAX_DT, abs=2e-3)


@needs_model
def test_target_does_not_wind_up_beyond_reach(teleop):
    from edge.teleop import REACH_SLACK

    t, goal = teleop
    for i in range(1, 200):
        goal = t.step(goal, "left", [0, 1, 0], 0, now=i * 0.05)
    assert np.linalg.norm(t.targets["left"] - tcp(t, "left", goal)) <= REACH_SLACK + 1e-6


@needs_model
def test_grip_maps_to_gripper_range_and_other_arm_is_untouched(teleop):
    t, goal = teleop
    new = t.step(goal, "left", [0, 0, 0], 1.0, now=0.05)
    assert new[t.arms["left"].grip] == pytest.approx(t.lower[t.arms["left"].grip])
    np.testing.assert_array_equal(new[t.arms["right"].idx], goal[t.arms["right"].idx])


def test_task_counts_a_hit_only_after_dwell():
    task = ReachTask(targets=[(0, 0, 0), (1, 0, 0)])
    task.start(now=0.0)
    on_target = [np.array([RADIUS / 2, 0, 0]), np.array([9, 9, 9])]
    assert task.tick(on_target, now=1.0) is None
    assert task.tick(on_target, now=1.0 + DWELL / 2) is None
    assert task.tick(on_target, now=1.0 + DWELL) == "hit"
    assert task.index == 1 and task.results == [pytest.approx(1.0 + DWELL)]


def test_task_leaving_target_resets_dwell_and_finishes():
    task = ReachTask(targets=[(0, 0, 0)])
    task.start(now=0.0)
    near, far = [np.zeros(3)], [np.array([1.0, 0, 0])]
    task.tick(near, now=0.0)
    task.tick(far, now=0.3)
    assert task.tick(near, now=0.4) is None
    assert task.tick(near, now=0.4 + DWELL) == "done"
    assert not task.active
