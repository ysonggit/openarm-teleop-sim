from pathlib import Path

import numpy as np
import pytest

from edge.task import DWELL, PULL, HarvestTask

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


START = [np.array([0, 0, 0.7])]
TCPS = [np.zeros(3), np.array([5.0, 5, 5])]


def test_task_counts_a_berry_once_off_the_stem_held_and_pulled_away_for_the_dwell():
    task = HarvestTask(starts=START + [np.array([1, 0, 0.7])])
    task.start(now=0.0)
    pulled = [np.array([-PULL - 0.01, 0, 0.7]), np.array([1, 0, 0.7])]
    tcps = [pulled[0], TCPS[1]]
    assert task.tick(pulled, [True, True], tcps, now=0.5) is None  # still on the stem
    assert task.status == "on the stem"
    assert task.tick([START[0], pulled[1]], [False, True], [START[0], TCPS[1]], now=0.6) is None  # not pulled yet
    assert task.tick(pulled, [False, True], tcps, now=1.0) is None
    assert task.status == "in the gripper"
    assert task.tick(pulled, [False, True], tcps, now=1.0 + DWELL) == "hit"
    assert task.index == 1 and task.results == [pytest.approx(1.0 + DWELL)]
    assert task.summary(2.0)["status"] is None  # until the next tick looks at the next berry


def test_task_dropping_the_berry_resets_dwell_and_finishes():
    task = HarvestTask(starts=START)
    task.start(now=0.0)
    held = [np.array([-0.1, 0, 0.7])]
    dropped = [np.array([-0.1, 0, 0.02])]
    task.tick(held, [False], held, now=0.0)
    task.tick(dropped, [False], held, now=0.3)
    assert task.status == "dropped"
    assert task.tick(held, [False], held, now=0.4) is None
    assert task.tick(held, [False], held, now=0.4 + DWELL) == "done"
    assert not task.active


@needs_model
def test_yaw_makes_stick_follow_the_view(teleop):
    import math

    t, goal = teleop
    start = tcp(t, "left", goal)
    for i in range(1, 11):  # view heading +y: forward stick moves along +y
        goal = t.step(goal, "left", [0, 1, 0], 0, now=i * 0.05, yaw=math.pi / 2)
    np.testing.assert_allclose(tcp(t, "left", goal) - start, [0, 0.075, 0], atol=2e-3)
