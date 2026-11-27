from pathlib import Path

import mujoco
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
    assert 0.8 < shoulder[2] < 0.9  # column sunk into the base


@pytest.fixture(scope="module")
def world():
    """The robot on its base in the greenhouse aisle."""
    from edge.mujoco_adapter import MujocoAdapter

    return MujocoAdapter(MODEL, base_path=BASE, scene="strawberry")


def make_teleop(a):
    from edge.teleop import CartesianTeleop

    t = CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, FRAME)
    t.reset(a.home, now=0.0)
    return t


def side_of(p):
    return "left" if p[1] > 0 else "right"


def test_near_berries_are_in_reach_and_later_ones_need_the_base(world):
    """Near berries are well within teleop's reach slack for the arm on their side. Those further
    down the aisle are out of reach from the start, and in reach once the base drives up to them."""
    from edge.greenhouse import FIXED_REACH_X
    from edge.teleop import REACH_SLACK

    t = make_teleop(world)
    err = lambda p: t.arms[side_of(p)].ik.solve(world.home[t.arms[side_of(p)].idx], p)[1]
    near = [p for p in world.berry_start if p[0] <= FIXED_REACH_X]
    far = [p for p in world.berry_start if p[0] > FIXED_REACH_X]
    assert len(near) == 2 and len(far) >= 3
    for p in near:
        assert err(p) < REACH_SLACK / 2, p
    for p in far:
        assert err(p) > 0.1, p
        assert err(p - [p[0] - 0.40, 0, 0]) < REACH_SLACK / 2, p  # base driven to x = p.x - 0.40


def move_to(world, t, goal, side, point, grip, seconds):
    """Steer one gripper target straight at `point` (base frame), stepping the sim."""
    for _ in range(int(seconds / 0.05)):
        v = FRAME.T @ np.clip((np.asarray(point) - t.targets[side]) / (0.15 * 0.05), -1, 1)
        goal = t.step(goal, side, v, grip, now=t.last_t + 0.05, yaw=None)
        world.write(goal)
        world.step(0.05)
        world.detach_grasped()
    return goal


def pick(world, t, goal, p):
    """Come in from behind with the gripper open, close on the berry, pull it back into the aisle."""
    side = side_of(p)
    goal = move_to(world, t, goal, side, p - [0.09, 0, 0], 0, 2.5)
    goal = move_to(world, t, goal, side, p, 0, 2.0)
    goal = move_to(world, t, goal, side, p, 1, 1.0)
    return move_to(world, t, goal, side, p + [-0.08, -np.sign(p[1]) * 0.06, 0], 1, 1.5)


def test_berry_hangs_until_squeezed_then_comes_off_and_counts(world):
    from edge.task import HarvestTask
    from edge.teleop import TCP

    world.reset_berries()
    world.step(0.5)
    np.testing.assert_allclose(world.berry_positions()[1], world.berry_start[1], atol=1e-3)  # hangs still
    t, goal = make_teleop(world), world.home.copy()
    task = HarvestTask(world.berry_start)
    task.start(now=0.0)
    task.index = 1  # the right-row berry
    p = world.berry_start[1]
    goal = pick(world, t, goal, p)
    assert world.berry_attached() == [True, False, True, True, True, True]
    args = world.berry_positions(), world.berry_attached(), world.body_positions(TCP.values())
    assert task.tick(*args, now=10.0) is None
    assert task.status == "in the gripper"
    assert task.tick(*args, now=11.0) == "hit"
    world.reset_berries()
    assert all(world.berry_attached())
    np.testing.assert_allclose(world.berry_positions()[1], p)
    world.write(world.home)
    world.step(1.0)


@pytest.mark.parametrize("offset, outcome", [
    ((0.015, 0.01, -0.015), "held"),  # a bit off-center: still a good grasp
    ((-0.02, 0, 0), "on the stem"),  # too shallow: the fingertips only catch its back
])
def test_off_center_grasps_hold_and_shallow_ones_leave_the_berry_on(world, offset, outcome):
    from edge.teleop import TCP

    world.reset_berries()
    t, goal = make_teleop(world), world.home.copy()
    goal = pick(world, t, goal, world.berry_start[0] + offset)
    tcp = world.body_positions([TCP["left"]])[0]
    if outcome == "held":
        assert not world.berry_attached()[0]
        assert np.linalg.norm(world.berry_positions()[0] - tcp) < 0.05
    else:
        assert world.berry_attached()[0]
    world.reset_berries()
    world.write(world.home)
    world.step(1.5)


def test_a_berry_down_the_aisle_can_be_picked_after_driving(world):
    world.reset_berries()
    t, goal = make_teleop(world), world.home.copy()
    p = world.berry_start[4]  # left row, 1.6 m down the aisle
    for x in np.linspace(0, p[0] - 0.40, 50):
        goal[0] = x
        world.write(goal)
        world.step(0.05)
    world.step(1.0)
    np.testing.assert_allclose(world.base_pose(), [p[0] - 0.40, 0, 0], atol=0.01)
    goal = pick(world, t, goal, p - [p[0] - 0.40, 0, 0])  # in the base frame
    assert not world.berry_attached()[4]
    assert np.linalg.norm(world.berry_positions()[4] - p) > 0.05
    world.reset_berries()
    world.write(world.home)
    world.step(4.0)


def test_driving_down_the_aisle_touches_nothing(world):
    """With the arms at home the base drives the whole aisle without touching gutters, legs or berries."""
    m, d = world.model, world.data
    world.reset_berries()
    robot = {b for b in range(1, m.nbody) if not m.body(b).name.startswith("berry")}
    goal = world.home.copy()
    for x in list(np.linspace(0, 1.5, 60)) + list(np.linspace(1.5, -1.0, 100)):
        goal[0] = x
        world.write(goal)
        world.step(0.05)
        hits = [c for c in d.contact[:d.ncon]
                if (m.geom_bodyid[c.geom1] in robot) != (m.geom_bodyid[c.geom2] in robot)
                and m.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE and m.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE]
        assert not hits, x
    assert all(world.berry_attached())
    goal[0] = 0
    world.write(goal)
    world.step(3.0)


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
    from edge.task import HarvestTask
    from edge.teleop import CartesianTeleop

    teleop = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper, FRAME)
    gate = SafetyGate(adapter.lower, adapter.upper, adapter.vmax)
    gate.tick(adapter.read()[0], now=0.0, dt=0.01)
    edge = Edge(adapter, gate, SessionLog(tmp_path), teleop, HarvestTask(adapter.berry_start))
    json.dumps(edge.hello(1), allow_nan=False)
    json.dumps(edge.state(), allow_nan=False)


def push(t, goal, side, v, seconds):
    for i in range(int(seconds / 0.05)):
        goal = t.step(goal, side, v, 0, now=t.last_t + 0.05, yaw=0.0)
    return goal


def test_gripper_stops_above_the_base_when_lowered_over_it(teleop):
    from edge.teleop import KEEP_OUT_TOP, KEEP_OUT_XY

    t, goal = teleop
    goal = push(t, goal, "left", [0, 0, 1], 1.0)  # up first, above the keep-out top
    goal = push(t, goal, "left", [0, -1, 0], 3.0)  # back over the base
    assert abs(t.targets["left"][0]) < KEEP_OUT_XY
    goal = push(t, goal, "left", [0, 0, -1], 6.0)  # then all the way down
    assert t.targets["left"][2] >= KEEP_OUT_TOP - 1e-9


def test_gripper_coming_in_low_from_the_side_stops_at_the_edge(teleop):
    from edge.teleop import KEEP_OUT_TOP, KEEP_OUT_XY

    t, goal = teleop
    goal = push(t, goal, "left", [0, 0, -1], 3.0)  # down in front of the base
    assert t.targets["left"][2] < KEEP_OUT_TOP
    goal = push(t, goal, "left", [0, -1, 0], 4.0)  # then back toward it
    assert t.targets["left"][0] >= KEEP_OUT_XY - 1e-9


def test_lowered_arms_never_touch_the_base(adapter):
    """Drive both grippers down over and in front of the base in the sim; no arm link may contact it."""
    from edge.teleop import CartesianTeleop

    t = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper, FRAME)
    t.reset(adapter.home, now=0.0)
    m, d = adapter.model, adapter.data
    base = m.body("base_link").id
    goal = adapter.home.copy()
    for side in ("left", "right"):
        for v, secs in (([0, 0, 1], 1.0), ([0, -1, 0], 3.0), ([0, 0, -1], 5.0), ([0, 1, 0], 3.0), ([0, 0, -1], 3.0)):
            for _ in range(int(secs / 0.05)):
                goal = t.step(goal, side, v, 0, now=t.last_t + 0.05, yaw=0.0)
                adapter.write(goal)
                adapter.step(0.05)
                touching = [c for c in d.contact[:d.ncon] if base in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
                assert not touching, (side, v)
    adapter.write(adapter.home)
    adapter.step(1.0)
