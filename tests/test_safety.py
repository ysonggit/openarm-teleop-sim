import numpy as np
import pytest

from edge.safety import ACTIVE, ESTOP, HOLD, SafetyGate

DT = 0.01


def make_gate():
    gate = SafetyGate(lower=[-1, -1], upper=[1, 1], vmax=[0.5, 0.5], watchdog_s=0.5)
    gate.activate([0, 0], now=0.0)
    return gate


def test_starts_in_hold_and_rejects_commands():
    gate = SafetyGate(lower=[-1], upper=[1], vmax=[1])
    assert gate.mode == HOLD
    assert gate.command([0.5], now=0.0) == "rejected:hold"


def test_goal_is_clamped_to_joint_limits():
    gate = make_gate()
    assert gate.command([5, -0.2], now=0.0) == "clamped"
    np.testing.assert_allclose(gate.goal, [1, -0.2])


def test_wrong_joint_count_is_an_error():
    gate = make_gate()
    with pytest.raises(ValueError):
        gate.command([0.1], now=0.0)


def test_target_moves_no_faster_than_vmax():
    gate = make_gate()
    gate.command([1, -1], now=0.0)
    target = gate.tick([0, 0], now=0.0, dt=DT)
    np.testing.assert_allclose(target, [0.005, -0.005])


def test_watchdog_holds_at_current_setpoint():
    gate = make_gate()
    gate.command([1, 1], now=0.0)
    before = gate.tick([0.2, 0.2], now=0.4, dt=DT).copy()
    assert gate.mode == ACTIVE
    target = gate.tick([0.3, 0.3], now=0.6, dt=DT)
    assert gate.mode == HOLD and gate.reason.startswith("watchdog")
    np.testing.assert_allclose(target, before)
    assert gate.command([1, 1], now=0.7) == "rejected:hold"


def test_hold_activate_cycles_do_not_drift_with_sag():
    gate = make_gate()
    sagged = [-0.03, -0.03]  # measured pose sits below the setpoint under gravity
    for i in range(5):
        gate.hold("released")
        gate.activate(sagged, now=float(i))
        gate.tick(sagged, now=float(i), dt=DT)
    np.testing.assert_allclose(gate.target, [0, 0])


def test_estop_latches_until_activate():
    gate = make_gate()
    gate.estop([0.1, 0.1])
    assert gate.mode == ESTOP
    assert gate.command([1, 1], now=0.0) == "rejected:estop"
    gate.activate([0.1, 0.1], now=1.0)
    assert gate.mode == ACTIVE
    np.testing.assert_allclose(gate.target, [0.1, 0.1])


def test_transitions_are_recorded_once():
    gate = make_gate()
    gate.hold("controller disconnected")
    gate.hold("controller disconnected")
    assert gate.events == [(ACTIVE, "operator activated"), (HOLD, "controller disconnected")]
