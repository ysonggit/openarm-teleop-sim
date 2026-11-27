"""Autonomous run with help tickets and episode recording, on a simulated clock."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
BASE = ROOT / "vendor/tidybot_base/base.xml"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh first")


class FakeWS:
    def __init__(self):
        self.json = []

    async def send_str(self, s):
        pass

    async def send_bytes(self, b):
        pass

    async def send_json(self, m):
        self.json.append(m)


def make_edge(tmp_path, scene):
    from edge.autopilot import PLANS, Autopilot
    from edge.episodes import Recorder
    from edge.mujoco_adapter import MujocoAdapter
    from edge.safety import SafetyGate
    from edge.server import LOOP_DT, Edge, SessionLog
    from edge.task import HarvestTask
    from edge.teleop import CartesianTeleop

    a = MujocoAdapter(MODEL, base_path=BASE, scene=scene)
    edge = Edge(a, SafetyGate(a.lower, a.upper, a.vmax), SessionLog(tmp_path / "logs"),
                CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, a.camera_frame()),
                HarvestTask(a.berry_start), autopilot=Autopilot(PLANS[scene]), recorder=Recorder(tmp_path / "ep"))
    clock = {"t": 0.0}
    edge.clock = lambda: clock["t"]
    # A fake head camera: real rendering is slow and needs a GL context; the frame format is the same.
    a.render_rgbd = lambda qpos: (np.full((240, 320, 3), 90, np.uint8), np.full((240, 320), 0.8))

    async def run(ticks, until=lambda: False, each=None):
        for _ in range(ticks):
            clock["t"] += LOOP_DT
            await edge.tick(round(clock["t"] / LOOP_DT))
            await asyncio.sleep(0)  # let finished renders land
            if each is not None and round(clock["t"] / LOOP_DT) % 5 == 0:
                await each()
            if until():
                return True
        return False

    return edge, run


def test_weeding_scene_has_weeds_held_in_the_soil():
    from edge import tunnel
    from edge.mujoco_adapter import MujocoAdapter

    a = MujocoAdapter(MODEL, base_path=BASE, scene="weeding")
    assert len(a.berry_ids) == len(tunnel.WEEDS)
    assert all(a.berry_attached())
    assert a.target_rgba == tunnel.WEED
    a.step(1.0)  # held by the weld: they don't fall over
    assert np.allclose(a.berry_positions(), a.berry_start, atol=1e-3)


def test_robot_asks_for_help_an_operator_solves_or_skips_and_every_attempt_is_recorded(tmp_path):
    """Strawberries: berries 2 and 4 are marked hard, so the robot raises tickets there.
    An operator solves the first (driven here by a second autopilot through the operator
    message path) and gives up on the second; the robot does the rest. Every picked berry
    ends up in the basket on the base and stays there while the base drives on."""
    from edge.autopilot import PLANS, Autopilot, Plan

    edge, run = make_edge(tmp_path, "strawberry")
    robot, human = FakeWS(), FakeWS()
    edge.clients = {robot: 1, human: 2}
    ap = edge.autopilot

    async def scenario():
        await edge.handle(human, {"type": "auto_start"})
        assert await run(6000, until=lambda: ap.ticket is not None)
        assert ap.ticket["index"] == 2 and ap.ticket["code"] == "occluded"
        assert edge.state()["actor"] == "none" and edge.gate.mode == "hold"

        await edge.handle(human, {"type": "accept"})
        assert edge.controller is human and edge.state()["actor"] == "human"
        p = PLANS["strawberry"]
        driver = Autopilot(Plan(p.base_dx, p.pre, p.grasp, p.pull, place=p.place))
        driver.start(edge.clock())

        async def drive():  # the "operator": same messages as the browser
            if edge.controller is human:
                msg, _ = driver.tick(edge._obs(), edge.clock())
                if msg:
                    await edge.handle(human, msg)

        assert await run(3000, until=lambda: edge.task.index == 3, each=drive)
        assert edge.controller is None and human.json[-1] == {"type": "control", "granted": False, "reason": "solved"}

        assert await run(6000, until=lambda: ap.ticket is not None)
        assert ap.ticket["index"] == 4
        await edge.handle(human, {"type": "accept"})
        await edge.handle(human, {"type": "skip"})
        assert human.json[-1]["reason"] == "skipped" and edge.task.index == 5

        assert await run(8000, until=lambda: not ap.running)

    asyncio.run(scenario())
    eps = list(reversed(edge.recorder.list()))
    assert [e["outcome"]["result"] for e in eps] == ["success"] * 4 + ["skipped"] + ["success"] * 3
    assert [e["outcome"]["by"] for e in eps] == ["auto", "auto", "human", "auto", "human", "auto", "auto", "auto"]
    placed = edge.adapter.berry_placed()
    assert all(placed[i] for i in (0, 1, 2, 3, 5, 6, 7)) and not placed[4]
    assert all(e["rgbd_frames"] > 0 for e in eps if e["outcome"]["result"] == "success")
    assert eps[2]["tickets"] == ["occluded"] and eps[2]["actors"] == ["auto", "human"]

    ep = json.loads(edge.recorder.path(eps[2]["id"]).read_text())
    assert ep["format"] == "openarm-episode/v0" and ep["scene"] == "strawberry" and ep["task"] == "pick"
    assert [s["actor"] for s in ep["segments"]] == ["auto", "none", "human"]
    assert any(e["kind"] == "takeover" and e["reason"] == "occluded" for e in ep["events"])
    assert any(e["kind"] == "detached" for e in ep["events"])
    steps = ep["steps"]
    n = len(steps["t"])
    assert all(len(steps[k]) == n for k in ("actor", "mode", "q", "goal", "base", "tcp", "target", "attached", "action"))
    assert len(steps["q"][0]) == len(ep["joint_names"])
    human_actions = [x for x, a in zip(steps["action"], steps["actor"]) if a == "human" and x]
    assert human_actions and {x["kind"] for x in human_actions} <= {"ee", "base", "home"}
    assert steps["attached"][0] and not steps["attached"][-1]
    assert len(ep["bodies"]) == (n + 1) // 2
    frames = ep["rgbd"]["frames"]
    assert ep["rgbd"]["camera"]["width"] == 320 and len(frames) == eps[2]["rgbd_frames"]
    assert all(b["t"] - a["t"] >= 0.2 - 1e-9 for a, b in zip(frames, frames[1:]))  # at most 5 Hz
    from edge import rgbd
    jpeg, depth = rgbd.unpack(edge.recorder.frame_path(ep["id"], frames[0]["file"]).read_bytes())
    assert jpeg[:2] == b"\xff\xd8" and depth.shape == (240, 320) and np.allclose(depth, 0.8)
    assert edge.recorder.frame_path(ep["id"], "../index.jsonl") is None


def test_weeding_run_needs_help_twice(tmp_path):
    edge, run = make_edge(tmp_path, "weeding")
    viewer = FakeWS()
    edge.clients = {viewer: 1}
    ap = edge.autopilot

    async def scenario():
        await edge.handle(viewer, {"type": "auto_start"})
        assert await run(6000, until=lambda: ap.ticket is not None)
        assert ap.ticket == {"index": 1, "code": "weed_or_crop", "text": ap.plan.hard[1][1]}
        await edge.handle(viewer, {"type": "accept"})
        await edge.handle(viewer, {"type": "skip"})
        assert await run(6000, until=lambda: ap.ticket is not None)
        assert ap.ticket["index"] == 4
        await edge.handle(viewer, {"type": "auto_stop"})
        assert not ap.running and edge.gate.mode == "hold"

    asyncio.run(scenario())
    results = [e["outcome"]["result"] for e in reversed(edge.recorder.list())]
    assert results == ["success", "skipped", "success", "success", "aborted"]


def test_weeding_beds_are_raised_trapezoids_of_rough_soil():
    from edge import tunnel

    v, f = tunnel.terrain()
    top = np.abs(v[:, 1] - 0.55) < 0.1  # middle of a bed
    assert abs(np.median(v[top & (np.abs(v[:, 0] - 3) < 2), 2]) - tunnel.BED_TOP) < 0.01
    aisle = (np.abs(v[:, 1]) < 0.2) & (np.abs(v[:, 0] - 3) < 2)
    assert np.median(v[aisle, 2]) < 0.02 and np.std(v[aisle, 2]) > 0.002  # low and not flat
    # sloped sides: half height halfway between foot and top edge
    assert abs(tunnel.bed_height(3.0, (tunnel.BED_IN[0] + tunnel.BED_IN[1]) / 2) - tunnel.BED_TOP / 2) < 1e-6
    assert f.max() < len(v)


def test_scene_switch_only_when_nobody_drives(tmp_path):
    from edge.mujoco_adapter import MujocoAdapter
    from edge.safety import SafetyGate
    from edge.task import HarvestTask
    from edge.teleop import CartesianTeleop

    edge, run = make_edge(tmp_path, "strawberry")
    a = MujocoAdapter(MODEL, base_path=BASE, scene="weeding")
    edge.add_world(a, SafetyGate(a.lower, a.upper, a.vmax),
                   CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, a.camera_frame()), HarvestTask(a.berry_start))
    driver, viewer = FakeWS(), FakeWS()
    edge.clients = {driver: 1, viewer: 2}

    async def scenario():
        await edge.handle(driver, {"type": "claim"})
        await edge.handle(viewer, {"type": "scene", "name": "weeding"})
        assert edge.scene == "strawberry" and viewer.json[-1]["error"] == "someone is driving"
        await edge.handle(driver, {"type": "release"})
        await edge.handle(viewer, {"type": "scene", "name": "weeding"})
        assert edge.scene == "weeding" and edge.adapter is a
        assert edge.hello(2)["scenes"] == ["strawberry", "weeding"]
        await run(10)

    asyncio.run(scenario())
