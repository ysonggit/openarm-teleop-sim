import asyncio
import json
import time
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parent.parent / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh first")


class FakeWS:
    def __init__(self):
        self.text, self.frames = 0, 0

    async def send_str(self, s):
        self.text += 1

    async def send_bytes(self, b):
        self.frames += 1


def test_a_slow_camera_frame_never_stalls_the_control_loop(tmp_path):
    """GPU readback can take 100s of ms; it runs on a worker thread, so state (and the
    watchdog's view of operator commands) keeps flowing at 20 Hz meanwhile."""
    from edge.mujoco_adapter import MujocoAdapter
    from edge.safety import SafetyGate
    from edge.server import Edge, SessionLog
    from edge.task import HarvestTask
    from edge.teleop import CartesianTeleop
    import numpy as np

    a = MujocoAdapter(MODEL)
    frame = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    edge = Edge(a, SafetyGate(a.lower, a.upper, a.vmax), SessionLog(tmp_path),
                CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, frame), HarvestTask(a.berry_start))
    a.render_jpeg = lambda qpos: (time.sleep(1.0), b"jpeg")[1]
    a.render_rgbd = lambda qpos: (time.sleep(0.2), (np.zeros((240, 320, 3), np.uint8), np.ones((240, 320))))[1]
    viewer, watcher, head = FakeWS(), FakeWS(), FakeWS()
    edge.clients = {viewer: 1, watcher: 2}
    edge.watching = {watcher}
    edge.rgbd_clients = {head}

    async def run_for(seconds):
        loop = asyncio.create_task(edge.run())
        await asyncio.sleep(seconds)
        loop.cancel()

    asyncio.run(run_for(1.3))
    assert viewer.text >= 15  # ~20 Hz state for 1.3 s despite a 1 s render in flight
    assert watcher.frames == 1 and viewer.frames == 0  # frames only to clients showing the camera
    assert head.frames >= 1 and head.text == 0  # RGB-D only on /rgbd, sharing the render thread
    json.dumps(edge.rgbd_hello(), allow_nan=False)


def test_idle_controller_is_released_but_keepalives_do_not_count(tmp_path, monkeypatch):
    """A left-open tab sends zero-velocity keepalives forever; it must not hold control."""
    import numpy as np

    from edge import server
    from edge.mujoco_adapter import MujocoAdapter
    from edge.safety import SafetyGate
    from edge.server import Edge, SessionLog
    from edge.task import HarvestTask
    from edge.teleop import CartesianTeleop

    monkeypatch.setattr(server, "IDLE_RELEASE_S", 0.4)
    a = MujocoAdapter(MODEL)
    frame = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    edge = Edge(a, SafetyGate(a.lower, a.upper, a.vmax), SessionLog(tmp_path),
                CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, frame), HarvestTask(a.berry_start))
    driver = FakeWS()
    driver.sent = []

    async def send_json(m):
        driver.sent.append(m)
    driver.send_json = send_json
    edge.clients = {driver: 1}

    async def scenario():
        loop = asyncio.create_task(edge.run())
        await edge.handle(driver, {"type": "claim"})
        await edge.handle(driver, {"type": "activate"})
        for _ in range(4):  # real input keeps control
            await edge.handle(driver, {"type": "ee", "arm": "left", "v": [0, 0.5, 0], "grip": 0})
            await asyncio.sleep(0.15)
        assert edge.controller is driver
        for _ in range(8):  # keepalives only: released after the idle time
            await edge.handle(driver, {"type": "ee", "arm": "left", "v": [0, 0, 0], "grip": 0})
            await asyncio.sleep(0.1)
        loop.cancel()

    asyncio.run(scenario())
    assert edge.controller is None
    assert {"type": "control", "granted": False, "reason": "idle"} in driver.sent


def _ws_app(tmp_path):
    import numpy as np
    from aiohttp import web

    from edge.mujoco_adapter import MujocoAdapter
    from edge.safety import SafetyGate
    from edge.server import Edge, SessionLog, rgbd_handler, ws_handler
    from edge.task import HarvestTask
    from edge.teleop import CartesianTeleop

    a = MujocoAdapter(MODEL)
    frame = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    edge = Edge(a, SafetyGate(a.lower, a.upper, a.vmax), SessionLog(tmp_path),
                CartesianTeleop(a.model, a.names, a.home, a.lower, a.upper, frame), HarvestTask(a.berry_start))
    app = web.Application()
    app["edge"] = edge
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/rgbd", rgbd_handler)
    return app, edge


def _run(app, scenario):
    from aiohttp.test_utils import TestClient, TestServer

    async def go():
        async with TestClient(TestServer(app)) as client:
            await scenario(client)
    asyncio.run(go())


def test_non_finite_numbers_never_reach_the_safety_gate(tmp_path):
    """Python's json accepts NaN/Infinity and np.clip passes NaN through; the driver
    must not be able to turn the joint goal into NaN."""
    import numpy as np

    app, edge = _ws_app(tmp_path)

    async def scenario(client):
        ws = await client.ws_connect("/ws")
        await ws.receive_json()  # hello
        await ws.send_json({"type": "claim"})
        await ws.send_json({"type": "activate"})
        for raw in ('{"type":"ee","arm":"left","v":[NaN,0,0],"seq":1}',
                    '{"type":"ee","arm":"left","v":[1e999,0,0],"seq":2}',
                    '{"type":"base","v":[0,Infinity,0]}',
                    '{"type":"ee","arm":"left","v":[0.5,0,0],"grip":NaN}',
                    '[1,2,3]', 'not json'):
            await ws.send_str(raw)
        await ws.send_json({"type": "ping", "t": 1})
        while (m := await ws.receive_json(timeout=5))["type"] != "pong":
            pass
        assert np.isfinite(edge.gate.goal).all()
        await ws.close()

    _run(app, scenario)


def test_connection_caps_and_message_flood(tmp_path, monkeypatch):
    import aiohttp

    from edge import server

    monkeypatch.setattr(server, "MAX_CLIENTS", 2)
    monkeypatch.setattr(server, "MAX_RGBD_CLIENTS", 1)
    app, edge = _ws_app(tmp_path)

    async def scenario(client):
        a, b = await client.ws_connect("/ws"), await client.ws_connect("/ws")
        with pytest.raises(aiohttp.WSServerHandshakeError) as e:
            await client.ws_connect("/ws")
        assert e.value.status == 503
        await client.ws_connect("/rgbd")
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await client.ws_connect("/rgbd")

        await b.receive_json()  # hello
        for i in range(server.MAX_MSG_PER_S * 3):
            await b.send_json({"type": "ping", "t": i})
        while (m := await b.receive(timeout=5)).type not in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            pass
        assert b.close_code == 1008
        await asyncio.sleep(0.05)
        assert len(edge.clients) == 1  # the flooder is gone; the other client stays
        c = await client.ws_connect("/ws")  # and its slot is free again
        await c.close(); await a.close()

    _run(app, scenario)
