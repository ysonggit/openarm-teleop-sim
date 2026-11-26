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
