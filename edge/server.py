"""Edge process: local control loop + WebSocket bridge to the operator UI.

The 100 Hz loop and the safety gate run here. The browser sends gamepad
velocities (~20 Hz); IK turns them into joint goals for the gate. The browser
receives state (20 Hz, incl. body poses for the 3D view) and JPEG frames
(~14 fps).

    uv run python -m edge.server [--port 8765]
"""

import argparse
import asyncio
import itertools
import json
import math
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from edge.mujoco_adapter import MARKER_ACTIVE, MARKER_DONE, MARKER_IDLE, MujocoAdapter
from edge.safety import ACTIVE, SafetyGate
from edge.scene import body_poses, export_scene
from edge.task import RADIUS, TARGETS, ReachTask
from edge.teleop import MAX_SPEED, TCP, CartesianTeleop

ROOT = Path(__file__).resolve().parent.parent
LOOP_DT = 0.01
STATE_EVERY = 5  # ticks -> 20 Hz
VIDEO_EVERY = 7  # ticks -> ~14 fps


class SessionLog:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._f = self.path.open("a", buffering=1)

    def write(self, kind, **fields):
        self._f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}) + "\n")


class Edge:
    def __init__(self, adapter, gate, log, teleop, task):
        self.adapter, self.gate, self.log = adapter, gate, log
        self.teleop, self.task = teleop, task
        self.clients = {}  # ws -> client id
        self.controller = None
        self._ids = itertools.count(1)

    def hello(self, cid):
        a = self.adapter
        return {
            "type": "hello",
            "client_id": cid,
            "names": a.names,
            "lower": a.lower.tolist(),
            "upper": a.upper.tolist(),
            "vmax": a.vmax.tolist(),
            "home": a.home.tolist(),
            "watchdog_s": self.gate.watchdog_s,
            "arms": list(self.teleop.arms),
            "max_speed": MAX_SPEED,
            "camera": a.camera_params(),
            "markers": {"ids": a.marker_ids, "idle": MARKER_IDLE, "active": MARKER_ACTIVE, "done": MARKER_DONE},
        }

    def state(self):
        q, _ = self.adapter.read()
        g = self.gate
        return {
            "type": "state",
            "mode": g.mode,
            "reason": g.reason,
            "controller": self.clients.get(self.controller),
            "q": q.tolist(),
            "goal": g.goal.tolist(),
            "target": g.target.tolist(),
            "task": self.task.summary(time.monotonic()),
            "bodies": body_poses(self.adapter.data),
        }

    async def run(self):
        next_t = time.monotonic()
        for tick in itertools.count():
            q, _ = self.adapter.read()
            self.adapter.write(self.gate.tick(q, time.monotonic(), LOOP_DT))
            self.adapter.step(LOOP_DT)
            for mode, reason in self.gate.events:
                self.log.write("mode", mode=mode, reason=reason)
            self.gate.events.clear()

            result = self.task.tick(self.adapter.body_positions(TCP.values()), time.monotonic())
            if result:
                self.log.write("task", event=result, index=self.task.index, seconds=self.task.results[-1])
                self.adapter.set_marker_colors(self.task.index if self.task.active else None, self.task.index)

            if tick % STATE_EVERY == 0:
                await self._broadcast(json.dumps(self.state()))
            if tick % VIDEO_EVERY == 0 and self.clients:
                await self._broadcast(self.adapter.render_jpeg())

            next_t += LOOP_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    async def _broadcast(self, payload):
        for ws in list(self.clients):
            try:
                await (ws.send_bytes if isinstance(payload, bytes) else ws.send_str)(payload)
            except ConnectionError:
                self.clients.pop(ws, None)

    async def handle(self, ws, msg):
        kind, cid, now = msg.get("type"), self.clients[ws], time.monotonic()
        q_meas = self.adapter.read()[0]

        if kind == "ping":
            await ws.send_json({"type": "pong", "t": msg.get("t")})
        elif kind == "claim":
            granted = self.controller in (None, ws)
            if granted:
                self.controller = ws
            self.log.write("claim", client=cid, granted=granted)
            await ws.send_json({"type": "control", "granted": granted})
        elif kind == "estop":  # any connected client may stop the robot
            self.gate.estop(q_meas, f"e-stop by client {cid}")
        elif ws is not self.controller:
            await ws.send_json({"type": "error", "error": "not_controller", "for": kind})
        elif kind == "activate":
            self.gate.activate(q_meas, now)
            self.teleop.reset(self.gate.goal, now)
        elif kind == "ee":
            yaw = msg.get("yaw")  # set when the operator drives from the 3D view
            if (self.gate.mode != ACTIVE or msg.get("arm") not in self.teleop.arms
                    or len(msg.get("v", ())) != 3 or not (yaw is None or (isinstance(yaw, (int, float)) and math.isfinite(yaw)))):
                return
            goal = self.teleop.step(self.gate.goal, msg["arm"], msg["v"], msg.get("grip", 0), now, yaw=yaw)
            result = self.gate.command(goal, now)
            self.log.write("ee", client=cid, seq=msg.get("seq"), arm=msg["arm"], v=msg["v"],
                           grip=msg.get("grip", 0), yaw=yaw, result=result)
        elif kind == "home":
            if self.gate.mode == ACTIVE:
                self.gate.command(self.adapter.home, now)
                self.teleop.reset(self.adapter.home, now)
        elif kind == "task_start":
            self.task.start(now)
            self.adapter.set_marker_colors(0, 0)
            self.log.write("task", event="start", client=cid)
        elif kind == "release":
            self._release("controller released")
        elif kind == "cmd":
            try:
                result = self.gate.command(msg["q"], now)
            except (KeyError, ValueError) as e:
                await ws.send_json({"type": "error", "error": str(e), "for": kind})
                return
            self.log.write("cmd", client=cid, seq=msg.get("seq"), q=msg["q"], result=result)

    def _release(self, reason):
        self.controller = None
        if self.gate.mode == ACTIVE:
            self.gate.hold(reason)

    def disconnect(self, ws):
        cid = self.clients.pop(ws, None)
        self.log.write("disconnect", client=cid)
        if ws is self.controller:
            self._release("controller disconnected")


async def ws_handler(request):
    edge = request.app["edge"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    cid = next(edge._ids)
    edge.clients[ws] = cid
    edge.log.write("connect", client=cid, peer=request.remote)
    await ws.send_json(edge.hello(cid))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await edge.handle(ws, json.loads(msg.data))
    finally:
        edge.disconnect(ws)
    return ws


def main():
    p = argparse.ArgumentParser()
    # Cartesian teleop uses Enactic OpenArm v1 bimanual names (see edge/teleop.py).
    p.add_argument("--model", default=ROOT / "vendor/openarm_mujoco/v1/openarm_bimanual.xml", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8765, type=int)
    args = p.parse_args()

    adapter = MujocoAdapter(args.model, markers=TARGETS, marker_radius=RADIUS)
    teleop = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper,
                             adapter.camera_frame())
    gate = SafetyGate(adapter.lower, adapter.upper, adapter.vmax)
    edge = Edge(adapter, gate, SessionLog(ROOT / "logs"), teleop, ReachTask())
    print(f"session log: {edge.log.path}")

    meta, blob = export_scene(adapter.model)
    print(f"3D scene: {len(meta['geoms'])} geoms, {meta['nvert']} vertices, {len(blob) / 1e6:.1f} MB")

    app = web.Application()
    app["edge"] = edge
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/scene.json", lambda r: web.json_response(meta))
    app.router.add_get("/scene.bin", lambda r: web.Response(body=blob, content_type="application/octet-stream"))
    app.router.add_get("/", lambda r: web.FileResponse(ROOT / "web/index.html"))
    app.router.add_static("/", ROOT / "web")

    async def start_loop(app):
        app["loop_task"] = asyncio.create_task(edge.run())

    app.on_startup.append(start_loop)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
