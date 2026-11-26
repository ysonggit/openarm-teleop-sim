"""Edge process: local control loop + WebSocket bridge to the operator UI.

The 100 Hz loop and the safety gate run here. The browser sends gamepad
velocities (~20 Hz); IK turns them into joint goals for the gate. The browser
receives state (20 Hz, incl. body poses for the 3D view) and JPEG frames
(~14 fps, only to clients showing the robot camera); the head RGB-D camera
streams on its own WebSocket, /rgbd, to view-only clients (web/rgbd.html).
The arms stand on a mobile base unless --no-base is given.

    uv run python -m edge.server [--port 8765] [--no-base]
"""

import argparse
import asyncio
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from edge import access, rgbd
from edge.mujoco_adapter import BERRY_ACTIVE, BERRY_DONE, BERRY_IDLE, MujocoAdapter
from edge.safety import ACTIVE, SafetyGate
from edge.scene import body_poses, export_scene
from edge.task import HarvestTask
from edge.teleop import BASE_SPEED, BASE_TURN, MAX_SPEED, TCP, CartesianTeleop

ROOT = Path(__file__).resolve().parent.parent
LOOP_DT = 0.01
STATE_EVERY = 5  # ticks -> 20 Hz
VIDEO_EVERY = 7  # ticks -> ~14 fps
RGBD_EVERY = 10  # ticks -> at most 10 fps; in practice limited by render time
IDLE_RELEASE_S = 180  # control is released after this long without operator input, so a left-open tab can't hog it


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
        self.watching = set()  # clients showing the robot camera; only they get JPEG frames
        self._renderer = ThreadPoolExecutor(1, thread_name_prefix="render")
        self._frame = None  # pending render
        self.rgbd_clients = set()  # view-only /rgbd connections
        self._rgbd = None
        self.controller = None
        self.last_input = 0.0  # time of the controller's last real input
        self._ids = itertools.count(1)

    def hello(self, cid):
        a = self.adapter
        return {
            "type": "hello",
            "client_id": cid,
            "names": a.names,
            "lower": finite_or_none(a.lower),  # base yaw is unlimited; JSON has no inf
            "upper": finite_or_none(a.upper),
            "vmax": a.vmax.tolist(),
            "home": a.home.tolist(),
            "watchdog_s": self.gate.watchdog_s,
            "arms": list(self.teleop.arms),
            "max_speed": MAX_SPEED,
            "base": None if a.base_qadr is None else
                    {"body": a.model.body("base_link").id, "speed": BASE_SPEED, "turn": BASE_TURN},
            "camera": a.camera_params(),
            "idle_release_s": IDLE_RELEASE_S,
            "berries": {"ids": a.berry_ids, "idle": BERRY_IDLE, "active": BERRY_ACTIVE, "done": BERRY_DONE},
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
            "base": np.round(self.adapter.base_pose(), 3).tolist(),
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

            for i, side in self.adapter.detach_grasped():
                self.log.write("picked", berry=i, arm=side)
            a = self.adapter
            result = self.task.tick(a.berry_positions(), a.berry_attached(), a.body_positions(TCP.values()),
                                    time.monotonic())
            if result:
                self.log.write("task", event=result, index=self.task.index, seconds=self.task.results[-1])
                a.set_berry_colors(self.task.index if self.task.active else None, self.task.index)

            if tick % STATE_EVERY == 0:
                await self._broadcast(json.dumps(self.state()))
            if tick % VIDEO_EVERY == 0 and self.watching and (self._frame is None or self._frame.done()):
                self._frame = asyncio.create_task(self._send_frame(self.adapter.data.qpos.copy()))
            if tick % RGBD_EVERY == 0 and self.rgbd_clients and (self._rgbd is None or self._rgbd.done()):
                self._rgbd = asyncio.create_task(self._send_rgbd(self.adapter.data.qpos.copy()))

            if self.controller is not None and time.monotonic() - self.last_input > IDLE_RELEASE_S:
                await self._release_idle()

            next_t += LOOP_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    async def _release_idle(self):
        ws = self.controller
        self._release("idle: control released")
        self.log.write("release", client=self.clients.get(ws), reason="idle")
        try:
            await ws.send_json({"type": "control", "granted": False, "reason": "idle"})
        except ConnectionError:
            pass

    async def _send_frame(self, qpos):
        """Render off the control loop; a slow frame only lowers the frame rate."""
        jpeg = await asyncio.get_running_loop().run_in_executor(self._renderer, self.adapter.render_jpeg, qpos)
        await self._broadcast(jpeg, self.watching)

    async def _send_rgbd(self, qpos):
        """Color + depth for one snapshot, rendered and packed on the render thread."""
        frame = await asyncio.get_running_loop().run_in_executor(
            self._renderer, lambda: rgbd.pack(*self.adapter.render_rgbd(qpos)))
        await self._broadcast(frame, self.rgbd_clients)

    def rgbd_hello(self):
        return {"type": "rgbd_hello", "camera": rgbd.NAME, **rgbd.intrinsics(), "pitch": rgbd.PITCH,
                "near": rgbd.NEAR, "far": rgbd.FAR,
                "depth_unit": 0.001, "fps_max": round(1 / (RGBD_EVERY * LOOP_DT))}

    async def _broadcast(self, payload, targets=None):
        for ws in list(self.clients if targets is None else targets):
            try:
                await (ws.send_bytes if isinstance(payload, bytes) else ws.send_str)(payload)
            except ConnectionError:
                self.clients.pop(ws, None)
                self.watching.discard(ws)
                self.rgbd_clients.discard(ws)

    async def handle(self, ws, msg):
        kind, cid, now = msg.get("type"), self.clients[ws], time.monotonic()
        q_meas = self.adapter.read()[0]
        if ws is self.controller and (kind in ("activate", "home", "task_start", "cmd") or (
                kind in ("ee", "base") and (any(msg.get("v") or ()) or msg.get("grip")))):
            self.last_input = now  # zero-velocity keepalives don't count as input

        if kind == "ping":
            await ws.send_json({"type": "pong", "t": msg.get("t")})
        elif kind == "view":  # which view the client shows: JPEG frames only for the robot camera
            (self.watching.add if msg.get("video") else self.watching.discard)(ws)
        elif kind == "claim":
            granted = self.controller in (None, ws)
            if granted:
                self.controller = ws
                self.last_input = now
            self.log.write("claim", client=cid, granted=granted)
            await ws.send_json({"type": "control", "granted": granted})
        elif kind == "estop":  # any connected client may stop the robot
            self.gate.estop(q_meas, f"e-stop by client {cid}")
        elif ws is not self.controller:
            await ws.send_json({"type": "error", "error": "not_controller", "for": kind})
        elif kind == "activate":
            self.gate.activate(q_meas, now)
            self.teleop.reset(self.gate.goal, now)
        elif kind in ("ee", "base"):
            yaw = msg.get("yaw")  # set when the operator drives from the 3D view
            if (self.gate.mode != ACTIVE or len(msg.get("v", ())) != 3
                    or not (yaw is None or (isinstance(yaw, (int, float)) and math.isfinite(yaw)))
                    or (kind == "ee" and msg.get("arm") not in self.teleop.arms)):
                return
            if kind == "base":
                goal = self.teleop.drive(self.gate.goal, msg["v"], now, yaw=yaw)
                result = self.gate.command(goal, now)
                self.log.write("base", client=cid, seq=msg.get("seq"), v=msg["v"], yaw=yaw, result=result)
                return
            goal = self.teleop.step(self.gate.goal, msg["arm"], msg["v"], msg.get("grip", 0), now, yaw=yaw)
            result = self.gate.command(goal, now)
            self.log.write("ee", client=cid, seq=msg.get("seq"), arm=msg["arm"], v=msg["v"],
                           grip=msg.get("grip", 0), yaw=yaw, result=result)
        elif kind == "home":  # arms only; the base stays put
            if self.gate.mode == ACTIVE:
                goal = self.teleop.home(self.gate.goal, self.adapter.home)
                self.gate.command(goal, now)
                self.teleop.reset(goal, now)
        elif kind == "task_start":
            self.adapter.reset_berries()
            self.task.start(now)
            self.adapter.set_berry_colors(0, 0)
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
        self.watching.discard(ws)
        self.log.write("disconnect", client=cid)
        if ws is self.controller:
            self._release("controller disconnected")


async def ws_handler(request):
    edge = request.app["edge"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    cid = next(edge._ids)
    edge.clients[ws] = cid
    edge.log.write("connect", client=cid, who=request.get("who"))  # invite label; no IP addresses
    await ws.send_json(edge.hello(cid))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await edge.handle(ws, json.loads(msg.data))
    finally:
        edge.disconnect(ws)
    return ws


async def rgbd_handler(request):
    """View-only head RGB-D stream; needs no control, sends nothing back."""
    edge = request.app["edge"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    await ws.send_json(edge.rgbd_hello())
    edge.rgbd_clients.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        edge.rgbd_clients.discard(ws)
    return ws


@web.middleware
async def no_stale_ui(request, handler):
    """Make browsers revalidate UI files, so an updated app.js is never skipped."""
    resp = await handler(request)
    if request.path not in ("/ws", "/rgbd"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def finite_or_none(xs):
    return [float(x) if math.isfinite(x) else None for x in xs]


def main():
    p = argparse.ArgumentParser()
    # Cartesian teleop uses Enactic OpenArm v1 bimanual names (see edge/teleop.py).
    p.add_argument("--model", default=ROOT / "vendor/openarm_mujoco/v1/openarm_bimanual.xml", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8765, type=int)
    p.add_argument("--base", default=ROOT / "vendor/tidybot_base/base.xml", type=Path, help="mobile base MJCF")
    p.add_argument("--no-base", action="store_true", help="arms on their fixed column, no mobile base")
    p.add_argument("--codes", type=Path, help="invite-code file (see edge/access.py); without it the server is open")
    args = p.parse_args()

    base = None if args.no_base else args.base
    adapter = MujocoAdapter(args.model, base_path=base, greenhouse_rows=True)
    teleop = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper,
                             adapter.camera_frame())
    gate = SafetyGate(adapter.lower, adapter.upper, adapter.vmax)
    edge = Edge(adapter, gate, SessionLog(ROOT / "logs"), teleop, HarvestTask(adapter.berry_start))
    print(f"session log: {edge.log.path}")

    meta, blob = export_scene(adapter.model)
    print(f"3D scene: {len(meta['geoms'])} geoms, {meta['nvert']} vertices, {len(blob) / 1e6:.1f} MB")

    middlewares = [no_stale_ui] if args.codes is None else [access.middleware(access.Codes(args.codes)), no_stale_ui]
    app = web.Application(middlewares=middlewares)
    app["edge"] = edge
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/rgbd", rgbd_handler)
    app.router.add_get("/healthz", lambda r: web.json_response(
        {"ok": True, "clients": len(edge.clients), "driving": edge.controller is not None}))
    app.router.add_get("/scene.json", lambda r: web.json_response(meta))
    app.router.add_get("/scene.bin", lambda r: web.Response(body=blob, content_type="application/octet-stream"))
    app.router.add_get("/", lambda r: web.FileResponse(ROOT / "web/index.html"))
    app.router.add_static("/", ROOT / "web")

    async def start_loop(app):
        app["loop_task"] = asyncio.create_task(edge.run())

    async def stop(app):  # free the GL contexts on the thread that made them
        app["loop_task"].cancel()
        await asyncio.get_running_loop().run_in_executor(edge._renderer, adapter.close_renderers)

    app.on_startup.append(start_loop)
    app.on_cleanup.append(stop)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
