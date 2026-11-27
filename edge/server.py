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
from edge.autopilot import PLANS, TASKS, Autopilot
from edge.episodes import Recorder
from edge.mujoco_adapter import BERRY_ACTIVE, BERRY_DONE, SCENES, MujocoAdapter
from edge.safety import ACTIVE, ESTOP, SafetyGate
from edge.scene import body_poses, export_scene
from edge.task import HarvestTask
from edge.teleop import BASE_SPEED, BASE_TURN, MAX_SPEED, TCP, CartesianTeleop, rot_z

ROOT = Path(__file__).resolve().parent.parent
LOOP_DT = 0.01
STATE_EVERY = 5  # ticks -> 20 Hz
VIDEO_EVERY = 7  # ticks -> ~14 fps
RGBD_EVERY = 10  # ticks -> at most 10 fps; in practice limited by render time
IDLE_RELEASE_S = 180  # control is released after this long without operator input, so a left-open tab can't hog it
# Limits for a server on the internet (a leaked invite link or a buggy client must not take it down):
MAX_CLIENTS = 30  # /ws connections; each costs ~60 KB/s of state
MAX_RGBD_CLIENTS = 6  # /rgbd connections; each costs ~175 KB/s
MAX_MSG_BYTES = 16 * 1024  # real messages are < 200 bytes
MAX_MSG_PER_S = 100  # the UI sends ~25/s; a client above this is closed (1008)


def _reject(token):
    raise ValueError(f"non-finite number {token}")


def _finite(s):
    x = float(s)
    if not math.isfinite(x):  # 1e999 parses to inf
        _reject(s)
    return x


def parse_message(data):
    """A client message as a dict, or None if malformed. NaN/Infinity are refused here:
    np.clip passes NaN through, so it would otherwise reach the joint goal."""
    try:
        msg = json.loads(data, parse_constant=_reject, parse_float=_finite)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def _seq(msg):  # logged per command; only a client counter, never arbitrary client data
    s = msg.get("seq")
    return s if isinstance(s, int) else None


class SessionLog:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._f = self.path.open("a", buffering=1)

    def write(self, kind, **fields):
        self._f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}) + "\n")


class Edge:
    def __init__(self, adapter, gate, log, teleop, task, autopilot=None, recorder=None):
        self.log, self.recorder = log, recorder
        self.worlds = {}  # scene name -> adapter, gate, teleop, task, autopilot (one robot, one scene at a time)
        self.exports = {}  # scene name -> (scene.json meta, scene.bin blob) for the 3D view
        self.add_world(adapter, gate, teleop, task, autopilot)
        self.use(adapter.scene)
        self.clients = {}  # ws -> client id
        self.watching = set()  # clients showing the robot camera; only they get JPEG frames
        self._renderer = ThreadPoolExecutor(1, thread_name_prefix="render")
        self._frame = None  # pending render
        self.rgbd_clients = set()  # view-only /rgbd connections
        self._rgbd = None
        self.controller = None
        self.last_input = 0.0  # time of the controller's last real input
        self._ids = itertools.count(1)
        self._action = None  # command applied since the last recorded step
        self._cut = None  # target whose episode was cut for length; not reopened
        self.clock = time.monotonic  # tests swap in a simulated clock
        self.record_rgbd = True  # head RGB-D frames go into the episodes too

    def add_world(self, adapter, gate, teleop, task, autopilot=None):
        self.worlds[adapter.scene] = dict(adapter=adapter, gate=gate, teleop=teleop, task=task, autopilot=autopilot)

    def use(self, scene):
        w = self.worlds[scene]
        self.scene = scene
        self.adapter, self.gate, self.teleop, self.task, self.autopilot = (
            w["adapter"], w["gate"], w["teleop"], w["task"], w["autopilot"])

    def hello(self, cid):
        a = self.adapter
        return {
            "type": "hello",
            "client_id": cid,
            "scene": self.scene,
            "scenes": [s for s in self.worlds if s],
            "autonomy": self.autopilot is not None,
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
            "berries": {"ids": a.berry_ids, "idle": a.target_rgba, "active": BERRY_ACTIVE, "done": BERRY_DONE},
        }

    def state(self):
        q, _ = self.adapter.read()
        g, rec = self.gate, self.recorder
        return {
            "type": "state",
            "scene": self.scene,
            "mode": g.mode,
            "reason": g.reason,
            "controller": self.clients.get(self.controller),
            "q": q.tolist(),
            "goal": g.goal.tolist(),
            "target": g.target.tolist(),
            "task": self.task.summary(self.clock()),
            "auto": None if self.autopilot is None else self.autopilot.summary(),
            "actor": self.actor(),
            "rec": None if rec is None or rec.ep is None else {"id": rec.ep["id"], "t": rec.t(self.clock())},
            "bodies": body_poses(self.adapter.data),
            "base": np.round(self.adapter.base_pose(), 3).tolist(),
        }

    def actor(self):
        """Who is commanding the robot right now: "human", "auto" or "none"."""
        if self.gate.mode != ACTIVE:
            return "none"
        if self.controller is not None:
            return "human"
        ap = self.autopilot
        return "auto" if ap is not None and ap.running and ap.ticket is None else "none"

    async def run(self):
        next_t = self.clock()
        for tick in itertools.count():
            await self.tick(tick)
            next_t += LOOP_DT
            await asyncio.sleep(max(0.0, next_t - self.clock()))

    async def tick(self, tick):
        """One 100 Hz control-loop step."""
        now = self.clock()
        a = self.adapter
        q, _ = a.read()
        a.write(self.gate.tick(q, now, LOOP_DT))
        a.step(LOOP_DT)
        for mode, reason in self.gate.events:
            self.log.write("mode", mode=mode, reason=reason)
            self._event(now, "mode", mode=mode, reason=reason)
        self.gate.events.clear()

        for i, side in a.detach_grasped():
            self.log.write("picked", berry=i, arm=side)
            self._event(now, "detached", target=i, arm=side)
        result = self.task.tick(a.berry_positions(), a.berry_attached(), a.body_positions(TCP.values()), now,
                                a.berry_placed())
        if result:
            await self._progress(result, now)

        if tick % STATE_EVERY == 0:
            self._drive_auto(now)
            self._record(now)
            await self._broadcast(json.dumps(self.state()))
        if tick % VIDEO_EVERY == 0 and self.watching and (self._frame is None or self._frame.done()):
            self._frame = asyncio.create_task(self._send_frame(a, a.data.qpos.copy()))
        rec = self.recorder
        record = rec is not None and rec.wants_rgbd(now)
        if (tick % RGBD_EVERY == 0 and (self.rgbd_clients or record)
                and (self._rgbd is None or self._rgbd.done())):
            ep_id = rec.ep["id"] if record else None
            if record:
                rec.mark_rgbd(now)
            self._rgbd = asyncio.create_task(self._send_rgbd(a, a.data.qpos.copy(), now, ep_id))

        if self.controller is not None and self.clock() - self.last_input > IDLE_RELEASE_S:
            await self._release_idle()

    async def _progress(self, result, now):
        """The current target was done or skipped."""
        seconds = self.task.results[-1]
        self.log.write("task", event=result, index=self.task.index, seconds=seconds)
        self.adapter.set_berry_colors(self.task.index if self.task.active else None, self.task.index)
        if seconds is None:
            self._close_episode(now, "skipped", "operator gave up", by="human")
        else:
            self._close_episode(now, "success", "target done")
        ap = self.autopilot
        if ap is not None and ap.running and self.controller is not None:  # an operator helped: hand back
            ws = self.controller
            self._release("solved: back to the robot" if seconds is not None else "skipped: back to the robot")
            ap.solved(now)
            try:
                await ws.send_json({"type": "control", "granted": False,
                                    "reason": "solved" if seconds is not None else "skipped"})
            except ConnectionError:
                pass

    def _drive_auto(self, now):
        """One step of the autopilot (20 Hz), when nobody else is driving."""
        ap = self.autopilot
        if ap is None or not ap.running:
            return
        if not self.task.active or self.gate.mode == ESTOP:
            reason = "all targets done" if not self.task.active else "e-stop"
            ap.stop()
            self.log.write("auto", event="stop", reason=reason)
            if self.gate.mode == ACTIVE and self.controller is None:
                self.gate.hold(f"autonomous run ended: {reason}")
            return
        if self.controller is not None or ap.ticket is not None:
            return  # an operator is driving, or the robot waits for one
        if self.gate.mode != ACTIVE:
            self.gate.activate(self.adapter.read()[0], now, "robot working on its own")
            self.teleop.reset(self.gate.goal, now)
        msg, ticket = ap.tick(self._obs(), now)
        if ticket:
            self.gate.hold(f"help needed: {ticket['text']}")
            self.log.write("ticket", index=ticket["index"], code=ticket["code"])
            if self.recorder is not None:
                self.recorder.ticket(now, ticket["code"], ticket["text"])
        elif msg:
            self._apply(msg, now)

    def _obs(self):
        a, t, i = self.adapter, self.task, self.task.index
        base = None if self.teleop.base is None else self.gate.goal[self.teleop.base]
        origin, yaw = (np.zeros(3), 0.0) if base is None else (np.array([base[0], base[1], 0.0]), base[2])
        return {
            "index": i, "target": t.starts[i], "target_pos": a.berry_positions()[i],
            "attached": a.berry_attached()[i], "base_goal": base,
            "tcp_target": {s: origin + rot_z(yaw) @ p for s, p in self.teleop.targets.items()},
        }

    def _apply(self, msg, now, cid=None):
        """Apply one ee / base / home command from an operator (cid) or the autopilot (None)."""
        kind, yaw = msg["type"], msg.get("yaw")
        if self.gate.mode != ACTIVE:
            return
        if kind == "home":  # arms only; the base stays put
            goal = self.teleop.home(self.gate.goal, self.adapter.home)
            self.gate.command(goal, now)
            self.teleop.reset(goal, now)
            self._action = {"kind": "home"}
            return
        if kind == "base":
            goal = self.teleop.drive(self.gate.goal, msg["v"], now, yaw=yaw)
            self._action = {"kind": "base", "v": list(msg["v"]), "yaw": yaw}
        else:
            goal = self.teleop.step(self.gate.goal, msg["arm"], msg["v"], msg.get("grip", 0), now, yaw=yaw)
            self._action = {"kind": "ee", "arm": msg["arm"], "v": list(msg["v"]), "grip": msg.get("grip", 0), "yaw": yaw}
        result = self.gate.command(goal, now)
        if cid is not None:  # the autopilot's commands are in the episode files, not the session log
            self.log.write(kind, client=cid, seq=_seq(msg), **{k: v for k, v in self._action.items() if k != "kind"},
                           result=result)

    def _record(self, now):
        rec, t, a = self.recorder, self.task, self.adapter
        if rec is None or not t.active or t.index == self._cut:
            return
        if rec.ep is None or rec.index != t.index:
            self._close_episode(now, "aborted", "target changed")
            name = a.model.geom(a.berry_ids[t.index]).name
            h = self.hello(None)
            cam = {k: v for k, v in self.rgbd_hello().items() if k not in ("type", "fps_max")}
            rec.open(now, self.scene, TASKS.get(self.scene, "task"), t.index, name, t.starts[t.index], a.names,
                     view={k: h[k] for k in ("camera", "base", "berries")}, rgbd=cam if self.record_rgbd else None)
        tcp = dict(zip(TCP, a.body_positions(TCP.values())))
        ok = rec.step(now, self.actor(), self.gate.mode, a.read()[0], self.gate.goal, a.base_pose(), tcp,
                      a.berry_positions()[t.index], a.berry_attached()[t.index], self._action,
                      np.hstack([a.data.xpos, a.data.xquat]).ravel())
        self._action = None
        if not ok:
            self._cut = t.index
            self._close_episode(now, "aborted", "too long")

    def _close_episode(self, now, result, reason, by=None):
        if self.recorder is not None:
            entry = self.recorder.close(now, result, reason, by)
            if entry:
                self.log.write("episode", id=entry["id"], result=result)

    def _event(self, now, kind, **fields):
        if self.recorder is not None:
            self.recorder.event(now, kind, **fields)

    async def _release_idle(self):
        ws = self.controller
        self._release("idle: control released")
        self.log.write("release", client=self.clients.get(ws), reason="idle")
        try:
            await ws.send_json({"type": "control", "granted": False, "reason": "idle"})
        except ConnectionError:
            pass

    async def _send_frame(self, adapter, qpos):
        """Render off the control loop; a slow frame only lowers the frame rate."""
        jpeg = await asyncio.get_running_loop().run_in_executor(self._renderer, adapter.render_jpeg, qpos)
        if adapter is self.adapter:  # not a frame of the scene we just switched away from
            await self._broadcast(jpeg, self.watching)

    async def _send_rgbd(self, adapter, qpos, t=None, episode_id=None):
        """Color + depth for one snapshot (taken at t), rendered and packed on the render
        thread; sent to /rgbd viewers and stored in the episode it was taken for."""
        frame = await asyncio.get_running_loop().run_in_executor(
            self._renderer, lambda: rgbd.pack(*adapter.render_rgbd(qpos)))
        if episode_id is not None:
            self.recorder.add_rgbd(t, episode_id, frame)
        if adapter is self.adapter:
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

    def _take(self, ws, now, reason):
        """Give control to ws (claim or ticket accept)."""
        self.controller = ws
        self.last_input = now
        self._event(now, "takeover", client=self.clients[ws], reason=reason)

    async def handle(self, ws, msg):
        kind, cid, now = msg.get("type"), self.clients[ws], self.clock()
        q_meas = self.adapter.read()[0]
        ap = self.autopilot
        if ws is self.controller and (kind in ("activate", "home", "task_start", "cmd") or (
                kind in ("ee", "base") and (any(msg.get("v") or ()) or msg.get("grip")))):
            self.last_input = now  # zero-velocity keepalives don't count as input

        if kind == "ping":
            await ws.send_json({"type": "pong", "t": msg.get("t")})
        elif kind == "view":  # which view the client shows: JPEG frames only for the robot camera
            (self.watching.add if msg.get("video") else self.watching.discard)(ws)
        elif kind == "claim":
            granted = self.controller in (None, ws)
            if granted and self.controller is None:
                self._take(ws, now, "operator" if ap is not None and ap.running else "claim")
            self.log.write("claim", client=cid, granted=granted)
            await ws.send_json({"type": "control", "granted": granted})
        elif kind == "accept":  # take an open help ticket: control, already active
            granted = ap is not None and ap.ticket is not None and self.controller is None
            if granted:
                self._take(ws, now, ap.ticket["code"])
                self.gate.activate(q_meas, now, "operator took the help ticket")
                self.teleop.reset(self.gate.goal, now)
            self.log.write("accept", client=cid, granted=granted)
            await ws.send_json({"type": "control", "granted": granted, "ticket": granted})
        elif kind == "estop":  # any connected client may stop the robot
            self.gate.estop(q_meas, f"e-stop by client {cid}")
        elif kind == "auto_stop":  # stopping is always allowed, like the e-stop
            if ap is not None and ap.running:
                ap.stop()
                self.task.active = False
                self._close_episode(now, "aborted", "autonomous run stopped")
                if self.controller is None:
                    self.gate.hold("autonomous run stopped")
                self.log.write("auto", event="stop", client=cid)
        elif kind in ("auto_start", "scene") and self.controller is not None:
            await ws.send_json({"type": "error", "error": "someone is driving", "for": kind})
        elif kind == "auto_start":
            if ap is None:
                return
            self._close_episode(now, "aborted", "restarted")
            self.adapter.reset_berries()
            self.task.start(now)
            self.adapter.set_berry_colors(0, 0)
            self._cut = None
            ap.start(now)
            self.log.write("auto", event="start", client=cid)
        elif kind == "scene":
            name = msg.get("name")
            if name in self.worlds and name != self.scene and not (ap is not None and ap.running):
                self._close_episode(now, "aborted", "scene changed")
                if self.gate.mode == ACTIVE:
                    self.gate.hold("scene changed")
                self.use(name)
                self.log.write("scene", name=name, client=cid)
                await self._broadcast(json.dumps({"type": "reload"}))
        elif ws is not self.controller:
            await ws.send_json({"type": "error", "error": "not_controller", "for": kind})
        elif kind == "activate":
            self.gate.activate(q_meas, now)
            self.teleop.reset(self.gate.goal, now)
        elif kind in ("ee", "base"):
            yaw = msg.get("yaw")  # set when the operator drives from the 3D view
            if (len(msg.get("v", ())) != 3
                    or not (yaw is None or (isinstance(yaw, (int, float)) and math.isfinite(yaw)))
                    or (kind == "ee" and msg.get("arm") not in self.teleop.arms)):
                return
            self._apply(msg, now, cid)
        elif kind == "home":
            self._apply({"type": "home"}, now, cid)
        elif kind == "skip":  # give up on the current target (e.g. a ticket that can't be solved)
            result = self.task.skip(now)
            if result:
                await self._progress(result, now)
        elif kind == "task_start":
            if ap is not None and ap.running:
                return
            self._close_episode(now, "aborted", "restarted")
            self.adapter.reset_berries()
            self.task.start(now)
            self.adapter.set_berry_colors(0, 0)
            self._cut = None
            self.log.write("task", event="start", client=cid)
        elif kind == "release":
            self._release("controller released")
        elif kind == "cmd":
            try:
                result = self.gate.command(msg["q"], now)
            except (KeyError, ValueError) as e:
                await ws.send_json({"type": "error", "error": str(e), "for": kind})
                return
            self.log.write("cmd", client=cid, seq=_seq(msg), q=msg["q"], result=result)

    def _release(self, reason):
        now = self.clock()
        self._event(now, "release", client=self.clients.get(self.controller), reason=reason)
        self.controller = None
        if self.gate.mode == ACTIVE:
            self.gate.hold(reason)
        ap = self.autopilot
        if ap is not None and ap.running and ap.ticket is None:
            ap.resume(now)  # an operator took over without a ticket: the robot starts this target over

    def disconnect(self, ws):
        cid = self.clients.pop(ws, None)
        self.watching.discard(ws)
        self.log.write("disconnect", client=cid)
        if ws is self.controller:
            self._release("controller disconnected")


async def ws_handler(request):
    edge = request.app["edge"]
    if len(edge.clients) >= MAX_CLIENTS:
        return web.Response(status=503, text="server full")
    ws = web.WebSocketResponse(max_msg_size=MAX_MSG_BYTES, heartbeat=30)
    await ws.prepare(request)
    cid = next(edge._ids)
    edge.clients[ws] = cid
    edge.log.write("connect", client=cid, who=request.get("who"))  # invite label; no IP addresses
    await ws.send_json(edge.hello(cid))
    window, count = time.monotonic(), 0
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            now = time.monotonic()
            window, count = (window, count + 1) if now - window < 1 else (now, 1)
            if count > MAX_MSG_PER_S:
                edge.log.write("flood", client=cid)
                await ws.close(code=1008, message=b"too many messages")
                break
            m = parse_message(msg.data)
            if m is None:
                continue
            try:
                await edge.handle(ws, m)
            except (TypeError, ValueError, KeyError, IndexError):  # wrong field types: drop the message
                pass
    finally:
        edge.disconnect(ws)
    return ws


async def rgbd_handler(request):
    """View-only head RGB-D stream; needs no control, sends nothing back."""
    edge = request.app["edge"]
    if len(edge.rgbd_clients) >= MAX_RGBD_CLIENTS:
        return web.Response(status=503, text="server full")
    ws = web.WebSocketResponse(max_msg_size=1024, heartbeat=30)
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


def _export(request):
    edge = request.app["edge"]
    return edge.exports.get(request.query.get("scene", edge.scene))


async def scene_json(request):  # ?scene=name: another scene's geometry, for episode replay
    e = _export(request)
    return web.json_response(e[0]) if e else web.Response(status=404)


async def scene_bin(request):
    e = _export(request)
    return web.Response(body=e[1], content_type="application/octet-stream") if e else web.Response(status=404)


async def episode_frame(request):
    path = request.app["edge"].recorder.frame_path(request.match_info["id"], request.match_info["frame"])
    return web.FileResponse(path) if path else web.Response(status=404)


async def episode_file(request):
    path = request.app["edge"].recorder.path(request.match_info["id"])
    return web.FileResponse(path) if path else web.Response(status=404)


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
    p.add_argument("--scenes", default="strawberry,weeding", help=f"comma-separated, first one shown: {', '.join(SCENES)}")
    p.add_argument("--codes", type=Path, help="invite-code file (see edge/access.py); without it the server is open")
    args = p.parse_args()

    base = None if args.no_base else args.base
    worlds = []
    for name in args.scenes.split(","):
        adapter = MujocoAdapter(args.model, base_path=base, scene=name)
        teleop = CartesianTeleop(adapter.model, adapter.names, adapter.home, adapter.lower, adapter.upper,
                                 adapter.camera_frame())
        gate = SafetyGate(adapter.lower, adapter.upper, adapter.vmax)
        worlds.append((adapter, gate, teleop, HarvestTask(adapter.berry_start), Autopilot(PLANS[name])))
    (adapter, gate, teleop, task, ap), *more = worlds
    edge = Edge(adapter, gate, SessionLog(ROOT / "logs"), teleop, task, autopilot=ap,
                recorder=Recorder(ROOT / "episodes"))
    for w in more:
        edge.add_world(*w)
    print(f"session log: {edge.log.path}")
    for name, w in edge.worlds.items():
        meta, blob = edge.exports[name] = export_scene(w["adapter"].model)
        meta["ground"] = getattr(SCENES[name], "GROUND", None)  # "soil": the 3D view draws soil, no floor grid
        print(f"3D scene {name}: {len(meta['geoms'])} geoms, {meta['nvert']} vertices, {len(blob) / 1e6:.1f} MB")

    middlewares = [no_stale_ui] if args.codes is None else [access.middleware(access.Codes(args.codes)), no_stale_ui]
    app = web.Application(middlewares=middlewares)
    app["edge"] = edge
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/rgbd", rgbd_handler)
    app.router.add_get("/healthz", lambda r: web.json_response(
        {"ok": True, "clients": len(edge.clients), "driving": edge.controller is not None}))
    app.router.add_get("/scene.json", scene_json)
    app.router.add_get("/scene.bin", scene_bin)
    app.router.add_get("/episodes", lambda r: web.json_response(edge.recorder.list()))
    app.router.add_get("/episodes/{id}", episode_file)
    app.router.add_get("/episodes/{id}/rgbd/{frame}", episode_frame)
    app.router.add_get("/", lambda r: web.FileResponse(ROOT / "web/index.html"))
    app.router.add_static("/", ROOT / "web")

    async def start_loop(app):
        app["loop_task"] = asyncio.create_task(edge.run())

    async def stop(app):  # free the GL contexts on the thread that made them
        app["loop_task"].cancel()
        edge._close_episode(time.monotonic(), "aborted", "server stopped")
        for w in edge.worlds.values():
            await asyncio.get_running_loop().run_in_executor(edge._renderer, w["adapter"].close_renderers)

    app.on_startup.append(start_loop)
    app.on_cleanup.append(stop)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
