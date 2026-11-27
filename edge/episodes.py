"""Episode recorder: one file per attempt at one task target (a berry or a weed).

An episode runs from the moment a target becomes the current one until it is done
(success), skipped by an operator, or cut short (aborted). Whoever acts in between,
the robot on its own ("auto") or an operator ("human"), is recorded step by step in the
same columns, so the hand-overs (help tickets, takeovers) are part of the data.

File: episodes/<id>.json, plus one line per episode in episodes/index.jsonl (for listing),
and the head RGB-D frames in episodes/<id>/rgbd/00000.bin, 00001.bin, ... (one file per frame,
the /rgbd wire format: uint32 LE n | n bytes JPEG color | zlib(uint16 LE depth in mm, row-major),
depth 0 = no return; see edge/rgbd.py).

    {
      "format": "openarm-episode/v0",
      "id", "scene", "task", "target": {"index", "name", "start": [x, y, z]},
      "started_at": ISO time (UTC), "duration_s",
      "outcome": {"result": "success" | "skipped" | "aborted", "reason", "by": "auto" | "human" | null},
      "segments": [{"actor": "auto" | "human" | "none", "t0", "t1"}],   # runs of one actor
      "tickets": [{"t", "code", "text"}],
      "events": [{"t", "kind", ...}],          # mode changes, takeovers, detach, release
      "joint_names": [...],                    # actuator order of q / goal
      "rate_hz": 20,
      "steps": {                               # columns, one entry per step
        "t":        seconds since the episode began,
        "actor":    "auto" | "human" | "none" (nobody commanding: waiting for help, held, e-stop),
        "mode":     safety gate mode: "active" | "hold" | "estop",
        "q":        measured joint positions (actuator space),
        "goal":     joint goal after the safety gate's clamp,
        "base":     [x, y, yaw] of the mobile base (world),
        "tcp":      {"left": [x, y, z], "right": [x, y, z]} gripper centers (world),
        "target":   target position (world); "attached": still on its stem / in the soil,
        "action":   the command applied in this step, or null:
                    {"kind": "ee", "arm", "v": [right, fwd, up] -1..1, "grip": 0..1, "yaw"}
                    {"kind": "base", "v": [right, fwd, turn] -1..1, "yaw"} | {"kind": "home"}
      },
      "bodies_hz": 10, "bodies": [[x, y, z, qw, qx, qy, qz] * nbody per frame],  # 3D replay only
      "view": {"camera", "base", "berries"},   # what the 3D view needs, as in the server's hello
      "rgbd": {"camera": {width, height, fx, fy, cx, cy, pitch, near, far, depth_unit},
               "frames": [{"t", "file"}]}      # head camera, at up to RGBD_HZ (fewer if rendering is slow)
    }

Velocities are stick deflections in the frame of the view the command was given in (yaw:
heading of that view, rad from +x); the robot runs at 0.15 m/s per full deflection.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

FORMAT = "openarm-episode/v0"
RATE_HZ = 20
BODY_EVERY = 2  # steps -> 10 Hz for the 3D replay
MAX_S = 300  # an episode longer than this is cut (aborted); bounds memory
RGBD_HZ = 5  # head RGB-D frames per second, at most; each ~0.1 MB
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
FRAME_RE = re.compile(r"^\d{5}\.bin$")


def _r(x, nd=4):
    return np.round(np.asarray(x, float), nd).tolist()


class Recorder:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ep = None  # the open episode
        self._t0 = None

    @property
    def index(self):
        return None if self.ep is None else self.ep["target"]["index"]

    def open(self, now, scene, task, index, name, start, joint_names, view=None, rgbd=None):
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self._t0 = now
        self.ep = {
            "format": FORMAT,
            "id": f"{stamp}-{int(time.time() * 1000) % 1000:03d}-{scene}-{name}",
            "scene": scene, "task": task,
            "target": {"index": index, "name": name, "start": _r(start)},
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_s": 0.0, "outcome": None, "segments": [], "tickets": [], "events": [],
            "joint_names": list(joint_names), "rate_hz": RATE_HZ,
            "steps": {k: [] for k in ("t", "actor", "mode", "q", "goal", "base", "tcp", "target", "attached", "action")},
            "bodies_hz": RATE_HZ // BODY_EVERY, "bodies": [],
            "view": view,  # 3D replay: robot camera, base body, target geoms and colors
            "rgbd": None if rgbd is None else {"camera": rgbd, "frames": []},
        }
        self._last_rgbd = None

    def wants_rgbd(self, now):
        """Is a head RGB-D frame due for the open episode?"""
        return (self.ep is not None and self.ep["rgbd"] is not None
                and (self._last_rgbd is None or now - self._last_rgbd >= 1 / RGBD_HZ))

    def add_rgbd(self, now, episode_id, frame):
        """Store one packed frame taken at `now` (it is rendered off the control loop, so it
        may arrive after the episode moved on: then it is dropped)."""
        if self.ep is None or self.ep["id"] != episode_id or self.ep["rgbd"] is None:
            return
        frames = self.ep["rgbd"]["frames"]
        name = f"{len(frames):05d}.bin"
        d = self.dir / episode_id / "rgbd"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(frame)
        frames.append({"t": self.t(now), "file": name})

    def mark_rgbd(self, now):
        self._last_rgbd = now

    def t(self, now):
        return round(now - self._t0, 3)

    def step(self, now, actor, mode, q, goal, base, tcp, target, attached, action, bodies):
        s = self.ep["steps"]
        n = len(s["t"])
        s["t"].append(self.t(now))
        s["actor"].append(actor)
        s["mode"].append(mode)
        s["q"].append(_r(q))
        s["goal"].append(_r(goal))
        s["base"].append(_r(base, 3))
        s["tcp"].append({k: _r(v) for k, v in tcp.items()})
        s["target"].append(_r(target))
        s["attached"].append(bool(attached))
        s["action"].append(action)
        if n % BODY_EVERY == 0:
            self.ep["bodies"].append(_r(bodies, 3))
        return now - self._t0 < MAX_S

    def event(self, now, kind, **fields):
        if self.ep is not None:
            self.ep["events"].append({"t": self.t(now), "kind": kind, **fields})

    def ticket(self, now, code, text):
        if self.ep is not None:
            self.ep["tickets"].append({"t": self.t(now), "code": code, "text": text})

    def close(self, now, result, reason, by=None):
        """Write the episode; returns its index entry (None if nothing was open).
        by: who decided the outcome; default: whoever acted last."""
        ep, self.ep = self.ep, None
        if ep is None:
            return None
        steps = ep["steps"]
        ep["duration_s"] = round(now - self._t0, 3)
        ep["segments"] = segments(steps["t"], steps["actor"], ep["duration_s"])
        acting = [a for a in steps["actor"] if a != "none"]
        ep["outcome"] = {"result": result, "reason": reason, "by": by or (acting[-1] if acting and result == "success" else None)}
        (self.dir / f"{ep['id']}.json").write_text(json.dumps(ep, separators=(",", ":")))
        entry = summary(ep)
        with (self.dir / "index.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def list(self, limit=200):
        """Newest first."""
        path = self.dir / "index.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-limit:]
        return [json.loads(x) for x in reversed(lines) if x.strip()]

    def frame_path(self, episode_id, name):
        if not ID_RE.match(episode_id) or not FRAME_RE.match(name):
            return None
        p = self.dir / episode_id / "rgbd" / name
        return p if p.exists() else None

    def path(self, episode_id):
        if not ID_RE.match(episode_id):
            return None
        p = self.dir / f"{episode_id}.json"
        return p if p.exists() else None


def segments(t, actor, end):
    """Runs of the same actor: [{"actor", "t0", "t1"}]."""
    out = []
    for ti, a in zip(t, actor):
        if out and out[-1]["actor"] == a:
            continue
        if out:
            out[-1]["t1"] = ti
        out.append({"actor": a, "t0": ti, "t1": end})
    return out


def summary(ep):
    return {
        "id": ep["id"], "scene": ep["scene"], "task": ep["task"], "target": ep["target"]["name"],
        "started_at": ep["started_at"], "duration_s": ep["duration_s"], "outcome": ep["outcome"],
        "actors": sorted({s["actor"] for s in ep["segments"]} - {"none"}),
        "tickets": [k["code"] for k in ep["tickets"]], "steps": len(ep["steps"]["t"]),
        "rgbd_frames": 0 if ep.get("rgbd") is None else len(ep["rgbd"]["frames"]),
    }
