"""Scripted autonomy: works through the task targets in order and asks for help when stuck.

It sends the same messages an operator's browser sends (stick velocities, grip, home), so
its commands pass the same safety gate and are recorded in the same format as a human's.

It has no perception: it knows where each target is from the scene, not from a camera.
Its help tickets come from two places:
  - targets the scene marks as hard (stand-ins for what a perception stack would be unsure
    about, such as an occluded berry or a weed right next to a seedling), and
  - its own failures: a phase that runs past its time limit.
"""

from dataclasses import dataclass, field

import numpy as np

GAIN = 3.0  # 1/s: position error -> velocity
TOL = 0.008  # m: an arm waypoint counts as reached
BASE_TOL = 0.02  # m
HOME_S = 2.0  # time for the arms to get home (the gate limits joint speed)
CLOSE_S = 1.0  # time to close the gripper
RELEASE_S = 3.0  # time for a dropped berry to settle in the basket
TIME_LIMIT = {"drive": 30.0, "pre": 8.0, "reach": 6.0, "pull": 6.0, "carry": 8.0, "release": RELEASE_S}  # s
HELD = 0.05  # m; further than this from the gripper while carrying = dropped


@dataclass
class Plan:
    """Where to put the base and the gripper relative to a target. Offsets are world-frame
    (dx, dy, dz); dy is multiplied by the target's side (+1 left row, -1 right row)."""
    base_dx: float  # base x = target x + base_dx (the base stays on the aisle center line)
    pre: tuple  # first waypoint, gripper open
    grasp: tuple  # where the gripper closes
    pull: tuple  # where to take the target after closing; the task counts it on the way
    hard: dict = field(default_factory=dict)  # target index -> (code, text): raise a ticket instead
    place: tuple = None  # base frame (x, y, z), y times side: carry it here and let go; None: just pull


def _at(target, offset, side):
    dx, dy, dz = offset
    return np.asarray(target, float) + [dx, side * dy, dz]


class Autopilot:
    def __init__(self, plan):
        self.plan = plan
        self.running = False
        self.phase = None
        self.index = None  # target this run is working on
        self.since = 0.0  # when the phase began
        self.ticket = None  # open ticket: {"index", "code", "text"}

    def start(self, now):
        self.running, self.ticket = True, None
        self._go("home", now)
        self.index = None

    def stop(self):
        self.running, self.phase, self.ticket = False, None, None

    def resume(self, now):
        """Start over on the current target, e.g. after an operator took over without solving it."""
        if self.running:
            self._go("home", now)

    def solved(self, now):
        """The open ticket was solved (or skipped) by an operator: go on with the next target."""
        self.ticket = None
        self.resume(now)

    def _go(self, phase, now):
        self.phase, self.since = phase, now
        self._home_sent = False

    def _raise(self, index, code, text, now):
        self.ticket = {"index": index, "code": code, "text": text}
        self._go("waiting", now)
        return self.ticket

    def tick(self, obs, now):
        """obs: index, target (where it started), target_pos, attached, base_goal (x, y, yaw),
        tcp_target {side: world position the arm is commanded to}.
        Returns (message or None, new ticket or None)."""
        if not self.running or self.ticket is not None:
            return None, None
        i, p = obs["index"], self.plan
        if i != self.index:  # a new target (start, or the last one was picked)
            self.index = i
            self._go("home", now)
        side = 1.0 if obs["target"][1] > 0 else -1.0
        arm = "left" if side > 0 else "right"
        elapsed = now - self.since

        if self.phase == "home":
            if not self._home_sent:
                self._home_sent = True
                return {"type": "home"}, None
            if elapsed < HOME_S:
                return _ee(arm, np.zeros(3), 0.0), None  # keepalive: the gate holds without commands
            if i in p.hard:
                return None, self._raise(i, *p.hard[i], now)
            self._go("drive", now)

        if elapsed > TIME_LIMIT.get(self.phase, np.inf):
            if self.phase == "release":
                code, text = "missed_basket", "let go, but the berry did not land in the basket"
            elif self.phase == "pull":
                code, text = ("not_detached", "grasp failed: still attached") if obs["attached"] else \
                             ("dropped", "lost it while pulling")
            else:
                code, text = "timeout", f"could not reach the {self.phase} position"
            return None, self._raise(i, code, text, now)

        if self.phase == "drive" and obs["base_goal"] is None:  # arms on a fixed column
            self._go("pre", now)
        if self.phase == "drive":
            goal = np.array([obs["target"][0] + p.base_dx, 0.0])
            err = goal - np.asarray(obs["base_goal"][:2])
            if np.linalg.norm(err) < BASE_TOL:
                self._go("pre", now)
            else:
                return _base(err), None

        tcp = np.asarray(obs["tcp_target"][arm])
        if self.phase == "pre":
            return self._move(arm, tcp, _at(obs["target"], p.pre, side), 0.0, "reach", now)
        if self.phase == "reach":
            return self._move(arm, tcp, _at(obs["target"], p.grasp, side), 0.0, "close", now)
        if self.phase == "close":
            if elapsed >= CLOSE_S:
                self._go("pull", now)
            return _ee(arm, _vel(_at(obs["target"], p.grasp, side) - tcp), 1.0), None
        if self.phase == "pull":
            goal = _at(obs["target"], p.pull, side)
            if p.place is not None and np.linalg.norm(goal - tcp) < TOL:
                if obs["attached"]:
                    return None, self._raise(i, "not_detached", "grasp failed: still attached", now)
                self._go("carry", now)
            return _ee(arm, _vel(goal - tcp), 1.0), None
        if self.phase == "carry":
            if np.linalg.norm(np.asarray(obs["target_pos"]) - tcp) > HELD:
                return None, self._raise(i, "dropped", "dropped it on the way to the basket", now)
            bx, by, yaw = obs["base_goal"]
            c, s = np.cos(yaw), np.sin(yaw)
            px, py, pz = p.place
            goal = np.array([bx + c * px - s * side * py, by + s * px + c * side * py, pz])
            return self._move(arm, tcp, goal, 1.0, "release", now)
        if self.phase == "release":  # open and wait: the task counts the berry once it rests in the basket
            return _ee(arm, np.zeros(3), 0.0), None
        return None, None

    def _move(self, arm, tcp, goal, grip, next_phase, now):
        err = goal - tcp
        if np.linalg.norm(err) < TOL:
            self._go(next_phase, now)
        return _ee(arm, _vel(err), grip), None

    def summary(self):
        return {"running": self.running, "phase": self.phase, "index": self.index, "ticket": self.ticket}


def _vel(err):
    """World velocity toward a waypoint, as a stick deflection in -1..1."""
    from edge.teleop import MAX_SPEED
    return np.clip(GAIN * np.asarray(err) / MAX_SPEED, -1, 1)


def _stick(world):
    """World (x, y, z) -> stick (right, forward, up) for a view heading +x (yaw 0)."""
    return [round(float(-world[1]), 3), round(float(world[0]), 3), round(float(world[2]), 3)]


def _ee(arm, world_v, grip):
    return {"type": "ee", "arm": arm, "v": _stick(world_v), "grip": grip, "yaw": 0.0}


def _base(err):
    from edge.teleop import BASE_SPEED
    v = np.clip(GAIN * np.append(err, 0.0) / BASE_SPEED, -1, 1)
    return {"type": "base", "v": _stick(v)[:2] + [0.0], "yaw": 0.0}


# Per scene. "hard" targets are scripted: the sim has no perception, so these stand in for
# what a real robot would be unsure about. Every other ticket comes from a real failure.
PLANS = {
    # Come in from the aisle side, close, pull back toward the aisle and down off the stem.
    # Then carry it over the basket on the base and let go.
    "strawberry": Plan(base_dx=-0.40, pre=(0, -0.10, 0), grasp=(0, 0, 0), pull=(0, -0.12, -0.04), place=(0.16, 0.03, 0.50),
                       hard={2: ("occluded", "leaves hide part of the berry: where to grip?"),
                             4: ("ripeness", "is this berry ripe enough to pick?")}),
    # Come down from above, close around the stem, pull straight up.
    "weeding": Plan(base_dx=-0.15, pre=(0, 0, 0.08), grasp=(0, 0, 0), pull=(0, 0, 0.10),
                    hard={1: ("weed_or_crop", "weed or crop? it looks a lot like a seedling"),
                          4: ("unknown_plant", "unknown plant: not on the weed list")}),
}
TASKS = {"strawberry": "pick", "weeding": "weed"}
