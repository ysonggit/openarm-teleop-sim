"""Harvest task: pick each highlighted strawberry in order with either gripper.

With a basket on the base, a berry counts once it rests in the basket for DWELL
seconds (pick, carry, drop). Without one (fixed column, or weeds) it counts as below.

A berry comes off its stem when both fingers of one gripper squeeze it. It
counts once it is off the stem, still in a gripper (within HOLD of a TCP) and
at least PULL away from where it hung, for DWELL seconds. Each trial's
duration is the time from the previous success (or start). Later berries are
further down the aisle: drive the base to reach them (see edge/greenhouse.py).
"""

import numpy as np

PULL = 0.05  # m away from where it hung
HOLD = 0.05  # m from a gripper center: still held, not dropped
DWELL = 0.5  # s


class HarvestTask:
    def __init__(self, starts):
        self.starts = np.asarray(starts, float)  # where the berries hang
        self.active = False
        self.index = 0
        self.results = []  # seconds per picked berry; None for one skipped
        self.trial_start = None
        self.inside_since = None
        self.dist = self.status = None

    def start(self, now):
        self.active, self.index, self.results = True, 0, []
        self.trial_start, self.inside_since = now, None
        self.dist = self.status = None

    def tick(self, berries, attached, tcps, now, placed=None):
        """berries: positions; attached: still on the stem; tcps: gripper centers;
        placed: resting in the basket (None: the scene has no basket, pulling off is enough).
        Returns 'hit' or 'done' on progress, else None."""
        if not self.active:
            return None
        i = self.index
        p = berries[i]
        self.dist = float(min(np.linalg.norm(t - p) for t in tcps))
        held = not attached[i] and self.dist < HOLD
        in_place = placed is not None and not attached[i] and placed[i]
        self.status = ("on the stem" if attached[i] else "in the basket" if in_place
                       else "in the gripper" if held else "dropped")
        done = in_place if placed is not None else held and np.linalg.norm(p - self.starts[i]) >= PULL
        if not done:
            self.inside_since = None
            return None
        if self.inside_since is None:
            self.inside_since = now
        if now - self.inside_since < DWELL:
            return None
        return self._advance(now - self.trial_start, now)

    def skip(self, now):
        """Give up on the current target (an operator decided it can't be done): no time counts."""
        return self._advance(None, now) if self.active else None

    def _advance(self, seconds, now):
        self.results.append(seconds)
        self.index += 1
        self.trial_start, self.inside_since = now, None
        self.dist = self.status = None
        if self.index == len(self.starts):
            self.active = False
            return "done"
        return "hit"

    def summary(self, now):
        return {
            "active": self.active,
            "index": self.index,
            "total": len(self.starts),
            "elapsed": round(now - self.trial_start, 2) if self.active else None,
            "dist": None if self.dist is None else round(self.dist, 3),
            "status": self.status,
            "results": [None if r is None else round(r, 2) for r in self.results],  # None: skipped
        }
