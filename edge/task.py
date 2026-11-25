"""Reach task: touch each target in order with either gripper center.

A target counts once a TCP stays within RADIUS for DWELL seconds. Each
trial's duration is the time from the previous success (or start).
"""

import numpy as np

RADIUS = 0.02  # m
DWELL = 0.5  # s

# Placed around the home TCPs (x 0.41, y +-0.15, z 0.48); alternates left / right.
TARGETS = [
    (0.45, 0.20, 0.55), (0.45, -0.20, 0.55),
    (0.35, 0.25, 0.40), (0.35, -0.25, 0.40),
    (0.50, 0.10, 0.45), (0.50, -0.10, 0.45),
    (0.40, 0.30, 0.50), (0.40, -0.30, 0.50),
    (0.30, 0.15, 0.60), (0.30, -0.15, 0.60),
]


class ReachTask:
    def __init__(self, targets=TARGETS):
        self.targets = np.asarray(targets, float)
        self.active = False
        self.index = 0
        self.results = []  # seconds per completed target
        self.trial_start = None
        self.inside_since = None
        self.dist = None

    def start(self, now):
        self.active, self.index, self.results = True, 0, []
        self.trial_start, self.inside_since = now, None

    def tick(self, tcps, now):
        """tcps: list of TCP positions. Returns 'hit' or 'done' on progress, else None."""
        if not self.active:
            return None
        self.dist = float(min(np.linalg.norm(p - self.targets[self.index]) for p in tcps))
        if self.dist > RADIUS:
            self.inside_since = None
            return None
        if self.inside_since is None:
            self.inside_since = now
        if now - self.inside_since < DWELL:
            return None
        self.results.append(now - self.trial_start)
        self.index += 1
        self.trial_start, self.inside_since = now, None
        if self.index == len(self.targets):
            self.active = False
            return "done"
        return "hit"

    def summary(self, now):
        return {
            "active": self.active,
            "index": self.index,
            "total": len(self.targets),
            "elapsed": round(now - self.trial_start, 2) if self.active else None,
            "dist": None if self.dist is None else round(self.dist, 3),
            "results": [round(r, 2) for r in self.results],
        }
