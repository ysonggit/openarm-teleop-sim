"""Poly tunnel with soil beds: two raised beds along +x with the aisle between them; the
task is to pull the weeds that grow along the aisle-side edge of each bed.

The robot starts in the aisle facing +x, as in the greenhouse scene. Each weed is held in
the soil by a weld until both fingers of one gripper squeeze it (same rule as a berry on its
stem), then pulled up and out.

Reach sets the bed height: with the arms mounted for the tabletop (shoulders ~0.85 m up),
the gripper gets down to ~0.30 m, so the beds here are 0.25 m high and the weeds stand
~5 cm above the soil. Lower beds would need a lower arm mount or a different chassis.
"""

import mujoco
import numpy as np

BED_IN, BED_OUT = 0.27, 0.75  # m from the aisle center line; the base is 0.48 m wide
BED_TOP = 0.25  # m above the floor (not relative to the arm mount: the soil is on the ground)
BED_X = (-0.7, 2.3)
CROP_Y = 0.52  # crop row along the middle of each bed
WEED_Y = 0.33  # weeds along the aisle-side edge
WEED_GRIP = 0.05  # m above the soil: where the gripper closes
WEED_BOX = (0.008, 0.008, 0.02)  # collision proxy around the stem
WEED_MASS = 0.01  # kg
# Weeds to pull, in task order: (x, side) with side +1 = left bed, -1 = right bed.
WEEDS = [(0.15, 1), (0.25, -1), (0.75, 1), (0.95, -1), (1.40, 1), (1.60, -1)]  # base reach: x <= 1.65
FIXED_REACH_X = 0.3  # without the base, only weeds this close are reachable
HOOP_R = 1.6  # m; tunnel hoops (visual only)
HOOP_X = (-0.6, 0.6, 1.8)

SOIL = [0.36, 0.27, 0.19, 1]
STEEL = [0.7, 0.71, 0.72, 1]
CROP = [0.2, 0.5, 0.22, 1]
WEED = [0.62, 0.7, 0.25, 1]  # yellow-green, unlike the crop
TARGET_RGBA = WEED

BOX, CAPSULE, ELLIPSOID = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID


def target_starts(mount_z, mobile):
    """World positions of the weeds' grip points."""
    z = BED_TOP + WEED_GRIP
    return [(x, s * WEED_Y, z) for x, s in WEEDS if mobile or x <= FIXED_REACH_X]


def build(spec, mount_z, mobile):
    """Add beds, crops, weeds and tunnel hoops to the world. Returns the weed names."""
    w = spec.worldbody
    visual = dict(contype=0, conaffinity=0)
    x0, x1 = BED_X
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    for s in (1, -1):
        cy, hy = s * (BED_IN + BED_OUT) / 2, (BED_OUT - BED_IN) / 2
        w.add_geom(type=BOX, size=[hx, hy, BED_TOP / 2], pos=[cx, cy, BED_TOP / 2], rgba=SOIL)
        for x in np.arange(x0 + 0.15, x1, 0.3):  # crop seedlings: a low rosette, visual only
            for k in range(6):
                a = 2 * np.pi * k / 6 + x
                c = [x + 0.05 * np.cos(a), s * CROP_Y + 0.05 * np.sin(a), BED_TOP + 0.02]
                w.add_geom(type=ELLIPSOID, size=[0.06, 0.03, 0.01], pos=c,
                           quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=CROP, **visual)
    for x in HOOP_X:  # half-circle hoops over the aisle and both beds
        th = np.linspace(0, np.pi, 13)
        pts = [(x, HOOP_R * np.cos(t), HOOP_R * np.sin(t)) for t in th]
        for p, q in zip(pts, pts[1:]):
            w.add_geom(type=CAPSULE, size=[0.012, 0, 0], fromto=[*p, *q], rgba=STEEL, **visual)

    names = []
    for i, (x, y, z) in enumerate(target_starts(mount_z, mobile)):
        body = w.add_body(name=f"weed{i}", pos=[x, y, z])
        body.add_freejoint()
        body.add_geom(name=f"weed{i}", type=ELLIPSOID, size=[0.006, 0.006, WEED_GRIP / 2 + 0.01],
                      pos=[0, 0, -WEED_GRIP / 2 + 0.01], mass=WEED_MASS, rgba=WEED, **visual)
        for k in range(4):  # leaves at the top, visual only
            a = np.pi * k / 2 + 0.3 * i
            body.add_geom(type=ELLIPSOID, size=[0.03, 0.012, 0.004], pos=[0.025 * np.cos(a), 0.025 * np.sin(a), 0.02],
                          quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], mass=0.001, rgba=WEED, **visual)
        body.add_geom(name=f"weed{i}_collision", type=BOX, size=WEED_BOX, mass=0, group=3, condim=4,
                      friction=[1, 0.01, 0.001], rgba=WEED)
        root = spec.add_equality(name=f"stem{i}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name1="world", name2=f"weed{i}")
        data = root.data.copy()
        data[:3] = 0  # anchor at the weed
        root.data = data
        names.append(f"weed{i}")
    return names
