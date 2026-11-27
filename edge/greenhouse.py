"""Greenhouse tabletop strawberries: two raised gutter rows along +x with the aisle between them.

The robot starts in the aisle facing +x. Ripe berries hang over the aisle-side edge of each
gutter, so the left arm picks the left row and the right arm the right one; berries further
down the aisle need the base. A berry hangs on its stem (a weld to the world) until both
fingers of one gripper squeeze it; then it comes free (see MujocoAdapter.detach_grasped).

Heights are set relative to the arm mount, for this robot's reach (shoulders ~0.85 m on the
base): the gutter bottom is at 0.80 m. Real tabletop systems are often built higher for
standing pickers; reaching those would need a taller column.
"""

import mujoco
import numpy as np

GUTTER_IN, GUTTER_OUT = 0.34, 0.54  # m from the aisle center line; the base is 0.48 m wide
GUTTER_DEPTH = 0.10
GUTTER_BOTTOM = 0.645  # m above the arm mount: 0.80 m on the mobile base
ROW_X = (-0.7, 2.3)  # row ends; the base may drive x -1.5 .. 1.5
LEG_Y = 0.50
BERRY_HALF = (0.017, 0.017, 0.022)  # m; a large strawberry, ellipsoid
BERRY_MASS = 0.02  # kg
BERRY_BOX = (0.015, 0.015, 0.02)  # m; collision proxy inside the ellipsoid
BERRY_Y = 0.32  # just over the gutter's aisle-side edge
HANG = 0.10  # berry center below the gutter bottom: fingers clear the gutter
# Ripe berries to pick, in task order: (x, side) with side +1 = left row, -1 = right row.
BERRIES = [(0.40, 1), (0.45, -1), (0.95, 1), (1.05, -1), (1.60, 1), (1.85, -1)]
FIXED_REACH_X = 0.6  # without the base, only berries this close are reachable

WHITE = [0.93, 0.93, 0.9, 1]
PEAT = [0.28, 0.2, 0.14, 1]
STEEL = [0.45, 0.46, 0.48, 1]
LEAF = [0.2, 0.45, 0.18, 1]
STEM = [0.35, 0.55, 0.25, 1]
BERRY = [0.62, 0.08, 0.1, 1]
TARGET_RGBA = BERRY

BOX, CAPSULE, CYLINDER, ELLIPSOID = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID)


def target_starts(mount_z, mobile):
    """World positions where the berries hang."""
    z = GUTTER_BOTTOM + mount_z - HANG
    return [(x, s * BERRY_Y, z) for x, s in BERRIES if mobile or x <= FIXED_REACH_X]


def build(spec, mount_z, mobile):
    """Add gutters, legs, plants and hanging berries to the world. Returns the berry names."""
    w = spec.worldbody
    bottom = GUTTER_BOTTOM + mount_z
    top = bottom + GUTTER_DEPTH
    x0, x1 = ROW_X
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    visual = dict(contype=0, conaffinity=0)
    for s in (1, -1):
        cy, hy = s * (GUTTER_IN + GUTTER_OUT) / 2, (GUTTER_OUT - GUTTER_IN) / 2
        w.add_geom(type=BOX, size=[hx, hy, GUTTER_DEPTH / 2], pos=[cx, cy, bottom + GUTTER_DEPTH / 2], rgba=WHITE)
        w.add_geom(type=BOX, size=[hx, hy - 0.01, 0.004], pos=[cx, cy, top + 0.004], rgba=PEAT, **visual)
        for x in np.arange(x0 + 0.1, x1, 0.8):  # legs and a cross bar under the gutter
            w.add_geom(type=BOX, size=[0.015, 0.015, bottom / 2], pos=[x, s * LEG_Y, bottom / 2], rgba=STEEL)
            w.add_geom(type=BOX, size=[0.015, hy, 0.015], pos=[x, cy, bottom - 0.015], rgba=STEEL)
        for x in np.arange(x0 + 0.15, x1, 0.3):  # plants: a rosette of leaves, visual only
            for k in range(5):
                a = 2 * np.pi * k / 5 + 0.4 * s + x
                c = [x + 0.06 * np.cos(a), cy + 0.06 * np.sin(a), top + 0.03 + 0.01 * (k % 2)]
                w.add_geom(type=ELLIPSOID, size=[0.07, 0.035, 0.008], pos=c,
                           quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=LEAF, **visual)

    names = []
    for i, (x, y, z) in enumerate(target_starts(mount_z, mobile)):
        s = np.sign(y)
        w.add_geom(type=CAPSULE, size=[0.0025, 0, 0], fromto=[x, s * (GUTTER_IN + 0.01), top, x, y, z + BERRY_HALF[2]],
                   rgba=STEM, **visual)
        body = w.add_body(name=f"berry{i}", pos=[x, y, z])
        body.add_freejoint()
        body.add_geom(name=f"berry{i}", type=ELLIPSOID, size=BERRY_HALF, mass=BERRY_MASS, rgba=BERRY, **visual)
        # Collision proxy with flat sides: the fingers' squeeze pops a round shape out of the grip.
        body.add_geom(name=f"berry{i}_collision", type=BOX, size=BERRY_BOX, mass=0, group=3, condim=4,
                      friction=[1, 0.01, 0.001], rgba=BERRY)
        body.add_geom(type=CYLINDER, size=[0.013, 0.003, 0], pos=[0, 0, BERRY_HALF[2]], mass=0.001,
                      rgba=LEAF, **visual)
        # The stem: a weld from the world, anchored at the berry so it hangs still.
        stem = spec.add_equality(name=f"stem{i}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name1="world", name2=f"berry{i}")
        data = stem.data.copy()
        data[:3] = 0  # anchor at the berry (MjSpec's default anchor is 0 1 0)
        stem.data = data
        names.append(f"berry{i}")
    return names
