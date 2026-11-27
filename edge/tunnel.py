"""Poly tunnel with soil beds: two raised beds along +x with the aisle (furrow) between them;
the task is to pull the weeds that grow along the aisle-side edge of each bed.

The robot starts in the aisle facing +x, as in the greenhouse scene. Each weed is held in
the soil by a weld until both fingers of one gripper squeeze it (same rule as a berry on its
stem), then pulled up and out.

The beds are raised ridges with a trapezoid cross-section and rough soil: one visual
terrain mesh (bed profile + clods), plus a smooth convex prism per bed for collision.

Reach sets the bed shape: with the arms mounted for the tabletop (shoulders ~0.85 m up),
the gripper gets down to ~0.30 m and only ~0.38 m out to the side, so the beds here are
0.25 m high with steep sides (~75 deg) right next to the base. Lower, gentler beds would
need a lower arm mount or a different chassis.
"""

import mujoco
import numpy as np

BED_TOP = 0.25  # m above the floor
BED_IN = (0.25, 0.32)  # aisle-side edge: (foot, top) in m from the aisle center line; the base is 0.48 m wide
BED_OUT = (0.85, 0.78)  # far edge: (foot, top)
BED_X = (-0.7, 9.3)  # a 10 m bed
END_RUN = 0.07  # m; the bed ends slope down too
CROP_Y = 0.55  # crop row along the middle of each bed
WEED_Y = 0.35  # weeds along the aisle-side edge of the top
WEED_GRIP = 0.05  # m above the soil: where the gripper closes
WEED_BOX = (0.008, 0.008, 0.02)  # collision proxy around the stem
WEED_MASS = 0.01  # kg
# Weeds to pull, in task order: (x, side) with side +1 = left bed, -1 = right bed.
WEEDS = [(0.15, 1), (0.25, -1), (1.40, 1), (2.60, -1), (3.90, 1), (5.30, -1), (6.80, 1), (8.40, -1)]
FIXED_REACH_X = 0.3  # without the base, only weeds this close are reachable
HOOP_R = 1.6  # m; tunnel hoops (visual only)
HOOP_X = np.arange(-0.6, 10.0, 1.2)
TERRAIN_X = (-2.0, 11.0)  # the visual soil covers the floor under the tunnel
TERRAIN_STEP = 0.05  # m between terrain vertices
CLOD = 0.008  # m; roughness of the soil surface
SEED = 7  # the soil looks the same every run

SOIL = [0.36, 0.27, 0.19, 1]
SOIL_DARK = [0.25, 0.18, 0.12]
STEEL = [0.7, 0.71, 0.72, 1]
CROP = [0.2, 0.5, 0.22, 1]
WEED = [0.62, 0.7, 0.25, 1]  # yellow-green, unlike the crop
TARGET_RGBA = WEED
GROUND = "soil"  # tells the 3D view: no floor grid, soil texture on geoms named soil*

BOX, CAPSULE, ELLIPSOID, MESH = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                 mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_MESH)


def target_starts(mount_z, mobile):
    """World positions of the weeds' grip points."""
    z = BED_TOP + WEED_GRIP
    return [(x, s * WEED_Y, z) for x, s in WEEDS if mobile or x <= FIXED_REACH_X]


def bed_height(x, y):
    """Smooth bed profile (no clods): trapezoid across, sloped at both ends."""
    a = np.abs(y)
    (f_in, t_in), (f_out, t_out) = BED_IN, BED_OUT
    across = np.clip(np.minimum((a - f_in) / (t_in - f_in), (f_out - a) / (f_out - t_out)), 0, 1)
    along = np.clip(np.minimum((x - BED_X[0]) / END_RUN, (BED_X[1] - x) / END_RUN) + 1, 0, 1)
    return BED_TOP * across * along


def terrain():
    """Vertices and faces of the soil surface: bed profile plus clods (deterministic)."""
    rng = np.random.default_rng(SEED)
    xs = np.arange(TERRAIN_X[0], TERRAIN_X[1] + 1e-9, TERRAIN_STEP)
    edges = [s * e for e in (*BED_IN, *BED_OUT) for s in (1, -1)]
    ys = np.unique(np.round(np.concatenate([np.arange(-HOOP_R, HOOP_R + 1e-9, 0.04), edges]), 4))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coarse = rng.normal(0, 1, (len(xs) // 4 + 2, len(ys) // 4 + 2))  # lumps ~20 cm across
    lumps = coarse[np.arange(len(xs)) // 4][:, np.arange(len(ys)) // 4]
    Z = bed_height(X, Y) + CLOD * (0.6 * rng.normal(0, 1, X.shape) + 0.4 * lumps) + 0.004
    Z = np.maximum(Z, 0.001)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    nx, ny = X.shape
    i = np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]
    i = i.ravel()
    faces = np.concatenate([np.column_stack([i, i + ny, i + 1]), np.column_stack([i + 1, i + ny, i + ny + 1])])
    return verts, faces


def bed_prism(side):
    """Convex collision shape of one bed (no clods)."""
    (f_in, t_in), (f_out, t_out) = BED_IN, BED_OUT
    x0, x1 = BED_X
    ring = [(f_in, 0), (t_in, BED_TOP), (t_out, BED_TOP), (f_out, 0)]
    return [(x, side * y, z) for x in (x0, x1) for y, z in ring]


def build(spec, mount_z, mobile):
    """Add soil beds, crops, weeds and tunnel hoops to the world. Returns the weed names."""
    w = spec.worldbody
    visual = dict(contype=0, conaffinity=0)
    tex = spec.add_texture(name="soil", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                           mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.35, rgb1=SOIL[:3], rgb2=SOIL[:3],
                           markrgb=SOIL_DARK, width=256, height=256)
    mat = spec.add_material(name="soil", texrepeat=[1, 1])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    v, f = terrain()
    mesh = spec.add_mesh(name="soil_terrain", uservert=v.ravel().tolist(), userface=f.ravel().tolist(),
                         usertexcoord=(v[:, :2] / 0.5).ravel().tolist())  # texture repeats every 0.5 m
    w.add_geom(name="soil_terrain", type=MESH, meshname=mesh.name, material="soil", rgba=[1, 1, 1, 1], mass=0, **visual)
    for s in (1, -1):
        prism = spec.add_mesh(name=f"bed{s:+d}", uservert=np.ravel(bed_prism(s)).tolist())
        w.add_geom(name=f"bed{s:+d}", type=MESH, meshname=prism.name, group=3, mass=0, rgba=SOIL)
        for x in np.arange(BED_X[0] + 0.15, BED_X[1], 0.3):  # crop seedlings: a low rosette, visual only
            for k in range(5):
                a = 2 * np.pi * k / 5 + x
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
