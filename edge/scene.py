"""Scene export for the browser 3D view.

The browser draws the robot itself from body poses (like RViz from
/joint_states), so it can orbit freely without new video frames. Meshes are
sent once, decimated by vertex clustering; poses go out with every state.
"""

import mujoco
import numpy as np

VISIBLE_GROUPS = (0, 2)  # 0: floor, greenhouse, berries; 2: OpenArm visual meshes (3 is collision)
CELL = 0.003  # m; vertex-clustering grid. ~2 MB for the bimanual model.
TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def decimate(vert, face, cell=CELL):
    """Merge vertices that fall in the same grid cell; drop collapsed faces."""
    key = np.floor(vert / cell).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    v = np.zeros((cnt.size, 3))
    np.add.at(v, inv, vert)
    v /= cnt[:, None]
    f = inv[face]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return v, f[keep]


def export_scene(model, cell=CELL):
    """Returns (meta, blob). blob = float32 vertices, then uint32 faces, per mesh."""
    m = model
    geoms, meshes, verts, faces, mesh_index = [], [], [], [], {}
    v_off = f_off = 0
    for i in range(m.ngeom):
        kind = TYPES.get(int(m.geom_type[i]))
        if kind is None or m.geom_group[i] not in VISIBLE_GROUPS:
            continue
        rgba = m.mat_rgba[m.geom_matid[i]] if m.geom_matid[i] >= 0 else m.geom_rgba[i]
        g = {
            "id": i,
            "name": m.geom(i).name,
            "type": kind,
            "body": int(m.geom_bodyid[i]),
            "size": m.geom_size[i].tolist(),
            "pos": m.geom_pos[i].tolist(),
            "quat": m.geom_quat[i].tolist(),  # w, x, y, z
            "rgba": [float(x) for x in rgba],
        }
        if kind == "mesh":
            k = int(m.geom_dataid[i])
            if k not in mesh_index:
                va, vn = m.mesh_vertadr[k], m.mesh_vertnum[k]
                fa, fn = m.mesh_faceadr[k], m.mesh_facenum[k]
                v, f = decimate(m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn], cell)
                mesh_index[k] = len(meshes)
                meshes.append({"v": [v_off, len(v)], "f": [f_off, len(f)]})
                verts.append(v.astype(np.float32))
                faces.append(f.astype(np.uint32))
                v_off, f_off = v_off + len(v), f_off + len(f)
            g["mesh"] = mesh_index[k]
        geoms.append(g)
    v = np.concatenate(verts) if verts else np.zeros((0, 3), np.float32)
    f = np.concatenate(faces) if faces else np.zeros((0, 3), np.uint32)
    meta = {"nbody": m.nbody, "geoms": geoms, "meshes": meshes, "nvert": len(v), "nface": len(f)}
    return meta, v.tobytes() + f.tobytes()


def body_poses(data, ndigits=4):
    """Flat [x, y, z, qw, qx, qy, qz] per body, for the state message."""
    return np.round(np.hstack([data.xpos, data.xquat]).ravel(), ndigits).tolist()
