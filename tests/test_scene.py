from pathlib import Path

import numpy as np
import pytest

from edge.scene import body_poses, decimate, export_scene

MODEL = Path(__file__).resolve().parent.parent / "vendor/openarm_mujoco/v1/openarm_bimanual.xml"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_model.sh first")


def test_decimate_merges_close_vertices_and_drops_collapsed_faces():
    v = np.array([[0, 0, 0], [0.001, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    nv, nf = decimate(v, f, cell=0.01)
    assert len(nv) == 3 and len(nf) == 1


@needs_model
def test_scene_has_visual_meshes_berries_and_fits_in_a_few_mb():
    from edge.mujoco_adapter import MujocoAdapter

    a = MujocoAdapter(MODEL, greenhouse_rows=True)
    assert len(a.berry_ids) == 2  # a fixed robot only gets the berries it can reach
    meta, blob = export_scene(a.model)
    names = [g["name"] for g in meta["geoms"]]
    assert sum(n.endswith("_visual_0") for n in names) > 10
    assert not any("collision" in n for n in names)
    assert all(f"berry{i}" in names for i in range(len(a.berry_ids)))
    assert "ellipsoid" in {g["type"] for g in meta["geoms"]}
    assert len(blob) == meta["nvert"] * 12 + meta["nface"] * 12 < 4e6
    assert len(body_poses(a.data)) == 7 * a.model.nbody
