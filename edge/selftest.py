"""Headless check for a server host: builds the full scene, steps it and renders
both cameras the way the server does (on a worker thread). Exits non-zero on failure.

    MUJOCO_GL=osmesa uv run python -m edge.selftest
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge import rgbd
from edge.mujoco_adapter import MujocoAdapter

ROOT = Path(__file__).resolve().parent.parent


def main():
    t = time.perf_counter()
    a = MujocoAdapter(ROOT / "vendor/openarm_mujoco/v1/openarm_bimanual.xml",
                      base_path=ROOT / "vendor/tidybot_base/base.xml", greenhouse_rows=True)
    a.write(a.home)
    t_step = time.perf_counter()
    a.step(1.0)
    step_s = time.perf_counter() - t_step
    print(f"model: {a.model.nbody} bodies, {len(a.berry_ids)} berries, built in {time.perf_counter() - t:.1f} s")
    print(f"physics: 1 s of sim in {step_s:.3f} s CPU ({100 * step_s:.0f}% of a core at real time)")

    def render():
        a.render_jpeg()
        rgbd.pack(*a.render_rgbd())  # warm-up: first frames compile shaders
        t = time.perf_counter()
        jpeg = a.render_jpeg()
        t_cam = time.perf_counter() - t
        t = time.perf_counter()
        frame = rgbd.pack(*a.render_rgbd())
        t_rgbd = time.perf_counter() - t
        a.close_renderers()
        return len(jpeg), t_cam, len(frame), t_rgbd

    n_jpeg, t_cam, n_rgbd, t_rgbd = ThreadPoolExecutor(1).submit(render).result()
    assert n_jpeg > 1000 and n_rgbd > 1000, "empty frames"
    print(f"render ({os.environ.get('MUJOCO_GL', 'default')}): robot camera {1000 * t_cam:.0f} ms, "
          f"RGB-D {1000 * t_rgbd:.0f} ms per frame")
    print("selftest ok")


if __name__ == "__main__":
    main()
