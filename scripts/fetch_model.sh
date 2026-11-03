#!/bin/sh
# Fetch the OpenArm MuJoCo models at the commit this sim was built against.
set -e
cd "$(dirname "$0")/.."
[ -d vendor/openarm_mujoco ] || git clone https://github.com/enactic/openarm_mujoco.git vendor/openarm_mujoco
git -C vendor/openarm_mujoco checkout -q 06b4c8ad6d547c3faaa3bf780629ee3e1ac2b0d0
