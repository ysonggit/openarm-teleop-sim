#!/bin/sh
# Fetch the OpenArm MuJoCo models at the commit this sim was built against.
set -e
cd "$(dirname "$0")/.."
[ -d vendor/openarm_mujoco ] || git clone https://github.com/enactic/openarm_mujoco.git vendor/openarm_mujoco
git -C vendor/openarm_mujoco checkout -q 06b4c8ad6d547c3faaa3bf780629ee3e1ac2b0d0

# Stanford TidyBot mobile base (MIT), from MuJoCo Menagerie. Base only, no Kinova arm.
MENAGERIE=https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/eea4c7431b63da8278b0ff31638e0b0e941cec0e/stanford_tidybot
mkdir -p vendor/tidybot_base/assets/base
for f in base.xml LICENSE README.md assets/base/bumper.stl assets/base/bottom_plate.stl \
         assets/base/body.stl assets/base/top_plate.stl assets/base/arm_plate.stl; do
  [ -f "vendor/tidybot_base/$f" ] || curl -sfL "$MENAGERIE/$f" -o "vendor/tidybot_base/$f"
done
