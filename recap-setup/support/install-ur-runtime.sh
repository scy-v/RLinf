#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"

uv pip install --python "$PYTHON" --no-deps \
  ur-rtde==1.6.3 pydhgripper==1.0.2 pyrealsense2==2.59.1.11053 \
  pyserial==3.5 crcmod==1.7

# OpenPI brings headless OpenCV; overlapping wheels overwrite the same cv2 files.
uv pip uninstall --python "$PYTHON" \
  opencv-python-headless opencv-python opencv-contrib-python-headless opencv-contrib-python
uv pip install --python "$PYTHON" --no-deps opencv-python==4.11.0.86

"$PYTHON" - <<'PY'
import importlib

import cv2

for name in (
    "rtde_control", "rtde_receive", "pyDHgripper", "pyrealsense2", "serial", "crcmod",
    "lerobot.cameras.realsense.camera_realsense",
):
    importlib.import_module(name)
gui = next(line.strip() for line in cv2.getBuildInformation().splitlines() if "GUI:" in line)
if gui.split(":", 1)[1].strip() in ("", "NONE"):
    raise RuntimeError(f"Expected OpenCV GUI support for UR visualization, got {gui}")
print(f"UR runtime imports passed; OpenCV {cv2.__version__}, {gui}")
PY
