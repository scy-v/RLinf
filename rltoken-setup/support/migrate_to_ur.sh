#!/usr/bin/env bash
# Add UR dependencies to the existing OpenPI venv without resyncing the model stack.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$SETUP_DIR")"
source "$SETUP_DIR/activate.sh"
mkdir -p "$SETUP_DIR/logs/ur-migration"
uv pip install --python "$REPO_DIR/.venv/bin/python" \
    -r "$REPO_DIR/requirements/embodied/envs/ur.txt"
python -c 'import rtde_control, rtde_receive, rtde_io, serial, crcmod; from pyDHgripper import PGE'
# Do not remove shared serial/camera/numerical dependencies or system ROS installs.
uv pip uninstall --python "$REPO_DIR/.venv/bin/python" \
    franky-control RLinf-dexterous-hands rospkg empy
uv pip freeze --python "$REPO_DIR/.venv/bin/python" \
    > "$SETUP_DIR/logs/ur-migration/after.freeze.txt"
