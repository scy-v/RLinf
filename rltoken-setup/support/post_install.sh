#!/usr/bin/env bash
# Same GUI-wheel repair used by RLinf's use_opencv_gui_wheel().
# OpenPI installs a headless wheel after the hardware installer repaired it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(dirname "$SCRIPT_DIR")"
source /home/suchenyu/RLinf-rltoken/.venv/bin/activate
export UV_CACHE_DIR="$SETUP_DIR/uv-cache"
export UV_LINK_MODE=hardlink
uv pip uninstall opencv-python opencv-python-headless
uv pip install --no-deps opencv-python==4.11.0.86
