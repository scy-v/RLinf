#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
export UV_CACHE_DIR="$REPO_DIR/recap-setup/uv-cache"
export UV_LINK_MODE=copy
uv pip install --python "$PYTHON" --no-deps \
  'git+https://github.com/huggingface/lerobot.git@d6ea3bbce0fc8c75138f02639ac4ceaf12a67829' \
  Send2Trash==1.8.3 pynput==1.8.1
bash "$REPO_DIR/recap-setup/support/install-ur-runtime.sh"
bash "$REPO_DIR/toolkits/ur_data_collection/run_rollout.sh" --check
