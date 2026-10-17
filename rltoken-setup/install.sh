#!/usr/bin/env bash
# RLinf OpenPI + local UR hardware dependency target; no weights or datasets.
set -euo pipefail
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=/home/suchenyu/RLinf
mkdir -p "$SETUP_DIR/logs"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone --depth 1 --branch rlt https://github.com/scy-v/RLinf.git "$REPO_DIR"
fi
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export UV_CACHE_DIR="$SETUP_DIR/uv-cache"
export UV_LINK_MODE=hardlink
export MAX_JOBS=4
bash requirements/install.sh embodied --model openpi --env ur \
    --venv "$REPO_DIR/.venv" --no-root --install-rlinf "$@" \
    2>&1 | tee "$SETUP_DIR/logs/install-$(date +%Y%m%d-%H%M%S).log"
bash "$SETUP_DIR/post_install.sh" 2>&1 | tee "$SETUP_DIR/logs/post-install.log"
