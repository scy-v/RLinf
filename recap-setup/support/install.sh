#!/usr/bin/env bash
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(dirname "$SETUP_DIR")"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_DIR"
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$SETUP_DIR/uv-cache"
# The official installer patches transformers in place. Copies keep the two
# environments and the download cache independent during that patch.
export UV_LINK_MODE=copy
export LIBERO_CONFIG_PATH="$SETUP_DIR/support/libero-config"
export DOWNLOAD_DIR="$SETUP_DIR/outputs/assets"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
# wget recognizes lowercase proxy variables; curl and uv also accept uppercase.
if [[ -n "${HTTPS_PROXY:-}" && -z "${https_proxy:-}" ]]; then
  export https_proxy="$HTTPS_PROXY"
fi
if [[ -n "${HTTP_PROXY:-}" && -z "${http_proxy:-}" ]]; then
  export http_proxy="$HTTP_PROXY"
fi
if [[ -n "${NO_PROXY:-}" && -z "${no_proxy:-}" ]]; then
  export no_proxy="$NO_PROXY"
fi
mkdir -p "$SETUP_DIR/logs" "$LIBERO_CONFIG_PATH" "$DOWNLOAD_DIR"
# A zero-byte zip from an interrupted wget otherwise looks complete to the
# official asset downloader. Remove only that empty partial download.
PHYSX_ZIP="$DOWNLOAD_DIR/.sapien/physx/105.1-physx-5.3.1.patch0/linux-so.zip"
if [[ -f "$PHYSX_ZIP" && ! -s "$PHYSX_ZIP" ]]; then
  rm -- "$PHYSX_ZIP"
fi
python3 "$SETUP_DIR/support/install.py" "$@" \
  2>&1 | tee "$SETUP_DIR/logs/install-$(date +%Y%m%d-%H%M%S).log"
