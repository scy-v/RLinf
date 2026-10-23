#!/usr/bin/env bash
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(dirname "$SETUP_DIR")"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SETUP_DIR/uv-cache"
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
export DOWNLOAD_DIR="$SETUP_DIR/outputs/assets"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$SETUP_DIR/logs" "$SETUP_DIR/support/wheels" "$DOWNLOAD_DIR"
exec > >(tee "$SETUP_DIR/logs/install-realworld-$(date +%Y%m%d-%H%M%S).log") 2>&1
source "$REPO_DIR/.venv/bin/activate"
uv pip install rlinf-openpi==0.1.1
uv pip install -r requirements/embodied/models/openpi.txt
python - <<'PY'
import pathlib
import shutil
import sys
site = pathlib.Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
source = site / 'openpi/models_pytorch/transformers_replace'
assert source.is_dir(), source
shutil.copytree(source, site / 'transformers', dirs_exist_ok=True)
PY
uv pip install 'tokenizers>=0.21,<0.22'
python - <<'PY'
import torch
assert torch.__version__.startswith('2.11.'), torch.__version__
assert torch.version.cuda.startswith('13.'), torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY
WHEEL_NAME='flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
WHEEL="$SETUP_DIR/support/wheels/$WHEEL_NAME"
if [[ ! -s "$WHEEL" ]]; then
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 900 \
    "https://github.com/RLinf/flash-attention/releases/download/v2.8.3/$WHEEL_NAME" \
    -o "$WHEEL.part"
  mv -- "$WHEEL.part" "$WHEEL"
fi
uv pip install --no-deps "$WHEEL"
bash requirements/embodied/download_assets.sh --assets openpi
# Camera previews need the GUI wheel; OpenPI also brings a headless wheel.
uv pip uninstall opencv-python-headless opencv-python
uv pip install --no-deps opencv-python==4.11.0.86
