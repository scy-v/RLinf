# Usage: source ~/RLinf-recap/recap-setup/activate.sh
_RECAP_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_RECAP_REPO_DIR/.venv/bin/activate"
export PYTHONPATH="$_RECAP_REPO_DIR"
export REPO_PATH="$_RECAP_REPO_DIR"
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$_RECAP_REPO_DIR/recap-setup/uv-cache"
export UV_LINK_MODE=copy
export HF_DATASETS_CACHE="$_RECAP_REPO_DIR/recap-setup/outputs/hf-datasets-cache"
export LIBERO_CONFIG_PATH="$_RECAP_REPO_DIR/recap-setup/support/libero-config"
export MS_ASSET_DIR="$_RECAP_REPO_DIR/recap-setup/outputs/assets/.maniskill"
export PHYSX_DIR="$_RECAP_REPO_DIR/recap-setup/outputs/assets/.sapien/physx/105.1-physx-5.3.1.patch0"
export OPENPI_DATA_HOME="$_RECAP_REPO_DIR/recap-setup/outputs/assets/.cache/openpi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
cd "$_RECAP_REPO_DIR"
unset _RECAP_REPO_DIR
