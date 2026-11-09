#!/usr/bin/env bash
set -euo pipefail
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO_PATH"
python "$REPO_PATH/evaluations/realworld/eval_ur5e_vla_precision.py" "$@"
