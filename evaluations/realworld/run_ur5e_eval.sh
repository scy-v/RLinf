#!/usr/bin/env bash
set -euo pipefail
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$REPO_PATH/results"
export RAY_TMPDIR="${RAY_TMPDIR:-$REPO_PATH/results}"
UR_EVAL_LOG_PATH="$REPO_PATH/results/$(date +%Y%m%d-%H%M%S)-ur5e-eval"
python "$REPO_PATH/evaluations/realworld/eval_ur5e.py" \
    "runner.logger.log_path=$UR_EVAL_LOG_PATH" "$@"
