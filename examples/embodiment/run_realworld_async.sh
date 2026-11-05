#!/usr/bin/env bash
set -eo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
# Preserve every structured timing row when Ray forwards worker logs.
export RAY_DEDUP_LOGS_ALLOW_REGEX="${RAY_DEDUP_LOGS_ALLOW_REGEX:+${RAY_DEDUP_LOGS_ALLOW_REGEX}|}TIMING_RECORD"

CONFIG_NAME="${1:-realworld_peginsertion_rlpd_cnn_async}"
if [ "$#" -gt 0 ]; then
    shift
fi

if [ "$CONFIG_NAME" = "ur_rlt_stage2_ac_mlp" ]; then
    RUN_LOG_ROOT="${REPO_PATH}/logs/online_rltoken"
    LOG_OPTIONS=(--compact --deduplicate-timing)
else
    RUN_LOG_ROOT="${REPO_PATH}/logs"
    LOG_OPTIONS=(--compact)
fi
LOG_DIR="${RLINF_LOG_DIR:-${RUN_LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}}"
mkdir -p "$LOG_DIR"
CMD=(python -u "${EMBODIED_PATH}/train_async.py"
    --config-path "${EMBODIED_PATH}/config" --config-name "$CONFIG_NAME"
    "runner.logger.log_path=$LOG_DIR" "$@")
echo "Using Python at $(command -v python)"
printf '%q ' "${CMD[@]}" > "$LOG_DIR/run_embodiment.log"
printf '\n' >> "$LOG_DIR/run_embodiment.log"
"${CMD[@]}" 2>&1 | python -u -m rlinf.utils.progress "${LOG_OPTIONS[@]}" --log-file "$LOG_DIR/run_embodiment.log"
