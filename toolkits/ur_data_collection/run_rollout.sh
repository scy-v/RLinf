#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_DIR/recap-setup/activate.sh"
python -m toolkits.ur_data_collection.run_rollout "$@"
