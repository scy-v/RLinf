#!/usr/bin/env bash
# Run independent RECAP CFG experiments with the supported default conditions.
set -euo pipefail

cd /home/suchenyu/RLinf-recap
source recap-setup/activate.sh
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

batch_id="$(date +%Y%m%d-%H%M%S)"

for mode in false true; do
    # Model-class defaults for false; official example YAML defaults for true.
    dropout=0.3
    if [[ "$mode" == true ]]; then dropout=0.1; fi
    experiment="ur_recap_cfg_${mode}"
    output="$PWD/logs/cfg_rl/${batch_id}-${experiment}"
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite existing run: $output" >&2
        exit 1
    fi
    mkdir -p "$output"
    echo "Starting $mode with unconditional_prob=$dropout"

    python examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py \
        --config-path "$PWD/examples/offline_rl/config" \
        --config-name ur_recap_cfg \
        actor.model.openpi.positive_only_conditional="$mode" \
        actor.model.openpi.unconditional_prob="$dropout" \
        runner.max_steps=20000 \
        actor.optim.total_training_steps=20000 \
        actor.optim.lr_warmup_steps=5000 \
        runner.save_interval=2500 \
        ++runner.resume_dir=null \
        runner.logger.log_path="$output" \
        runner.logger.experiment_name="$experiment" \
        2>&1 | tee "$output/run.log"

    checkpoint="$output/$experiment/checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt"
    if [[ ! -s "$checkpoint" ]]; then
        echo "Missing checkpoint; stopping: $checkpoint" >&2
        exit 1
    fi
    echo "Completed: $checkpoint"
done

echo 'Both Step 4 runs completed.'
