# UR5e RLToken Training and Inference Commands

## View TensorBoard

Run in a separate terminal on the server to view curves during training:

```bash
cd ~/RLinf-rltoken
.venv/bin/tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

Forward port `6006` through the VS Code **Ports** panel, open http://localhost:6006 in your browser, and select training runs under **Runs** to view or compare them. Keep this terminal running.

## 1. Activate the Environment

```bash
source ~/RLinf-rltoken/rltoken-setup/activate.sh
```

## 2. Compute Normalization Statistics

Run before the first training session or after changing data or input dimensions. Skip this step if matching statistics already exist. Use the same `obs_indices` in the training configuration:

```bash
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state \
  --repo-id "$HOME/.cache/huggingface/lerobot/scylearning/insert_black_rack_into_blue_slot_20260916_v01" \
  --obs-indices 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --output-dir "$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state19"
```

This writes `norm_stats.json` and `norm_metadata.json`. If you change the output directory, update `actor.model.openpi_data.norm_stats_path` in the training configuration.

## 3. Start Stage1 SFT

Start with the dataset, normalization statistics, and weights already prepared on this machine:

```bash
bash examples/sft/run_vla_sft.sh ur_rlt_stage1_sft_openpi_pi05
```

Edit [ur_rlt_stage1_sft_openpi_pi05.yaml](../../examples/sft/config/ur_rlt_stage1_sft_openpi_pi05.yaml) to change the settings.

| Setting | Current value |
|---|---|
| GPU / batch | GPUs 0–3, micro 2 per GPU, global 32 |
| Training mode | Full parameters, `no_shard`; upstream SFT forward disables gradient checkpointing |
| `runner.max_steps` | 30000 optimizer updates |
| `runner.save_interval` | Save every 5000 optimizer updates |
| `actor.model.openpi_data.obs_indices` | One-based dimensions 1–19 |

Checkpoints are saved under:

```text
/data/suchenyu/RLinf-rltoken/logs/<timestamp>-ur_rlt_stage1_sft_openpi_pi05/ur_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<step>/actor/
```

## 4. Run UR5e Inference Locally

Run on the local machine connected to the arm, gripper and cameras. Prepare a separate environment once:

```bash
cd ~/RLinf-rltoken
bash requirements/install.sh embodied --model openpi --env ur --venv .ur-eval-venv
```

Check the workspace bounds in the [environment config](../../examples/embodiment/config/env/realworld_ur5e_sft_env.yaml). Check the hardware, local model and normalization paths in the [inference config](../../evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml), then run:

```bash
cd ~/RLinf-rltoken
source .ur-eval-venv/bin/activate
bash evaluations/run_eval.sh realworld_ur5e_eval_pi05_sft
```

Evaluation logs, statistics and videos use the real `results/` directory inside the repository. RLinf evaluation creates timestamped directories; VLA-Precision inference uses `results/vla-precision-ur/`. Ray diagnostics default to `results/ray/`. Training checkpoints remain under `logs/`.

The inference YAML sets `runner.ckpt_path` to the step-30000 checkpoint from the `20261018-22:40:04` training run. RTC defaults to off; set `runner.rtc.enabled: true` in that YAML to enable it. Update the checkpoint, base-model and normalization paths in YAML if local deployment uses different paths.

UR resets to a TCP pose with independent uniform ±3 cm offsets along XYZ and fixed orientation, without periodic joint resets.

**Startup moves the robot to its initial pose; after the first reset and every later reset, press Enter once to start policy execution.** During execution, press Enter twice within 0.5 seconds on the local physical keyboard to end the episode, reset and wait for another single Enter press. This does not count as task success. The environment YAML enables the start wait with `keyboard_wrapper_kwargs.wait_for_start: True`. Evaluation has a total budget of 30000 environment steps; use `Ctrl+C` to end the process.

See [technical notes](TRAINING_NOTES.en.md) for weight conversion, data handling, Stage2 adaptation, and verification records. [中文](../训练命令.md).
