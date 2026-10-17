# UR5e RLToken Training Preparation

Select the raw state dimensions with `obs_indices`, then use the same selection for normalization, SFT, and Stage2. The default is the first 19 dimensions: TCP pose (6), velocity (6), force/torque (6), and `gripper_raw_position` (1).

## Data and Normalization

```bash
source ~/RLinf/rltoken-setup/activate.sh
DATA="$HOME/.cache/huggingface/lerobot/scylearning/insert_black_rack_into_blue_slot_20260916_v01"
OUT="$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state19"
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state --repo-id "$DATA" --output-dir "$OUT"
```

The statistics script chooses its reader by storage format: v3 defaults to numeric parquet; v2 defaults to the existing LeRobot loader. Use `--data-backend parquet` to select the shared numeric reader explicitly. This reader follows the configured field mapping and action sequence keys, independently of robot type.

This reads all 200 episodes and 67,740 frames and writes `norm_stats.json` and `norm_metadata.json` under `$OUT`, preserving the source dataset. The numeric parquet reader supports this dataset's v3.0 format and includes partial batches. Action statistics use 20-step windows with terminal action repetition, matching OpenPI. The seven recorded delta-pose/gripper actions are padded to 32 dimensions without an additional delta transform.

To select pose and gripper only, run:

```bash
OBS_INDICES=1,2,3,4,5,6,19 python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state --repo-id "$DATA" \
  --output-dir "$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state7"
```

Indices are **one-based**, matching `~/openpi`. `--obs-indices 1,2,3,4,5,6,19` overrides the environment variable. Without either, the default is dimensions 1–19. Reordering is supported; empty, duplicate, nonpositive, out-of-range indices and selections larger than 32 dimensions are rejected.

## Stage1 SFT

Use `examples/sft/config/ur_rlt_stage1_sft_openpi_pi05.yaml`. Set `actor.model.model_path` to a directory containing an RLinf OpenPI `model.safetensors`. Dataset, default statistics paths, and two camera inputs are already configured for this machine.

Store the selection in YAML so Ray workers receive the same settings:

```yaml
actor:
  model:
    openpi_data:
      obs_indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
      norm_stats_path: /home/suchenyu/RLinf/rltoken-setup/outputs/ur5e_insert_black_rack_state19/norm_stats.json
    openpi:
      config_name: pi05_ur_state
      num_images_in_input: 2
```

Normalization, SFT, and rollout share `SelectState` before robot-specific transforms. The UR adapter owns field mapping, two camera inputs, and action representation. Franka configurations can enable the same selection with `openpi_data.obs_indices`; without it, their original behavior is preserved. The exterior camera maps to `base_0_rgb`; the wrist camera maps to `left_wrist_0_rgb`. Recompute statistics and update the path whenever the selection or order changes. Keep `norm_metadata.json` beside the statistics to validate the order on load.

The current `.venv` LeRobot loader supports v2.1, while this dataset uses v3.0. Before full SFT, adapt the dataset loader or create a separate compatible dataset copy. Only numeric normalization and input transforms have been verified. After resolving format compatibility and setting the model path, use:

```bash
bash examples/sft/run_vla_sft.sh ur_rlt_stage1_sft_openpi_pi05
```

## Stage2

After adapting the UR driver, task, and deployment, use the same data selection for the feature model and set the actor's `proprio_dim` to its length:

```yaml
rollout:
  rlt_feature_model:
    model_path: /第一阶段checkpoint/actor
    openpi_data:
      repo_id: scylearning/insert_black_rack_into_blue_slot_20260916_v01
      obs_indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
      norm_stats_path: /home/suchenyu/RLinf/rltoken-setup/outputs/ur5e_insert_black_rack_state19/norm_stats.json
    openpi:
      config_name: pi05_ur_state
      num_images_in_input: 2
      state_indices: []
actor:
  model:
    proprio_dim: 19
```

The VLA input and raw MLP proprio use the same selection. Keep the existing zero-based `openpi.state_indices` empty when using `openpi_data.obs_indices`; both backends reject simultaneous selection settings. The environment must provide the full raw 48-dimensional state in dataset order.

The hardware and task in `realworld_rlt_stage2_ac_mlp.yaml` still target Franka. This fragment documents UR model settings; it does not provide a UR deployment. No hardware control or full training was run.
