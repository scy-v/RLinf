# UR5e RLToken Training Preparation

[Chinese instructions](../训练命令.md).

Select the raw state dimensions with `obs_indices`, then use the same selection for normalization, SFT, and Stage2. The default is the first 19 dimensions: TCP pose (6), velocity (6), force/torque (6), and `gripper_raw_position` (1).

## Data and Normalization

```bash
source ~/RLinf/rltoken-setup/activate.sh
DATA="$HOME/.cache/huggingface/lerobot/scylearning/insert_black_rack_into_blue_slot_20260916_v01"
OUT="$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state19"
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state --repo-id "$DATA" \
  --obs-indices 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --output-dir "$OUT"
```

The statistics script chooses its reader by storage format: v3 defaults to numeric parquet; v2 defaults to the existing LeRobot loader. Use `--data-backend parquet` to select the shared numeric reader explicitly. This reader follows the configured field mapping and action sequence keys, independently of robot type.

This reads all 200 episodes and 67,740 frames and writes `norm_stats.json` and `norm_metadata.json` under `$OUT`, preserving the source dataset. The numeric parquet reader supports this dataset's v3.0 format and includes partial batches. Action statistics use 20-step windows with terminal action repetition, matching OpenPI. The seven recorded delta-pose/gripper actions are padded to 32 dimensions without an additional delta transform.

When using the environment variable, explicitly select all dimensions 1–19 for this dataset:

```bash
OBS_INDICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_ur_state --repo-id "$DATA" \
  --output-dir "$PWD/rltoken-setup/outputs/ur5e_insert_black_rack_state19"
```

Indices are **one-based**, matching `~/openpi`. `--obs-indices 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19` overrides the environment variable. Without either, the default is dimensions 1–19. Reordering is supported; empty, duplicate, nonpositive, out-of-range indices and selections larger than 32 dimensions are rejected.

## Stage1 SFT

Use `examples/sft/config/ur_rlt_stage1_sft_openpi_pi05.yaml`. The cached OpenPI JAX `pi05_base` has been converted to RLinf's PyTorch layout. `actor.model.model_path` points to `/home/suchenyu/RLinf/rltoken-setup/outputs/pi05_base_openpi`, containing `model.safetensors` and `config.json`. Dataset paths, statistics for dimensions 1–19, and two camera inputs are configured for this machine.

On another machine, prepare the normalization results above and the cached OpenPI weights, then convert with the same 20-step horizon as training:

```bash
JAX_PLATFORMS=cpu python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
  --input-model "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base" \
  --input-norm-stats "$OUT/norm_stats.json" \
  --output-model "$PWD/rltoken-setup/outputs/pi05_base_openpi" \
  --output-norm-stats "$PWD/rltoken-setup/outputs/pi05_base_openpi/scylearning/insert_black_rack_into_blue_slot_20260916_v01/norm_stats.json" \
  --action-dim 32 --action-horizon 20 --max-token-len 200
```

This converts the existing weights and copies normalization statistics. It does not download or train a model. Weights remain in a local ignored directory.

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

The UR installation target pins LeRobot to `d6ea3bbce0fc8c75138f02639ac4ceaf12a67829`, matching your `~/openpi` environment (package version 0.3.4, dataset format v3.0). OpenPI's compatibility import uses `lerobot.datasets.lerobot_dataset`. The UR adapter maps each sample's native `task` text to the prompt, using `meta/tasks.parquet`; the dataset task is `Insert the black rack into the blue slot`.

The conversion was checked tensor by tensor: after mapping the local OpenPI converter's output to RLinf's parameter layout, all 667 base weight tensors match exactly. The additional RLToken modules are initialized from scratch.

The original microbatch of 16 ran out of memory on one A800 80GB. The current configuration uses a microbatch of 2 and gradient checkpointing, while retaining a global batch of 64. Two consecutive real-data training steps passed forward propagation, backward propagation, and optimizer updates, using the default eight data workers and TensorBoard. `activate.sh` uses a separate Hugging Face Arrow cache under `outputs/hf-datasets-cache` to avoid reading caches generated by another `datasets` version in the OpenPI environment. The original LeRobot dataset location is unchanged. Start training with:

```bash
bash examples/sft/run_vla_sft.sh ur_rlt_stage1_sft_openpi_pi05
```

The configuration uses GPU 0 for 20000 steps and saves a checkpoint every 2000 steps. About 106GB of disk space remains, which cannot hold all checkpoints from the full run. Transfer older checkpoints as training proceeds, or place logs and checkpoints on a disk with enough capacity. The preflight did not test checkpoint saving or restoration, and the full training run has not been executed.

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
