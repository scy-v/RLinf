# UR5e RLToken Technical Notes

Data processing, weight conversion, compatibility, and verification records. For daily use, see [training commands](TRAINING_COMMANDS.en.md). [Chinese notes](TRAINING_NOTES.zh.md).

Select the raw state dimensions with `obs_indices`, then use the same selection for normalization, SFT, and Stage2. The default is the first 19 dimensions: TCP pose (6), velocity (6), force/torque (6), and `gripper_raw_position` (1).

## Data and Normalization

```bash
source ~/RLinf-rltoken/rltoken-setup/activate.sh
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

Use `examples/sft/config/ur_rlt_stage1_sft_openpi_pi05.yaml`. The cached OpenPI JAX `pi05_base` has been converted to RLinf's PyTorch layout. `actor.model.model_path` points to `/home/suchenyu/RLinf-rltoken/rltoken-setup/outputs/pi05_base_openpi`, containing `model.safetensors` and `config.json`. Dataset paths, statistics for dimensions 1–19, and two camera inputs are configured for this machine.

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
      norm_stats_path: /home/suchenyu/RLinf-rltoken/rltoken-setup/outputs/ur5e_insert_black_rack_state19/norm_stats.json
    openpi:
      config_name: pi05_ur_state
      num_images_in_input: 2
```

Normalization, SFT, and rollout share `SelectState` before robot-specific transforms. The UR adapter owns field mapping, two camera inputs, and action representation. Franka configurations can enable the same selection with `openpi_data.obs_indices`; without it, their original behavior is preserved. The exterior camera maps to `base_0_rgb`; the wrist camera maps to `left_wrist_0_rgb`. Recompute statistics and update the path whenever the selection or order changes. Keep `norm_metadata.json` beside the statistics to validate the order on load.

The UR installation target pins LeRobot to `d6ea3bbce0fc8c75138f02639ac4ceaf12a67829`, matching your `~/openpi` environment (package version 0.3.4, dataset format v3.0). OpenPI's compatibility import uses `lerobot.datasets.lerobot_dataset`. The UR adapter maps each sample's native `task` text to the prompt, using `meta/tasks.parquet`; the dataset task is `Insert the black rack into the blue slot`.

The conversion was checked tensor by tensor: after mapping the local OpenPI converter's output to RLinf's parameter layout, all 667 base weight tensors match exactly. The additional RLToken modules are initialized from scratch.

The current configuration uses four A800 GPUs, micro batch 2 per GPU and global batch 32, with four gradient-accumulation steps per optimizer update. `activate.sh` keeps Hugging Face Arrow caches under `outputs/hf-datasets-cache` to separate caches produced by other `datasets` versions. The original LeRobot data stays in its usual location. Use the author's SFT launcher:

```bash
bash examples/sft/run_vla_sft.sh ur_rlt_stage1_sft_openpi_pi05
```

Training uses GPUs 0–3 for 30000 total steps and saves weights every 5000 steps. `runner.resume_dir` still points to the full step-5000 checkpoint from the `20261017-22:41:05` run; the command therefore resumes training. Set that field to `null` for a fresh run. The `20261018-22:40:04` run produced step-30000 weights at `/data/suchenyu/RLinf-rltoken/logs/20261018-22:40:04-ur_rlt_stage1_sft_openpi_pi05/ur_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt`. `~/RLinf-rltoken/logs` is a symlink to the data-disk log directory.

The configuration uses `is_lora: False`, `train_expert_only: False` and the author's joint SFT forward. `no_shard` matches the author's Stage1 configuration: each GPU retains full parameters, gradients and optimizer states. `gradient_checkpointing: False` also matches the author's configuration; the original `Pi0.sft_forward()` additionally calls `gradient_checkpointing_disable()` on every forward.

### Comparison with the Author's Franka Stage1

The baseline is upstream commit `3d0470015d98919d48ac65fad3be13253c1ee8ba`. Byte comparison found 22 unchanged entry, model and update files, including `run_vla_sft.sh`, `train_vla_sft.py`, `pi0.py`, `rlt_token_transformer.py`, the SFT data loader and the base SFT worker. Nine methods in files with checkpoint changes also matched, covering data loading, the loss entry point, optimizer construction and updates. The saved TensorBoard configuration matches the current training YAML settings except for the launcher-generated log directory.

The PI₀.₅ backbone, RLT encoder/decoder, `rlt_loss + 1.0 * vla_loss`, RLT feature dimension 2048, two layers, eight heads, prefix length 1024, masks and joint-training settings match upstream. Peak LR `1e-5`, AdamW settings, clipping at 1, warmup of 1000, cosine scheduler type and minimum LR `2.5e-6` also match. Keep the following differences explicit:

| Setting | Author's Franka | Current UR |
|---|---|---|
| GPUs, micro / global batch | 1 GPU, 16 / 64 | 4 GPUs, 2 / 32 |
| Valid camera inputs | One | Exterior and wrist |
| Task and state | Franka data adapter | UR data adapter and one-based dimensions 1–19 |
| Maximum steps and LR schedule length | 20000 | 30000, extending the cosine decay period |
| Save interval | 2000 | 5000 |
| Saved content | Weights and full training state | Weights only; new checkpoints do not support full resume |
| Terminal logging | Every step | Every 10 steps; TensorBoard still logs each step |
| Starting state | Fresh by default | Current config resumes step 5000 |
| Gradient-checkpointing config | False | True; both SFT forwards actively disable it |

UR also pins a LeRobot revision supporting format v3 and adds compatible normalization-statistics loading. These are data-path adaptations. All 54 installed Python files from `rlinf-openpi==0.1.1` and all 221 installed LeRobot Python files match their installation-record hashes. Seventeen input-selection tests, eight logging tests and two CPU checkpoint tests passed. This audit did not restart GPU training.

The accurate description is therefore "the author's training algorithm with UR data adaptation and the settings above", rather than "only obs_indices was added" or "all settings except batch and placement are identical".

The current OpenPI model cannot switch directly to working LoRA training with `is_lora: True`: the generic PEFT defaults target an unsupported `ModuleList`, and attention LoRA in the native `*_lora` Gemma variants is not implemented. The VLM, action expert, and RLToken training scopes need explicit adaptation and validation first. The current command uses full-parameter training.

## Stage2

After adapting the UR driver, task, and deployment, use the same data selection for the feature model and set the actor's `proprio_dim` to its length:

```yaml
rollout:
  rlt_feature_model:
    model_path: /第一阶段checkpoint/actor
    openpi_data:
      repo_id: scylearning/insert_black_rack_into_blue_slot_20260916_v01
      obs_indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
      norm_stats_path: /home/suchenyu/RLinf-rltoken/rltoken-setup/outputs/ur5e_insert_black_rack_state19/norm_stats.json
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
