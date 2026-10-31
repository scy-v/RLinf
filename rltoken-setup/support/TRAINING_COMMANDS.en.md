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

Run on the local machine connected to the arm, gripper and cameras. Choose either inference method below. Prepare `.venv` using the [installation commands](INSTALL_COMMANDS.en.md) before first use. Both methods use the same environment:

```bash
cd ~/RLinf-rltoken
source .venv/bin/activate
```

### 4.1 Native RLinf Inference with UR Integration

Check the hardware, `rollout.model.model_path` and normalization path in the [inference config](../../evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml), and the workspace bounds in the [environment config](../../examples/embodiment/config/env/realworld_ur5e_sft_env.yaml).

```bash
bash evaluations/realworld/run_ur5e_eval.sh
```

Results directory: `results/<timestamp>-ur5e-eval/`. Append `runner.rtc.enabled=true` to enable RTC. Keyboard controls use the local physical keyboard.

### 4.2 VLA-Precision Inference

Use the standalone VLA-Precision inference workflow. Check the hardware, `model.checkpoint_dir`, `model.norm_stats_path` and reset pose in its [config](../../evaluations/realworld/vla_precision_ur.yaml).

```bash
bash evaluations/realworld/run_ur5e_vla_precision.sh
```

The config's `output_root` sets the results directory, currently `results/vla-precision-ur/`. Keyboard controls use the current terminal.

Both methods reset the arm at startup; press Enter when prompted to begin. During execution, press Space twice within 0.5 seconds to record success, or Enter twice to record failure and reset. Use `Ctrl+C` to exit.

## 5. Start Stage2 UR Online RL

```bash
bash examples/embodiment/run_realworld_async.sh ur_rlt_stage2_ac_mlp
```

[Training config](../../examples/embodiment/config/ur_rlt_stage2_ac_mlp.yaml) · [Environment config](../../examples/embodiment/config/env/realworld_ur5e_rlt_env.yaml). Freeze the step 5,000 feature model and train the MLP actor/critic. Current settings use 4 denoising steps, 3 executed actions per chunk and force mode `false`.

To enable continuous background tracking at a target rate of 100Hz, append `cluster.node_groups.0.hardware.configs.0.impedance.continuous_tracking=true` to the command.

Set `runner.timing_mode=detailed` to record startup sections while retaining total timings. Actor, rollout MLP and frozen VLA loads are recorded separately. Warmup records identify each synthetic invocation; when startup warmup is disabled, the first real inference is measured. Read `runner.logger.log_path/startup_timing/<component>_<rank>.jsonl` for wall time, process CPU time, storage reads, page faults and available compiler metrics. Device synchronization is added only during startup; nested stages and compiler metrics must not be added together.

Configure action lengths in two places: `rollout.rlt_feature_model.openpi.action_horizon` controls the VLA prediction and retained reference length (currently 20), and `actor.model.num_action_chunks` controls the reference prefix used by the MLP, its output length and the environment execution length (currently 3). Other length fields reference these settings. Set denoising iterations through `rollout.rlt_feature_model.openpi.num_steps` (currently 4). Changing the MLP chunk length changes network dimensions, so existing Stage 2 checkpoints cannot be reused directly.

After models load, press Enter to allow the first reset, then Enter again to start. Subsequent episodes reset automatically when they end; press Enter after reset to start the next episode. Exit other programs controlling the arm first. Use the local physical keyboard: double `.` within 0.5 seconds marks subsequent chunks as critical, double Space within 0.5 seconds confirms success, and `Ctrl+C` exits. MLP takeover becomes available after collecting 150 critical chunk transitions, completing 150 critic updates and synchronizing rollout weights.

Log: `~/RLinf-rltoken/logs/online_rltoken/<timestamp>-ur_rlt_stage2_ac_mlp/run_embodiment.log`. Checkpoints are saved every 1000 runner training steps.

Rotational PD gains now apply directly: online training and both supplied UR inference profiles use P=20 and D=4, equivalent to the former P=2000 and D=400 divided by 100. `frequency`/`rtde_freq` controls only continuous-tracking cadence. To migrate a custom legacy profile, divide both rotational gains by its original configured frequency before using the new formula.

The UR training config defaults to `env.train.flush_replay_on_episode_end=true`: success or timeout stops force control, reuses the pre-reset terminal observation returned by the environment, completes the final RLT next-state features, and submits the actual episode length. Reset waits for the actor to acknowledge replay ingestion; episodes without critical data are also acknowledged. Only critical transitions enter replay. The episode limit is `env.train.max_episode_chunks=200`, with `rollout_epoch=1`. Continuous training remains enabled; new data is available after episode completion, so initial warmup and MLP takeover may move to a later episode. Set `env.train.flush_replay_on_episode_end=false` to restore fixed batches of `env.train.max_chunks_per_rollout_epoch=50`. Episode replay requires one UR, one rollout worker, one actor, and the async launcher. An unfinished episode interrupted by an error or `Ctrl+C` is not submitted; exit still stops force control. `runner.timing_mode=basic` is the default: it records Learner update sections and the major Rollout/Env modules. Set it to `disabled` to turn off additional timing, or `detailed` to add submodules and diagnostics. Disabled mode reads no profiling clocks, starts no timing threads and emits no timing records; original RLinf logs and control pacing retain their behavior. Timing is controlled solely by `runner.timing_mode`.

Rollout modules cover post-receive input processing, inference, action sending and training-data submission. Env modules cover post-receive input processing, action execution and feedback publishing. Each worker's `work_ms` equals its major-module sum and excludes upstream input waiting, episode-end replay waits and reset. Detailed mode adds preprocessing, KV cache, z encoding, reference actions, MLP/output preparation, and Env action preparation, chunk execution, result processing, request construction, transition construction and sending. UR command processing, pacing, observation reads, action cadence and background PD are separate diagnostics; nested or background values must not be added again to major-module totals. With `continuous_tracking=true`, PD runs concurrently and reports its own cadence.

All additional timing uses worker-local host elapsed time without added device synchronization, cross-process barriers or waits. GPU operations can be asynchronous, so these values describe runtime delays rather than exact GPU kernel durations and are not directly comparable with earlier synchronized measurements. The run directory contains readable `timing.log` tables and raw `timing.tsv` records. `work_hz` is reciprocal duration; `action_interval_hz` and `measured_hz` report actual action/PD cadence.

`rollout.rlt_feature_compile_mode=max-autotune` compiles only the frozen OpenPI reference-action denoising loop; the prefix and RLT feature encoder keep their original execution paths; set it to `null` to disable this compilation. By default, `rollout.rlt_inference_warmup.enabled=true` runs three synthetic inferences before environment initialization/reset, without sending actions. Set it to `false` to initialize on the first actual inference.

Each `Learner update` now includes `sample_ms` (replay sampling), `prepare_ms` (batch preparation and device transfer), `critic_ms` and `actor_ms` (each forward, backward, clipping and optimizer update), `target_ms` (target update), and `other_ms` (remaining work and measurement overhead). `work_ms` covers the complete call, rather than only critic plus actor. When `actor_updated=false`, `actor_ms` is empty because the actor was skipped; tasks with a learned temperature also report `alpha_ms`. In detailed mode, `Runner step` records a complete round with its `step`, actual `critic_updates`, and `actor_updates`, including update calls, weight-sync waits, and scheduled checkpoint/evaluation work; the first round may also wait for warmup data. Sections use local host elapsed time without added device synchronization.

See [technical notes](TRAINING_NOTES.en.md) for weight conversion, data handling, Stage2 adaptation, and verification records. [中文](../训练命令.md).
