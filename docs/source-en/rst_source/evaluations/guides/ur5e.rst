UR5e Evaluation
================

Deploy an RLinf OpenPI checkpoint on a UR5e with a DH Robotics PGI/PGE gripper and two RealSense cameras. This guide uses UR-specific observation, action and episode controls matching the local VLA-Precision UR profile, while retaining RLinf's model sampling and RTC replanning.

Prepare the Environment
-----------------------

Install the OpenPI and UR dependencies on the GPU/robot host, then activate the resulting environment. See :doc:`realworld` for robot placement across separate hosts.

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env ur \
       --venv .venv --no-root --install-rlinf
   source rltoken-setup/activate.sh

This creates the model and hardware environment. The UR target includes RTDE 1.6.3, the serial gripper and RealSense dependencies.

Set the Hardware and Model Paths
---------------------------------

Edit ``evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml`` for the hardware endpoints and checkpoint. ``rollout.model.model_path`` points directly to the trained ``actor`` directory containing ``model_state_dict/full_weights.pt``. ``rollout.model.openpi_data.norm_stats_path`` points to the supplied ``checkpoints/norm_stats.json``. ``runner.ckpt_path`` is null because the model factory loads the full checkpoint once. The UR profile sets both ``precision`` and ``embed_dtype`` to ``bf16``. The OpenPI loader keeps normalization, vision input and action-head parameters in FP32 while using BF16 for the main backbone and embeddings, matching Franka's default mixed precision.

The supplied hardware profile uses IP ``192.168.201.11``, gripper port ``/dev/ur5e_left_gripper``, exterior camera ``348522072391`` and wrist camera ``944622073455``. Cameras capture 640 by 480 images at 15 FPS. Gripper force is 60, speed is 100 and raw travel is 0 to 1000. The model's gripper action is thresholded at 0.7: values above it open; other values close. Measured gripper feedback remains raw position divided by 1000.

Set task and motion fields in ``examples/embodiment/config/env/realworld_ur5e_sft_env.yaml``. The fixed reset TCP pose is ``[0.389694, 0.107973, 0.382018, -3.14, 0.0, -1.57]``: base-frame XYZ in metres and Euler XYZ in radians. Random reset is disabled. Workspace bounds remain active. The force-mode selection vector is ``[1, 1, 1, 0, 0, 1]``; payload is 1.601 kg with centre of gravity ``[0.011, -0.002, 0.052]`` metres. Confirm these values against the physical setup and put the UR controller in remote-control mode.

Run Synchronous or RTC Evaluation
----------------------------------

Use the UR entry to stop cleanly after the configured episode count. Activate the environment before launching it:

.. code-block:: bash

   bash evaluations/realworld/run_ur5e_eval.sh
   # Overlap action execution and model replanning:
   bash evaluations/realworld/run_ur5e_eval.sh runner.rtc.enabled=true

The launcher creates a timestamped directory under the repository’s real ``results/`` directory. Unless ``RAY_TMPDIR`` is already set, Ray stores its temporary files under ``/tmp/ray/``. This short path avoids the Unix-domain socket path-length limit. Training checkpoints remain under ``logs/``. Synchronous evaluation requests a new prediction after executing the current chunk. RTC uses RLinf's existing guidance sampler and replanning scheduler; the UR worker adds episode-budget completion. Both modes use 10 denoising steps, predict 20 actions and execute chunks of 3 actions. The supplied profile enables continuous PD tracking during active episodes. Set ``cluster.node_groups[0].hardware.configs[0].impedance.continuous_tracking=false`` for one PD calculation and force command per action, following the standalone VLA-Precision inference backend.

The UR evaluation and standalone VLA-Precision entry both create RTDE connections with only the robot IP, using the SDK's robot-dependent default communication frequency (500 Hz on UR5e). ``impedance.frequency: 100.0`` sets only the update rate when continuous tracking is enabled; it does not scale torque or set action cadence. Rotational gains apply directly: the supplied profiles use ``kp_rotation: 20.0`` and ``kd_rotation: 4.0``, equivalent to the former 2000/400 gains divided by 100. The standalone profile uses ``kp_rot`` and ``kd_rot`` for the same gains. When migrating a custom profile, divide both old rotational gains by its original configured frequency, then retain those gains when changing the update rate. Standalone omitted-gain defaults are 32/6.4, preserving the former 4000/800 defaults divided by 125. ``env.eval.override_cfg.step_frequency`` controls action cadence.

Per-action UR control uses the same measured pose and velocity for target composition and impedance calculation.

The distributed evaluation profile enables the camera preview through ``env.eval.override_cfg.enable_camera_player: true``. Set it to ``false`` to disable the window. Model images use RGB; the UR preview converts its own copies to BGR for OpenCV. Preview updates follow observation refreshes.

Inspect Startup and Loop Timing
-------------------------------

The distributed UR entry loads the model, resets the robot and performs a discarded warmup inference from real observations before asking for Enter. Warmup actions are never executed, and compilation time does not consume the first episode's timeout. ``[LOAD]`` and ``[WARMUP]`` messages report elapsed time every second. Routine load, warmup, inference, loop and episode timings are always reported; completed load, warmup, inference and loop measurements are saved when ``env.eval.ur_evaluation.save_logs`` is true.

With ``env.eval.ur_evaluation.save_logs: true`` (the default), the distributed entry writes VLA-style YAML prediction records and a combined timing JSONL under ``results/rlinf-ur/logs/all_logs``. ``latest.yaml`` and ``latest.jsonl`` in its parent log directory point to the same run. ``env.eval.ur_evaluation.log_dir`` selects this directory; run-specific configuration, metrics and raw execution records remain under ``results/rlinf-ur/runs``. Each YAML prediction includes the input ``state``, ``actions``, ``actions_delta``, prompt, episode metadata and a global ``infer_index``; ``infer_time`` retains the same integer for compatibility. Match this ID to the environment and rollout timing rows. Their ``component`` field distinguishes inference from execution. Detailed action rows additionally carry ``action_index`` within the chunk. Loading and warmup are separate events without a prediction record. ``episode_boundary: true`` marks environment samples that may include reset and operator waiting. RTC writes each prediction with its ``chunk_id``; ``rtc_step`` summaries cover a time window rather than one inference. Console loop summaries are throttled to once per second. Synchronous inference between chunks still reduces the overall action rate.

Set ``env.eval.ur_evaluation.action_timing_enabled=true`` to print detailed per-action timings and, when saving is enabled, include them in the combined timing JSONL and the run-specific ``ur5e-actions.jsonl``. Its ``control_sec`` and ``control_hz`` cover action processing plus pacing sleep, while ``observation_sec`` is measured afterwards. The switch defaults to false and does not disable routine timings. ``save_logs: false`` disables the UR action/state, timing and episode-result files while leaving console output enabled; Ray logs and video recording have separate controls. With saving enabled, action/state records are saved even when detailed action timings are off.

With ``continuous_tracking: true``, ``[PD]`` messages report the measured background force-send rate separately from the foreground action rate. The evaluation wrapper collects these summaries about once per second while stepping, and at episode end, reset and close. With log saving enabled, they appear as ``component: controller``, ``event: force_mode`` in the combined timing JSONL, independently of ``action_timing_enabled``. ``target_hz`` is the configured rate; ``measured_hz`` uses intervals between successful ``forceMode`` call completions. ``send_interval_sec_mean`` and ``send_interval_sec_max`` describe average and longest intervals; ``work_overruns`` counts cycles whose read, compute and send work exceeds the configured period. ``held_target_samples`` counts updates that reuse the previous target. The controller accumulates counters without file I/O, including during inference within an active episode; measurements are written when the wrapper next collects them. Force control is paused during warmup and operator waiting. Reset and waiting pauses are excluded from the rate. These measurements describe SDK call timing, not RTDE packet freshness or a hard real-time guarantee.

Control Episodes and Read Results
---------------------------------

After each reset, press Enter once on the robot host's physical keyboard to start. Double Enter within 0.5 seconds records a failed reset; double Space records success. Each episode times out after 60 seconds. The default evaluation records 20 outcomes, homes the robot after the final outcome and exits. The UR wrapper blocks the remaining actions of an ended chunk; RTC drains and discards pending predictions before a new episode.

The evaluation profile sets ``impedance.start_paused: true``. Connecting and resetting prepare the arm with force control paused. Warmup and waiting for Enter keep it paused; Enter explicitly starts force control from the measured pose. Success, failure and timeout stop force mode before recording the result and resetting. Each subsequent episode requires another Enter, and final homing leaves force mode paused. Initialization and episode resets still execute the configured ``moveL`` homing motion.

Set ``RLINF_KEYBOARD_DEVICE`` before launching Ray if device selection is ambiguous:

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/your-keyboard-event-kbd

Select a readable physical keyboard event device. Its input events are distinct from text typed into an SSH terminal.

Configure the budget under ``env.eval.ur_evaluation``: ``num_episodes``, ``episode_timeout_sec``, ``double_press_interval`` and ``wait_for_start``. After each outcome, ``ur5e-evaluation.json`` records the status, success flag, duration and executed action count, plus aggregate success rate. Videos and runner metrics use the same log directory. ``max_steps_per_rollout_epoch`` is an additional upper bound on execution steps.

Match Observation and Action Semantics
--------------------------------------

Keep the deployment representation consistent with the training data. The UR adapter removes the controller's TCP offset to compute EE-relative observations and local EE actions, then reapplies that offset to obtain the commanded TCP target. Pose is relative to the measured EE pose at reset; RTDE linear velocity, angular velocity, force and torque remain in their original base-frame representation. The Euler-converted state has 19 values: pose 6, velocity 6, force/torque 6 and measured gripper opening 1.

Camera images retain the full field of view: the UR adapter converts the RealSense BGR frame to RGB and letterboxes it to 224 by 224 using the OpenPI PIL resize convention. This replaces the shared centre-square crop only for UR.

UR actions use ``p_target = p + R * delta_p`` and ``R_target = R * EulerXYZ(delta_r)`` in the EE frame. UR handles this itself and applies only the Euler observation wrapper; ``use_relative_frame`` is false to avoid applying the shared adjoint transform as well. Idle hold tests base-frame position axes against 0.0001 metres and the complete rotation increment against 0.001 radians, latching inactive axes at their hold targets.

Reset opens the gripper, stops force mode, calls ``servoStop`` and ``moveL``, reapplies payload, waits 0.25 seconds, zeros the F/T sensor and waits another 0.50 seconds before accepting the next action. The controller clips position and velocity errors at 0.048 metres and 4.0 m/s. Rotation-error clipping is disabled by ``rotation_error_clip_degrees: null``. These UR implementations leave Franka's shared environment, transforms and controller behavior unchanged.

Validation Scope
-----------------

SDK fakes verify local action composition with nonzero tool offset, unchanged base-frame velocity input, idle anchors, binary gripper commands, reset ordering and force-sensor zeroing. Keyboard tests cover success, failed reset, timeout and stale chunk cancellation. Synthetic observations can check checkpoint loading and synchronous/RTC GPU inference without connecting hardware. These checks do not measure task success, contact quality or real robot timing; validate those on the configured UR setup.

Compare With VLA-Precision Execution
------------------------------------

Use the copied VLA-Precision backend to compare robot execution against the distributed RLinf path above. ``evaluations/realworld/vla_precision_ur/`` contains the single-UR hardware loop, recording, image preprocessing and UR policy transforms copied from the local VLA-Precision checkout. ``provenance.json`` records the source revision, file hashes and preserved methods, including changes present in that checkout. The original checkout is not needed at runtime.

Configure this path in ``evaluations/realworld/vla_precision_ur.yaml``. Hardware, camera, task, force-control and episode settings follow the local deployment profile. Match ``robot.action_fps`` and ``env.eval.override_cfg.step_frequency`` when comparing action cadence. The output root is ``results/vla-precision-ur``; ``model.checkpoint_dir`` selects the supplied RLinf actor checkpoint and ``model.norm_stats_path`` selects its statistics. ``model.network`` records the trained architecture: a 20-step prediction horizon, 32 internal action dimensions, seven physical action dimensions and RLT enabled. Inference uses the same BF16/FP32 mixed precision as the distributed UR entry, with both ``precision`` and ``embed_dtype`` set to ``bf16``. The statistics contain 19 state channels and 32 padded action channels; only seven action channels reach the robot.

The standalone policy uses RLinf's installed model input and output transforms. Both UR entries use the shared RLinf observation conversion, including CPU image normalization, NHWC image layout and float32 state. The standalone adapter directly returns RLinf's decoded action chunk, without repeating observation conversion or output decoding. In both entries, ``openpi.action_horizon`` controls the prediction length (20 by default), and ``openpi.action_chunk`` controls the returned and executed length (three by default). ``num_action_chunks`` references ``openpi.action_chunk``. Change ``rollout.model.openpi.action_chunk`` for distributed inference or ``model.network.openpi.action_chunk`` for standalone inference; the robot executes the entire returned chunk.

The standalone entry always prints routine load, warmup and inference-loop timings and, when ``record.save_logs: true`` (the default), saves their measurements in ``logs/all_logs/timing_<timestamp>.jsonl`` under its experiment directory. Set ``record.action_timing_enabled: true`` in the standalone YAML to also print detailed action measurements and save them when logging is enabled: state reads, target construction (including TCP-offset reads), impedance calculation, command sending, gripper target updates, remaining processing and pacing sleep. File durations use seconds; console action durations use milliseconds. The action diagnostic switch defaults to false and leaves routine console timings enabled. ``record.save_logs: false`` disables saved timings, action/input-state records, episode records and summary plots without disabling terminal output or video recording.

Both standalone log files have latest links: ``logs/latest.yaml`` for input states, action chunks and episode results, and ``logs/latest.jsonl`` for timings. Each normal loop timing uses the same ``infer_index`` as the YAML prediction; detailed action timings also identify ``action_index`` within that chunk. Routine loop timing is saved before episode reset or operator waiting. Prediction records are queued without dropping them when the writer is busy.

The copied UR entry also accepts a compilation mode in ``model.network.openpi.torch_compile``. Set it to ``max-autotune`` to wrap the complete ``sample_actions`` method with ``torch.compile``, following OpenPI's PyTorch setting. The first inference performs compilation during the existing warmup before episode execution. Loading and warmup display total elapsed time in place once per second and prints the final duration on completion or interruption. The same timer is used by ``--check-model`` for its first inference. Changing input shapes can trigger recompilation, and compiled floating-point outputs can differ from the original path. A multi-input offline check found a maximum gripper-output difference of about 0.684 after compilation, so task behavior needs validation when selecting this mode. The boolean values retain their previous behavior: ``false`` disables local kernel compilation, while ``true`` compiles local kernels and keeps the Python sampling loop. Other accepted mode strings are ``default``, ``reduce-overhead`` and ``max-autotune-no-cudagraphs``. The distributed UR entry selects complete-sampler compilation separately with ``rollout.sampler_compile_mode``; its supplied value is also ``max-autotune``. Use ``rollout.sampler_compile_mode=null`` to keep only the existing local-kernel setting. The shared model loader is unchanged.

Startup diagnostics keep the existing total load and warmup durations and add ``startup_stage``, ``startup_summary`` and ``startup_compile`` records to the same timing JSONL files. Load stages separate model construction, transform resources, checkpoint mapping, parameter copying, precision conversion and device transfer. Each stage includes process CPU time, Linux storage-read bytes and page-fault counts. A memory-mapped checkpoint may perform its disk reads during ``checkpoint.apply`` rather than ``checkpoint.read``. Compare both stages across runs; a zero read-byte delta indicates that no storage reads were charged to this process during that stage, rather than proving which cache supplied every byte.

Warmup records separate input processing, action sampling and output decoding. Compiler records report frame compilation, runtime autotuning, CUDA graph setup and available compiler-cache hit counters. These are explanatory measurements inside the sampling call and must not be added to its wall time. A compiled full sampler is measured as one call so profiling does not split its graph. Online RLT additionally separates prefix encoding, RL-token extraction, reference-action sampling and MLP routing; with ``runner.timing_mode=detailed``, each worker writes ``startup_timing/<component>_<rank>.jsonl`` under ``runner.logger.log_path``. Startup measurement synchronizes an initialized accelerator at stage boundaries; ordinary control and training iterations retain their existing timing behavior. The first online inference is measured when synthetic startup warmup is disabled.

Activate the existing environment and check imports and paths before running. The optional GPU check uses synthetic observations and fixed diffusion noise to compare the standalone adapter and RLinf worker model I/O; neither check connects hardware:

.. code-block:: bash

   source rltoken-setup/activate.sh
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check-model
   # Connect hardware, reset, warm up the model and wait for Enter:
   bash evaluations/realworld/run_ur5e_vla_precision.sh
   # Select a different self-contained YAML:
   bash evaluations/realworld/run_ur5e_vla_precision.sh --config /path/to/ur.yaml

This entry uses OpenPI's native ``Policy.infer()`` in one process, with an adapter for RLinf's network and checkpoint format. It retains VLA-Precision's RGB camera reads, 224-by-224 padding, state construction, pose deltas, gripper thread, force computation, terminal keyboard controls and video-save prompt. Keyboard input comes from the running terminal, including an SSH terminal. Double Enter records failure and resets; double Space records success. The default budget remains 20 episodes of at most 60 seconds each.

VLA-Precision predicts a full horizon and executes its first three actions synchronously. The standalone default, ``robot.force_mode.continuous_tracking: false``, sends one force command per action. Set this YAML value to ``true`` to enable a local background controller: the foreground updates absolute TCP targets at ``robot.action_fps``, and the background reads current state, recalculates the existing PD formula and calls ``forceMode`` at ``robot.force_mode.rtde_freq``. Each target is acknowledged after its first successful send; this does not wait for the arm to reach the target. Between predictions, the controller keeps tracking the last absolute target.

The standalone background controller starts when the episode begins, after Enter. Success, failure, timeout, reset and close stop it. Reset waits for the thread to exit before homing, applying payload and zeroing F/T; force control stays off during warmup and between episodes. Controller failures abort execution, and a failed stop prevents homing and sensor zeroing. The existing zero-output filter still disables force selection for filtered actions. ``[PD]`` logs and saved ``controller`` timing events report actual successful send frequency. In continuous mode, per-action ``send_ms`` measures target submission and acknowledgement, while PD computation runs in the background. This option belongs to the copied standalone backend and does not use RLinf's robot controller.

The distributed UR entry independently selects per-action or continuous tracking through its impedance config. Arm and gripper connections run inside the RLinf environment process when their controller is on that same node. The gripper performs serial I/O in a background thread, and complete camera observations refresh after the last action of each synchronous chunk. The copied path also retains the source's absence of RLinf workspace clipping and does not enable RTC. The source YAML's ``model.infer_fps`` is unused by the hardware loop and is omitted here.

Matching outputs for fixed synthetic inputs checks the model I/O bridge, while real-world comparison tests the camera, control and timing path. The RLinf network implementation remains shared, so this comparison alone cannot rule out differences between that network and the official OpenPI model or determine checkpoint quality. No Franka or shared environment/controller implementation is changed.
