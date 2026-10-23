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

The launcher creates a timestamped directory under the repository’s real ``results/`` directory. Unless ``RAY_TMPDIR`` is already set, Ray stores its temporary files under ``results/ray/``. Training checkpoints remain under ``logs/``. Synchronous evaluation requests a new prediction after executing the current chunk. RTC uses RLinf's existing guidance sampler and replanning scheduler; the UR worker adds episode-budget completion. Both modes use 10 denoising steps, predict 20 actions and execute chunks of 3 actions. The 100 Hz background controller keeps tracking the last absolute TCP target while the model is busy.

Control Episodes and Read Results
---------------------------------

After each reset, press Enter once on the robot host's physical keyboard to start. Double Enter within 0.5 seconds records a failed reset; double Space records success. Each episode times out after 60 seconds. The default evaluation records 20 outcomes, homes the robot after the final outcome and exits. The UR wrapper blocks the remaining actions of an ended chunk; RTC drains and discards pending predictions before a new episode.

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

Reset opens the gripper, stops force mode, calls ``servoStop`` and ``moveL``, reapplies payload, waits 0.25 seconds, zeros the F/T sensor and waits another 0.50 seconds before resuming tracking. The controller clips position and velocity errors at 0.048 metres and 4.0 m/s. Rotation-error clipping is disabled by ``rotation_error_clip_degrees: null``. These UR implementations leave Franka's shared environment, transforms and controller behavior unchanged.

Validation Scope
-----------------

SDK fakes verify local action composition with nonzero tool offset, unchanged base-frame velocity input, idle anchors, binary gripper commands, reset ordering and force-sensor zeroing. Keyboard tests cover success, failed reset, timeout and stale chunk cancellation. Synthetic observations can check checkpoint loading and synchronous/RTC GPU inference without connecting hardware. These checks do not measure task success, contact quality or real robot timing; validate those on the configured UR setup.

Compare With VLA-Precision Execution
------------------------------------

Use the copied VLA-Precision backend to compare robot execution against the distributed RLinf path above. ``evaluations/realworld/vla_precision_ur/`` contains the single-UR hardware loop, recording, image preprocessing and UR policy transforms copied from the local VLA-Precision checkout. ``provenance.json`` records the source revision, file hashes and preserved methods, including changes present in that checkout. The original checkout is not needed at runtime.

Configure this path in ``evaluations/realworld/vla_precision_ur.yaml``. Hardware, camera, task, force-control and episode settings match the source YAML. The output root is ``results/vla-precision-ur``; ``model.checkpoint_dir`` selects the supplied RLinf actor checkpoint and ``model.norm_stats_path`` selects its statistics. ``model.network`` records the trained architecture: a 20-step prediction horizon, 32 internal action dimensions, seven physical action dimensions and RLT enabled. Inference uses the same BF16/FP32 mixed precision as the distributed UR entry, with both ``precision`` and ``embed_dtype`` set to ``bf16``. The statistics contain 19 state channels and 32 padded action channels; only seven action channels reach the robot.

The copied UR entry also accepts a compilation mode in ``model.network.openpi.torch_compile``. Set it to ``max-autotune`` to wrap the complete ``sample_actions`` method with ``torch.compile``, following OpenPI's PyTorch setting. The first inference performs compilation during the existing warmup before episode execution. Warmup displays total elapsed time in place once per second and prints the final duration on completion or interruption. The same timer is used by ``--check-model`` for its first inference. Changing input shapes can trigger recompilation, and compiled floating-point outputs can differ from the original path. A multi-input offline check found a maximum gripper-output difference of about 0.684 after compilation, so task behavior needs validation when selecting this mode. The boolean values retain their previous behavior: ``false`` disables local kernel compilation, while ``true`` compiles local kernels and keeps the Python sampling loop. Other accepted mode strings are ``default``, ``reduce-overhead`` and ``max-autotune-no-cudagraphs``. This extension applies only to the copied UR entry; the shared model loader and distributed UR/RTC entry retain their existing settings.

Activate the existing environment and check imports and paths before running. The optional GPU check uses synthetic observations and fixed diffusion noise to compare native OpenPI policy and RLinf model I/O; neither check connects hardware:

.. code-block:: bash

   source rltoken-setup/activate.sh
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check
   bash evaluations/realworld/run_ur5e_vla_precision.sh --check-model
   # Connect hardware, reset, warm up the model and wait for Enter:
   bash evaluations/realworld/run_ur5e_vla_precision.sh
   # Select a different self-contained YAML:
   bash evaluations/realworld/run_ur5e_vla_precision.sh --config /path/to/ur.yaml

This entry uses OpenPI's native ``Policy.infer()`` in one process, with an adapter for RLinf's network and checkpoint format. It retains VLA-Precision's RGB camera reads, 224-by-224 padding, state construction, pose deltas, gripper thread, force computation, terminal keyboard controls and video-save prompt. Keyboard input comes from the running terminal, including an SSH terminal. Double Enter records failure and resets; double Space records success. The default budget remains 20 episodes of at most 60 seconds each.

VLA-Precision predicts a full horizon and executes its first three actions synchronously. It sends force commands during action execution and has no Python background controller refreshing the target during inference; this differs from the RLinf controller above. The copied path also retains the source's absence of RLinf workspace clipping and does not enable RTC. The source YAML's ``model.infer_fps`` is unused by the hardware loop and is omitted here. These are properties of the comparison, rather than changes to the common robot code.

Matching outputs for fixed synthetic inputs checks the model I/O bridge, while real-world comparison tests the camera, control and timing path. The RLinf network implementation remains shared, so this comparison alone cannot rule out differences between that network and the official OpenPI model or determine checkpoint quality. No Franka or shared environment/controller implementation is changed.
