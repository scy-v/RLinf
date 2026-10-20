UR5e Evaluation
================

This guide deploys an RLinf OpenPI checkpoint on a UR5e with a DH Robotics PGI/PGE serial gripper and two RealSense cameras. It uses the standard evaluation runner, robot placement and camera drivers. Hardware dependencies were checked against `VLA-Precision <https://github.com/scy-v/VLA-Precision/tree/41edacde80d8c9d8146d610202299d71b0e79f7c>`_; its HTTP server and actor loop are not used. The UR driver adapts its Cartesian PD impedance calculation and force-mode execution.

Install and Configure
---------------------

Install the robot dependencies on the control host. On a host that also runs the model, include OpenPI. Use a separate venv when another training run is active.

.. code-block:: bash

   bash requirements/install.sh embodied --env ur --venv .ur-robot-venv
   # Combined robot and model host:
   bash requirements/install.sh embodied --model openpi --env ur --venv .ur-eval-venv

The existing UR extra supplies ``ur-rtde==1.6.3``, ``pydhgripper==1.0.2``, serial and RealSense dependencies. VLA-Precision pins RTDE 1.6.2; this integration retains RLinf's existing 1.6.3 pin. It does not install Franka SDKs.

The defaults in ``evaluations/realworld/realworld_ur5e_eval_pi05_sft.yaml`` and ``examples/embodiment/config/env/realworld_ur5e_sft_env.yaml`` follow VLA-Precision's pinned single-UR ``ur.yaml`` inference profile. The robot IP is ``192.168.201.11``. Exterior camera ``348522072391`` maps to ``exterior_image``; wrist camera ``944622073455`` maps to ``wrist_image``. Both use RLinf's RealSense defaults, 640×480 at 15 FPS. The reset TCP pose is ``[0.392232, 0.104898, 0.335162, -3.14, 0.0, -1.57]`` (base-frame XYZ and Euler XYZ, metres/radians). The PGI port is ``/dev/ur5e_left_gripper``, force 80, speed 100 and raw position limits 80–1000. Payload mass is 1.601 kg and centre of gravity is ``[0.011, -0.002, 0.052]`` metres.

The task workspace is X 0–0.90 m, Y −0.45–0.45 m and Z 0–0.60 m. Absolute Euler XYZ bounds use the nominal orientation with offsets of ±30°, ±30° and ±90°, converted to radians. UR uses its own TCP reset implementation without periodic joint resets. ``enable_random_reset: true`` and ``reset_pose_range: [0.03, 0.03, 0.03, 0, 0, 0]`` apply independent uniform ±3 cm offsets along XYZ with fixed orientation. Range entries use metres for translation and degrees for angular offsets, following VLA-Precision; the nominal pose itself still uses metres and radians. ``target_ee_pose`` matches the reset pose as the orientation anchor for safety clipping; the keyboard wrapper suppresses target-based episode completion. Keep the black-rack task description, dataset and normalization assets from your training run.

The workspace bounds ``ee_pose_limit_min`` and ``ee_pose_limit_max`` each contain six values: base-frame XYZ in metres and Euler XYZ in radians. They clip TCP targets produced by policy actions, independently of the random reset range. UR TCP reset follows its configured pose and range; these bounds do not provide trajectory protection for all physical motion. The preset supplies these values. Keep angular bounds continuous across −π when changing them; leaving a value empty does not disable the limits.

The supplied placement uses one GPU/robot host. For separate hosts, keep the same runner and move the ``env`` placement to a robot node group, as described in :doc:`realworld`. Connect the arm in remote-control mode. ``UR5eArm`` receives absolute XYZ + XYZW targets. Its background loop runs at a nominal 100 Hz, computes a clipped Cartesian PD wrench from the measured pose and velocity, rotates that wrench into the configured force task frame, and calls ``forceMode``. It keeps tracking the latest target between policy requests. The gains, clipping and rotational torque division by controller frequency follow VLA-Precision's running-controller formula. Joint reset stops force mode, executes ``moveJ`` and restarts tracking from the measured pose; normal TCP actions do not use ``moveL``.

Run the Policy
--------------

Activate the prepared evaluation environment, configure Ray as in :doc:`realworld`, and start the ordinary Stage1 policy through RLinf's entry script:

.. code-block:: bash

   source .ur-eval-venv/bin/activate
   bash evaluations/run_eval.sh realworld_ur5e_eval_pi05_sft

The nominal environment step rate is 100 Hz, matching the reference's ``action_fps``. Each policy call uses ten denoising steps and executes three actions, following the reference. The predicted horizon stays at 20 to match the trained RLT model. ``rollout.model.model_path`` supplies the converted OpenPI base; ``runner.ckpt_path`` overlays the step-30000 Stage1 weights from the ``20261018-22:40:04`` run. Configure these paths in YAML when deploying on another host. The normalization path, two images and one-based state indices match the local UR SFT config. The state vector is ordered as TCP pose, speed, force, torque and normalized measured gripper position (19 values after the Euler wrapper). The gripper action is an absolute opening in [0, 1], where 0 is closed and 1 is open; device limits are applied by the gripper driver.

Set ``runner.rtc.enabled: true`` in the entry YAML for RTC, or ``false`` for synchronous inference (the default). The model guidance switch follows this field. RTC uses RLinf's existing workers to overlap action execution with replanning and guide chunk continuity. The selected task environment is ``UR5eEnv-v1``; it provides ordinary reset and stepping independently of the RTC choice.

Keyboard Reset
--------------

On the robot host, press Enter once after the initial reset to start the policy. During execution, double-press Enter within 0.5 seconds to end the current episode and return to the configured reset pose. After every reset, the policy waits for another single Enter press. This marks an operator-requested reset, not success. Holding Enter does not count as two presses. Use the physical keyboard attached to the environment worker's host; typing into SSH does not generate these device events. If more than one keyboard is present, select the readable device before starting Ray:

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/your-keyboard-event-kbd

``keyboard_reward_wrapper: double_enter_reset`` selects the shared ``KeyboardResetWrapper``. ``keyboard_wrapper_kwargs.wait_for_start: true`` enables this start wait. Immediately before starting, ``observe()`` refreshes the robot and camera observation without moving. ``keyboard_wrapper_kwargs.double_press_interval`` sets the double-press interval, and ``auto_reset: true`` lets ``RealWorldEnv`` call the robot's own reset implementation. The wrapper suppresses target-based completion and per-episode step timeouts. ``max_steps_per_rollout_epoch: 30000`` remains the total evaluation step budget; a keyboard reset starts a new episode inside that budget. Synchronous execution discards the remaining actions of the current chunk. RTC discards the old pending prediction and bootstraps from the reset observation. These controls are independent of the robot driver and the RTC switch.

Action Semantics to Confirm
---------------------------

``UR5eEnv`` and ``FrankaEnv`` independently inherit ``CartesianEnv``, which owns stepping, observations and reset. Each robot supplies its hardware builder and end-effector behavior. Both use the existing ``RelativeFrame``/Euler wrappers and the same TCP delta composition. With the relative-frame wrapper enabled, RLinf transforms a six-value action with the current pose's adjoint, then adds translation to measured TCP position and left-multiplies the Euler increment into measured orientation.

VLA-Precision's UR controller instead computes ``p_next = p + R.apply(delta_p)`` and ``R_next = R * EulerXYZ(delta_r)``. Its direct local-pose composition differs from RLinf's adjoint-based six-vector conversion. The frame used for reset-relative observations also needs comparison with the collected dataset. Matching dimension names alone does not establish matching physical semantics. This integration deliberately preserves RLinf's behavior; confirm these differences against the collection code before physical deployment.

Validation
----------

The keyboard tests run the real evdev listener against a pipe-backed fake device and exercise reset through both UR and Franka environments in dummy mode, including long-press rejection, capture timestamps, chunk cancellation and episode bookkeeping. The fake-SDK tests cover the Cartesian PD wrench, error clipping and task-frame rotation; target holding without new policy calls; force-mode stop/restart for joint reset; failure propagation and connection cleanup; PGI units; RLinf delta reuse; and the 19-value model observation. ``ur5e_mock_sac_cnn`` exercises the existing embodied pipeline with fake hardware in CI. The integration does not establish real-robot success rates, contact behavior or timing; those require the physical UR5e and cameras.
