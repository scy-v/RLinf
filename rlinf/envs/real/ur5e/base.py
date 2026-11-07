# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UR5e observations and local EE actions with RTDE impedance tracking."""

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.config import get_episode_step_limit
from rlinf.envs.real.cartesian import CartesianEnv, CartesianEnvConfig
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics import Robot
from rlinf.robotics.parts.arms.ur5e import UR5eRobotState
from rlinf.robotics.parts.end_effectors import EndEffector
from rlinf.robotics.robots.ur5e import UR5eConfig, UR5eRobot
from rlinf.utils.logging import get_logger
from rlinf.utils.progress import StageTiming, current_worker_timing, emit_timing_record

from .motion import URTargetComposer, matrix_pose, pose_matrix, tcp_to_ee


@dataclass
class UR5eEnvConfig(CartesianEnvConfig):
    """UR TCP reset settings; range uses metres and angular offsets in degrees."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_pose_range: np.ndarray = field(
        default_factory=lambda: np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
    )

    reference_frame: str = "tcp"
    step_frequency: float = 25.0
    action_idle_hold_enabled: bool = True
    action_idle_position_threshold: float = 0.0001
    action_idle_rotation_threshold: float = 0.001
    gripper_close_threshold: float = 0.7
    payload_settle_seconds: float = 0.25
    ft_zero_settle_seconds: float = 0.50
    manual_episode_control_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reset_pose_range = np.asarray(self.reset_pose_range, dtype=np.float64)
        if (
            self.reset_pose_range.shape != (6,)
            or not np.isfinite(self.reset_pose_range).all()
        ):
            raise ValueError("Expected reset_pose_range to contain six finite values.")
        self.reset_pose_range = np.abs(self.reset_pose_range)
        if self.reference_frame not in {"tcp", "base"}:
            raise ValueError("Expected UR reference_frame 'tcp' or 'base'.")
        if not 0 <= self.gripper_close_threshold <= 1:
            raise ValueError("Expected gripper_close_threshold in [0, 1].")
        settings = (
            self.action_idle_position_threshold,
            self.action_idle_rotation_threshold,
            self.payload_settle_seconds,
            self.ft_zero_settle_seconds,
        )
        if not np.isfinite(settings).all() or min(settings[2:]) < 0:
            raise ValueError(
                "Expected finite UR thresholds and nonnegative settle times."
            )


class UR5eEnv(CartesianEnv):
    """Compose local EE deltas without changing other Cartesian robots."""

    CONFIG_CLS = UR5eEnvConfig
    HARDWARE_CONFIG_CLS = UR5eConfig
    STATE_CLS = UR5eRobotState
    TELEOP = ("keyboard",)
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ("Quat2EulerWrapper",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._composer = URTargetComposer()
        self._reference_ee_pose: np.ndarray | None = None
        self._tcp_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self._gripper_target: float | None = None
        self._camera_steps_remaining = 0
        self._camera_refresh_on_request = False
        self._action_trace_enabled = True
        self._last_action_started: float | None = None
        self._last_target_sent: float | None = None
        self._infer_index: int | None = None
        self._chunk_action_index = 0
        self._closed = False
        from rlinf.utils.robot_lifecycle import register_robot_cleanup

        register_robot_cleanup(self.close)
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            try:
                self.close()
            except BaseException:
                from rlinf.utils.logging import get_logger

                get_logger().exception("Failed to close UR after initialization error.")
            raise

    def close(self) -> None:
        """Stop the arm before releasing cameras, gripper and other resources."""
        if self._closed:
            return
        self._closed = True
        from rlinf.utils.robot_lifecycle import unregister_robot_cleanup

        errors = []
        arm = getattr(self, "_arm", None)
        robot = getattr(self, "robot", None)
        player = getattr(self, "camera_player", None)
        for callback in (
            arm.disconnect if arm is not None else None,
            robot.disconnect if robot is not None else None,
            getattr(player, "stop", None),
        ):
            if callback is not None:
                try:
                    callback()
                except BaseException as error:
                    errors.append(error)
        self._cameras = {}
        unregister_robot_cleanup(self.close)
        if errors:
            raise errors[0]

    def stop_motion(self) -> np.ndarray:
        """Stop force control and return the zero-motion terminal hold action."""
        if not self.config.is_dummy:
            self._arm.stop_motion()
        gripper = self._gripper_target
        if gripper is None:
            gripper = float(self._robot_state.gripper_position > 0.5)
        return np.r_[np.zeros(6), gripper].astype(np.float32)

    def start_motion(self) -> None:
        """Enable UR force control when an episode starts."""
        if not self.config.is_dummy:
            self._arm.start_motion()

    def drain_control_timing(self) -> dict[str, Any] | None:
        """Collect background PD timing without reading robot or camera state."""
        if self.config.is_dummy:
            return None
        return self._arm.drain_control_timing()

    def begin_action_chunk(self, size: int, infer_index: int | None = None) -> None:
        """Refresh images at the chunk boundary and associate action records."""
        if size < 1:
            raise ValueError(f"Expected positive UR action chunk size, got {size}.")
        self._infer_index = infer_index
        self._camera_steps_remaining = size
        self._chunk_action_index = 0
        self._last_target_sent = None

    def defer_camera_refresh(self) -> None:
        """Let RTC requests own image refreshes instead of individual actions."""
        self._camera_refresh_on_request = True

    def refresh_camera_observation(self) -> dict[str, Any]:
        """Refresh policy images without sending an action or changing proprioception."""
        if self.config.is_dummy:
            obs = self._get_observation()
            return {key: obs[key] for key in ("frames", "depths") if key in obs}
        reading = {
            name: camera.get_observation() for name, camera in self._cameras.items()
        }
        self._last_frames, self._last_depths = self._frames_from(reading)
        return copy.deepcopy({"frames": self._last_frames, "depths": self._last_depths})

    @staticmethod
    def episode_wrappers(cfg: Mapping[str, Any]) -> tuple:
        """Attach UR episode controls without registering a global mode."""
        settings = cfg.get("ur_evaluation", {})
        if not settings.get("enabled", False):
            if cfg.get("keyboard_reward_wrapper") == "rlt_policy_switch":
                from functools import partial

                return (
                    partial(
                        URTrainingEpisodeWrapper,
                        max_episode_steps=get_episode_step_limit(cfg),
                        ignore_terminations=cfg.get("ignore_terminations", False),
                        timing_enabled=cfg.get("online_timing_enabled", False),
                    ),
                )
            return ()
        from functools import partial

        from .evaluation import UREvaluationWrapper

        return (
            partial(
                UREvaluationWrapper,
                **{key: value for key, value in settings.items() if key != "enabled"},
            ),
        )

    def _init_action_obs_spaces(self) -> None:
        super()._init_action_obs_spaces()
        for name in self.observation_space["frames"]:
            self.observation_space["frames"][name] = gym.spaces.Box(
                0,
                255,
                shape=(224, 224, 3),
                dtype=np.uint8,
            )
            self._base_observation_space["frames"][name] = self.observation_space[
                "frames"
            ][name]
        if "depths" in self.observation_space:
            for name in self.observation_space["depths"]:
                space = gym.spaces.Box(0.0, np.inf, shape=(224, 224), dtype=np.float32)
                self.observation_space["depths"][name] = space
                self._base_observation_space["depths"][name] = space

    def _frames_from(self, reading: dict[str, Any]) -> tuple[dict, dict]:
        """Preserve the full RGB field of view with OpenPI's letterbox resize."""
        from PIL import Image

        frames, depths, display = {}, {}, {}
        for name in self._cameras:
            rgb = np.asarray(reading[name]["frame"])[..., ::-1].copy()
            height, width = rgb.shape[:2]
            ratio = max(width / 224, height / 224)
            size = (int(width / ratio), int(height / ratio))
            origin = ((224 - size[0]) // 2, (224 - size[1]) // 2)
            canvas = Image.new("RGB", (224, 224), 0)
            canvas.paste(
                Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), origin
            )
            frames[name] = np.asarray(canvas).copy()
            display[name] = frames[name][..., ::-1].copy()
            display[f"{name}_full"] = rgb[..., ::-1].copy()
            if "depth" in reading[name]:
                depth = np.asarray(reading[name]["depth"], dtype=np.float32).squeeze()
                canvas_depth = Image.new("F", (224, 224), 0.0)
                canvas_depth.paste(
                    Image.fromarray(depth).resize(size, Image.Resampling.NEAREST),
                    origin,
                )
                depths[name] = np.asarray(canvas_depth).copy()
        self.camera_player.put_frame(display)
        return frames, depths

    def _get_observation(self) -> dict[str, Any]:
        observation = super()._get_observation()
        if self.config.is_dummy:
            return observation
        ee = tcp_to_ee(observation["state"]["tcp_pose"], self._tcp_offset)
        if self.config.reference_frame == "tcp" and self._reference_ee_pose is not None:
            ee = matrix_pose(
                np.linalg.inv(pose_matrix(self._reference_ee_pose)) @ pose_matrix(ee)
            )
        observation["state"]["tcp_pose"] = ee.astype(np.float32)
        # RTDE velocity and wrench retain the training data's base-frame convention.
        return observation

    def get_tcp_pose(self) -> np.ndarray:
        """Read the measured base-frame TCP pose without fetching camera frames."""
        if not self.config.is_dummy:
            measured = self._arm.get_state()
            measured.gripper_position = self._robot_state.gripper_position
            self._robot_state = measured
        return self._robot_state.tcp_pose.copy()

    def get_keyboard_action_transform(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the physical control-frame to UR policy-action conversion."""
        return self.keyboard_action_to_env

    def keyboard_action_to_env(self, action: np.ndarray) -> np.ndarray:
        """Map physical keyboard deltas through control axes into policy units.

        Input translation is in metres and Euler XYZ rotation in radians.
        Gripper opening is binary. Selection and rotation match UR force mode.
        """
        converted = np.asarray(action, dtype=np.float64).copy()
        impedance = self.hardware.impedance
        selected = np.asarray(impedance.get("selection_vector", [1] * 6))
        control_rotation = Rotation.from_euler(
            "xyz", impedance.get("control_frame_euler_deg", [0, 0, 0]), degrees=True
        )
        converted[:6] *= selected
        converted[:3] = control_rotation.apply(converted[:3])
        delta = (
            control_rotation
            * Rotation.from_euler("xyz", converted[3:6])
            * control_rotation.inv()
        )
        if self.config.reference_frame == "tcp":
            if not self.config.is_dummy:
                self._tcp_offset = self._arm.get_tcp_offset()
                measured = self._arm.get_state()
                measured.gripper_position = self._robot_state.gripper_position
                self._robot_state = measured
            ee = tcp_to_ee(self._robot_state.tcp_pose, self._tcp_offset)
            rotation = Rotation.from_quat(ee[3:])
            converted[:3] = rotation.inv().apply(converted[:3])
            delta = rotation.inv() * delta * rotation
        if np.any(self.config.action_scale[:2] <= 0):
            raise ValueError("Expected positive UR action_scale for keyboard control.")
        converted[:3] /= self.config.action_scale[0]
        converted[3:6] = delta.as_euler("xyz") / self.config.action_scale[1]
        return converted

    def step(self, action: np.ndarray) -> tuple:
        """Apply a local EE increment and return state with the latest camera frames."""
        if self._closed:
            raise RuntimeError(
                "Cannot execute a UR action after the environment is closed."
            )
        started = time.monotonic()
        record_trace = self._action_trace_enabled
        interval = (
            None
            if not record_trace or self._last_action_started is None
            else started - self._last_action_started
        )
        if record_trace:
            self._last_action_started = started
        trace: dict[str, Any] | None = {} if record_trace else None
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite UR action values.")
        self._target_interval = None
        if not self.config.is_dummy:
            gripper_position = self._robot_state.gripper_position
            self._robot_state = self._arm.get_state()
            self._robot_state.gripper_position = gripper_position
            self._tcp_offset = self._arm.get_tcp_offset()
            if record_trace:
                trace["tcp_before"] = self._robot_state.tcp_pose.tolist()
                trace["velocity_before"] = self._robot_state.tcp_vel.tolist()
            scaled = action.copy()
            scaled[:3] *= self.config.action_scale[0]
            scaled[3:6] *= self.config.action_scale[1]
            target = self._composer.target(
                self._robot_state.tcp_pose,
                self._tcp_offset,
                scaled,
                reference_frame=self.config.reference_frame,
                hold_enabled=self.config.action_idle_hold_enabled,
                position_threshold=self.config.action_idle_position_threshold,
                rotation_threshold=self.config.action_idle_rotation_threshold,
            )
            self.next_position = self._clip_position_to_safety_box(target)
            if record_trace:
                trace["target_tcp"] = self.next_position.tolist()
            self._move_action(self.next_position)
            self._end_effector_action(action[6:])
        if self.config.is_dummy:
            self._record_target_submission()
        target_interval = self._target_interval
        self._num_steps += 1
        # Match CartesianEnv: action processing shares the period; observation follows.
        wait_started = time.monotonic()
        time.sleep(
            max(0.0, 1.0 / self.config.step_frequency - (wait_started - started))
        )
        paced = time.monotonic() if record_trace else 0.0
        pace_wait = paced - wait_started if record_trace else 0.0
        refresh_cameras = (
            not self._camera_refresh_on_request and self._camera_steps_remaining <= 1
        )
        self._camera_steps_remaining = max(0, self._camera_steps_remaining - 1)
        fresh_observation = not self.config.is_dummy
        if fresh_observation:
            self._robot_state = self._read_robot(refresh_cameras=refresh_cameras)
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, True)
        terminated = (
            reward == 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        if (
            (terminated or truncated)
            and not self.config.manual_episode_control_only
            and not refresh_cameras
        ):
            self.refresh_camera_observation()
            observation = self._get_observation()
            refresh_cameras = True
        if record_trace:
            trace.update(
                infer_index=self._infer_index,
                action_index=self._chunk_action_index + 1,
                timestamp=time.time(),
                action=action.tolist(),
                action_interval_sec=interval,
                target_hz=self.config.step_frequency,
                control_sec=paced - started,
                control_hz=1.0 / (paced - started),
                control_overrun_sec=max(
                    0.0, (target_interval or 0.0) - 1.0 / self.config.step_frequency
                ),
                command_sec=paced - started - pace_wait,
                pace_wait_sec=pace_wait,
                target_interval_sec=target_interval,
                observation_sec=time.monotonic() - paced,
                fresh_observation=fresh_observation,
                camera_refreshed=refresh_cameras and fresh_observation,
            )
            measurement = current_worker_timing()
            if measurement is not None:
                measurement.add_detail("command", trace["command_sec"])
                measurement.add_detail("pace_wait", trace["pace_wait_sec"])
                measurement.add_detail("observation", trace["observation_sec"])
        self._chunk_action_index += 1
        if record_trace and fresh_observation:
            trace["tcp_after"] = self._robot_state.tcp_pose.tolist()
            trace["velocity_after"] = self._robot_state.tcp_vel.tolist()
        return (
            observation,
            reward * self.config.reward_scale,
            terminated,
            truncated,
            {"ur_action_trace": trace} if record_trace else {},
        )

    def _record_target_submission(self) -> None:
        """Record target publication intervals without pacing the controller."""
        previous = self._last_target_sent
        sent = time.monotonic()
        if self._action_trace_enabled:
            self._target_interval = None if previous is None else sent - previous
        self._last_target_sent = sent

    def _move_action(self, position: np.ndarray) -> None:
        """Compose targets and calculate impedance from the same UR snapshot."""
        if self.config.is_dummy:
            super()._move_action(position)
            return
        self._clear_error()
        self._arm.send_action_from_state(
            {"tcp_pose": position.astype(np.float32)},
            self._robot_state,
            before_send=self._record_target_submission,
        )

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5) -> None:
        """Use the UR moveL reset path, including construction-time homing."""
        del timeout
        self._logger.info("[RESET] Opening gripper and moving UR to reset pose.")
        self._end_effector.open()
        self._gripper_target = 1.0
        self._arm.reset_tcp(
            pose,
            self.config.payload_settle_seconds,
            self.config.ft_zero_settle_seconds,
        )
        self._composer.clear()
        self._tcp_offset = self._arm.get_tcp_offset()
        self._robot_state = self._read_robot()
        self._reference_ee_pose = tcp_to_ee(
            self._robot_state.tcp_pose, self._tcp_offset
        )
        self._logger.info(
            "[RESET] Completed forceModeStop -> servoStop -> moveL -> "
            "configured payload -> zeroFtSensor; "
            "settle_time=%.2f s; cleared action hold and refreshed reference pose.",
            self.config.payload_settle_seconds + self.config.ft_zero_settle_seconds,
        )

    def reset(
        self,
        joint_reset: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return to a sampled TCP pose, without periodic joint resets."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed UR environment.")
        if joint_reset:
            raise ValueError(
                "UR5e task reset uses TCP poses; joint_reset must be False."
            )
        gym.Env.reset(self, seed=seed)
        self._composer.clear()
        self._gripper_target = None
        self._camera_steps_remaining = 0
        self._last_action_started = None
        self._last_target_sent = None
        seed_sampled_spaces(seed, self._base_observation_space)
        if self.config.is_dummy:
            return self._get_observation(), {}
        self._success_hold_counter = 0
        self._arm.reconfigure_compliance_params(self.config.compliance_param)
        self.go_to_rest()
        self._clear_error()
        self._num_steps = 0
        return self.observe(), {}

    def sample_reset_pose(self) -> np.ndarray:
        """Sample base-frame XYZ and XYZW, following VLA-Precision's reset ranges."""
        pose = self.config.reset_ee_pose.copy()
        if self.config.enable_random_reset:
            bounds = self.config.reset_pose_range
            pose[:3] += self.np_random.uniform(-bounds[:3], bounds[:3])
            pose[3:] += np.deg2rad(self.np_random.uniform(-bounds[3:], bounds[3:]))
        return np.concatenate(
            (pose[:3], Rotation.from_euler("xyz", pose[3:]).as_quat())
        )

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Open the gripper, moveL to the reset TCP and zero the force sensor."""
        if joint_reset:
            raise ValueError(
                "UR5e task reset uses TCP poses; joint_reset must be False."
            )
        self._interpolate_move(self.sample_reset_pose())

    def _build_robot(
        self, controller_node_rank: int, camera_node_rank: int | None
    ) -> Robot:
        return UR5eRobot.build(
            config=self.hardware,
            node_rank=(
                None if controller_node_rank == self.node_rank else controller_node_rank
            ),
            worker_rank=self.env_worker_rank,
            env_idx=self.env_idx,
            cameras={info.name: info for info in self._camera_infos},
        )

    @property
    def _ee_interface(self) -> EndEffector | type[EndEffector]:
        """Resolve the UR gripper before connecting, or return its live part."""
        part = getattr(self, "_end_effector", None)
        return (
            part
            if part is not None
            else EndEffector.backend(self.hardware.end_effector_type)
        )

    def _read_robot(self, refresh_cameras: bool = True) -> UR5eRobotState:
        """Read current robot state, optionally refreshing the policy images."""
        if refresh_cameras:
            state = super()._read_robot()
        else:
            state = self._state_from(
                {
                    "arm": self._arm.get_observation(),
                    "end_effector": self._end_effector.get_observation(),
                }
            )
        state.gripper_position /= self._end_effector.max_width
        return state

    def _end_effector_action(self, ee_action: np.ndarray) -> bool:
        """Threshold the predicted opening, preserving measured feedback separately."""
        value = float(ee_action[0] > self.config.gripper_close_threshold)
        if value == self._gripper_target:
            return False
        changed = self._end_effector.command(
            np.array([value * self._end_effector.max_width])
        )
        self._gripper_target = value
        return changed


class URTrainingEpisodeWrapper(gym.Wrapper):
    """Run force control only between the first RL action and episode end."""

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int | None = None,
        ignore_terminations: bool = False,
        timing_enabled: bool = False,
    ) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.ignore_terminations = ignore_terminations
        self._active = False
        self._closed = False
        self._steps = 0
        self._terminal = None
        self._hold_action = None
        self._last_control_report = 0.0
        self._episode = 0
        self._chunk = 0
        self._logger = get_logger()
        self._timing_enabled = timing_enabled
        self.unwrapped._action_trace_enabled = timing_enabled
        self._action_timing = StageTiming(
            "UR action", timing_enabled, self._logger.info
        )
        if not self.unwrapped.config.is_dummy:
            self.unwrapped._arm.set_control_timing_enabled(timing_enabled)

    def _stop(self) -> None:
        self._hold_action = self.unwrapped.stop_motion()
        self._active = False
        self._report_control_timing(force=True)

    def _report_control_timing(self, *, force: bool = False) -> None:
        if not self._timing_enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_control_report < 1.0:
            return
        self._last_control_report = now
        timing = self.unwrapped.drain_control_timing()
        if timing is not None:
            emit_timing_record(
                self._logger.info,
                "UR background",
                episode=self._episode,
                chunk=self._chunk,
                action=self._steps,
                samples=timing["samples"],
                intervals=timing["intervals"],
                active_interval_ms=1000 * timing["active_interval_sec"],
                target_hz=timing["target_hz"],
                measured_hz=timing["measured_hz"],
                work_ms=1000 * timing["cycle_work_sec_mean"],
                force_mode_ms=1000 * timing["force_mode_sec_mean"],
                max_send_interval_ms=(
                    1000 * timing["send_interval_sec_max"]
                    if timing["send_interval_sec_max"] is not None
                    else None
                ),
                work_overruns=timing["work_overruns"],
                held_target_samples=timing["held_target_samples"],
            )
            self._logger.info(
                "[PD] UR background: target_hz=%.2f measured_hz=%s "
                "max_send_interval_ms=%.2f work_overruns=%d samples=%d",
                timing["target_hz"],
                f"{timing['measured_hz']:.2f}"
                if timing["measured_hz"] is not None
                else "n/a",
                1000 * (timing["send_interval_sec_max"] or 0.0),
                timing["work_overruns"],
                timing["samples"],
            )

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Stop before homing and leave force control paused after reset."""
        self._stop()
        self._terminal = None
        self._steps = 0
        self._episode += 1
        self._chunk = 0
        return self.env.reset(**kwargs)

    def begin_action_chunk(self, size: int, infer_index: int | None = None) -> None:
        """Associate action timings with their execution chunk."""
        self._chunk += 1
        self.env.get_wrapper_attr("begin_action_chunk")(size, infer_index)

    def step(self, action: np.ndarray) -> tuple:
        """Start on demand and suppress the remainder of a terminal chunk."""
        if self._terminal is not None:
            obs, _, terminated, truncated, info = copy.deepcopy(self._terminal)
            info["intervene_action"] = self._hold_action.copy()
            info["rlt_policy_switch_event"] = None
            return obs, 0.0, terminated, truncated, info
        try:
            if not self._active:
                self._logger.info(
                    "[EPISODE] First RL action received; starting force control."
                )
                self.unwrapped.start_motion()
                self._active = True
            obs, reward, terminated, truncated, info = self.env.step(action)
            if (
                self.max_episode_steps is not None
                and self._steps + 1 >= self.max_episode_steps
                and not (terminated or truncated)
                and self.unwrapped._camera_steps_remaining > 0
            ):
                refresh_started = time.monotonic()
                obs = self.unwrapped.observe()
                trace = info.get("ur_action_trace")
                if trace is not None:
                    trace["observation_sec"] += time.monotonic() - refresh_started
                    trace["camera_refreshed"] = not self.unwrapped.config.is_dummy
                    trace["tcp_after"] = self.unwrapped._robot_state.tcp_pose.tolist()
                    trace["velocity_after"] = (
                        self.unwrapped._robot_state.tcp_vel.tolist()
                    )
            if self._timing_enabled:
                trace = info.get("ur_action_trace", {})
                if trace:
                    self._action_timing.record(
                        trace["control_sec"] + trace["observation_sec"],
                        context={
                            "episode": self._episode,
                            "chunk": self._chunk,
                            "action": self._steps + 1,
                            "target_hz": trace["target_hz"],
                        },
                        control=trace["control_sec"],
                        command=trace["command_sec"],
                        pace_wait=trace["pace_wait_sec"],
                        observation=trace["observation_sec"],
                        **(
                            {"target_interval": trace["target_interval_sec"]}
                            if trace.get("target_interval_sec") is not None
                            else {}
                        ),
                        **(
                            {"action_interval": trace["action_interval_sec"]}
                            if trace.get("action_interval_sec") is not None
                            else {}
                        ),
                    )
            self._steps += 1
            if self.max_episode_steps is not None:
                truncated = self._steps >= self.max_episode_steps
            if truncated or (terminated and not self.ignore_terminations):
                self._stop()
                self._logger.info(
                    "[EPISODE] Stopped after %d actions: reward=%s, "
                    "terminated=%s, truncated=%s.",
                    self._steps,
                    reward,
                    terminated,
                    truncated,
                )
                self._terminal = copy.deepcopy(
                    (obs, reward, terminated, truncated, info)
                )
            else:
                self._report_control_timing()
            return obs, reward, terminated, truncated, info
        except BaseException:
            self._stop()
            raise

    def close(self) -> None:
        """Stop force control before closing the environment."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stop()
        finally:
            self.env.close()
