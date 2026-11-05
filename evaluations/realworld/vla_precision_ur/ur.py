# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ported from VLA-Precision; see provenance.json for source and changes.

"""VLA-Precision UR hardware execution with RLinf checkpoint loading."""

import argparse
import json
import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import crcmod
import numpy as np
import serial
import yaml
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from pyDHgripper import PGE
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R

from rlinf.utils.ur_logging import publish_latest

from . import image_tools
from .recording import Recorder
from .utils import FpsCounter, validate_single_arm_norm_stats

logging.basicConfig(level=logging.INFO, format="%(message)s")

PAYLOAD_SETTLE_TIME_S = 0.25
FT_ZERO_SETTLE_TIME_S = 0.50


class _ContinuousForceController:
    """Own force sends and acknowledge each target after its first SDK send."""

    def __init__(self, engine):
        self.engine = engine
        self.io_lock = threading.RLock()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._target = None
        self._sequence = 0
        self._applied = -1
        self._samples = self._intervals = self._overruns = self._held = 0
        self._interval_sum = self._interval_max = 0.0
        self._last_sent = self._last_sequence = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def _require_success(result, operation):
        if isinstance(result, (bool, np.bool_)) and not result:
            raise RuntimeError(f"RTDE {operation} returned False")

    def _check(self):
        if self._error is not None:
            raise RuntimeError("Background UR force controller failed") from self._error
        if self._stop.is_set() or not self.running:
            raise RuntimeError("Background UR force controller is not running")

    def start(self):
        if self._error is not None:
            raise RuntimeError("Background UR force controller failed") from self._error
        if self.running:
            return
        with self.io_lock:
            self._target = np.asarray(
                self.engine.rtde_r.getActualTCPPose(), dtype=float
            ).copy()
        self._sequence, self._applied = 0, -1
        self._last_sent = self._last_sequence = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="VLA-Precision-force-mode", daemon=True
        )
        self._thread.start()
        self._wait_applied(0)

    def _wait_applied(self, sequence):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._applied >= sequence
                    or self._error is not None
                    or self._stop.is_set()
                ),
                timeout=2.0,
            )
            self._check()
            if not ready or self._applied < sequence:
                self._stop.set()
                raise RuntimeError("Background UR force target send timed out")

    def update_target(self, target):
        with self._condition:
            self._check()
            self._target = None if target is None else np.asarray(target).copy()
            self._sequence += 1
            sequence = self._sequence
        self._wait_applied(sequence)

    def _run(self):
        engine = self.engine
        period = 1.0 / engine.rtde_freq
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                with self._condition:
                    target = None if self._target is None else self._target.copy()
                    sequence = self._sequence
                with self.io_lock:
                    pose = np.asarray(engine.rtde_r.getActualTCPPose(), dtype=float)
                    velocity = np.asarray(
                        engine.rtde_r.getActualTCPSpeed(), dtype=float
                    )
                    wrench = (
                        np.zeros(6)
                        if target is None
                        else engine._calculate_force(target, pose, velocity)
                    )
                    self._require_success(
                        engine.rtde_c.forceMode(
                            engine.task_frame,
                            [0] * 6 if target is None else engine.select_vector,
                            wrench,
                            engine.force_type,
                            engine.force_limit,
                        ),
                        "forceMode",
                    )
                    sent = time.perf_counter()
                with self._condition:
                    self._samples += 1
                    self._overruns += sent - started > period
                    self._held += sequence == self._last_sequence
                    if self._last_sent is not None:
                        interval = sent - self._last_sent
                        self._intervals += 1
                        self._interval_sum += interval
                        self._interval_max = max(self._interval_max, interval)
                    self._last_sent, self._last_sequence = sent, sequence
                    self._applied = sequence
                    self._condition.notify_all()
                self._stop.wait(max(0.0, period - (time.perf_counter() - started)))
        except BaseException as error:
            with self._condition:
                self._error = error
                self._stop.set()
                self._condition.notify_all()
        finally:
            try:
                with self.io_lock:
                    self._require_success(
                        engine.rtde_c.forceModeStop(), "forceModeStop"
                    )
            except BaseException as error:
                with self._condition:
                    self._error = self._error or error
                    self._condition.notify_all()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self.running:
                raise RuntimeError("Background UR force controller did not stop")
        if self._error is not None:
            raise RuntimeError("Background UR force controller failed") from self._error

    def drain_timing(self):
        with self._condition:
            if not self._samples:
                return None
            timing = {
                "target_hz": self.engine.rtde_freq,
                "samples": self._samples,
                "measured_hz": self._intervals / self._interval_sum
                if self._interval_sum > 0
                else None,
                "max_send_interval_ms": self._interval_max * 1000,
                "work_overruns": self._overruns,
                "held_target_samples": self._held,
            }
            self._samples = self._intervals = self._overruns = self._held = 0
            self._interval_sum = self._interval_max = 0.0
            return timing


def update_latest_symlink(target: Path, link_name: Path):
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)


def resolve_model_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def is_none_config(value) -> bool:
    return value is None or (
        isinstance(value, str) and value.lower() in ("none", "null")
    )


class Inference:
    @staticmethod
    def _resolve_model_path(value):
        return resolve_model_path(value)

    @classmethod
    def _resolve_actor_checkpoint_dir(cls, value):
        path = cls._resolve_model_path(value)
        if path is None:
            return None
        if (path / "params").exists():
            return path
        if not path.exists():
            return path

        candidates = []
        for child in path.iterdir():
            if child.is_dir() and child.name.isdigit() and (child / "params").exists():
                candidates.append((int(child.name), child))
        if not candidates:
            return path
        step, latest = max(candidates, key=lambda item: item[0])
        logging.info(
            "[MODEL] Resolved actor checkpoint root %s to latest step %s: %s",
            path,
            step,
            latest,
        )
        return latest

    def __init__(self, config_path: Path, *, inference_stage: str | None = None):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if inference_stage is not None:
            cfg["inference_stage"] = inference_stage

        # RLinf checkpoint adaptation; hardware execution below is VLA-Precision's.
        from .policy import load_norm_stats, model_config_from_yaml

        robot = cfg["robot"]
        self.remote_policy = False
        self.model_config = model_config_from_yaml(cfg)
        from rlinf.algorithms.rlt.inference import CriticalPhaseSwitch, is_stage2

        self.stage2_enabled = is_stage2(cfg)
        self._critical_switch = CriticalPhaseSwitch()
        self.action_chunk = int(self.model_config.num_action_chunks)
        self.checkpoint_dir = Path(self.model_config.model_path).expanduser()
        self.sample_steps = int(self.model_config.num_steps)
        self.norm_stats_path = Path(
            self.model_config.openpi_data.norm_stats_path
        ).expanduser()
        self.norm_stats = load_norm_stats(self.norm_stats_path)
        from rlinf.utils.keyboard_focus import validate_terminal_keyboard

        # This backend already reads all episode controls from terminal stdin.
        validate_terminal_keyboard(cfg.get("keyboard_terminal_only", False))
        self._native_config = cfg

        # Camera config
        cam = cfg["cameras"]
        self.wrist_cam_serial = cam["wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)
        self.cam_width = cam.get("width", 640)
        self.cam_height = cam.get("height", 480)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]
        record = cfg.get("record", {})
        self.num_episodes = record.get("num_episodes", 10)
        self.ep_timeout = record.get("episode_timeout_sec", 30)
        self.action_timing_enabled = bool(record.get("action_timing_enabled", False))
        self.save_logs = bool(record.get("save_logs", True))

        # Robot config
        self.robot_ip = robot["ip"]
        self.gripper_port = robot["gripper_port"]
        self.gripper_reverse = robot["gripper_reverse"]
        self.init_gripper = robot.get("init_gripper", True)
        self.fix_gripper = bool(robot.get("fix_gripper", False))
        if not self.remote_policy:
            validate_single_arm_norm_stats(
                self.norm_stats,
                fixed_gripper=self.fix_gripper,
                learned_state_dim=19,
                path=self.norm_stats_path,
                padded_action_dim=int(self.model_config.openpi.model_action_dim),
            )
        self.close_threshold = (
            cfg["stage2"]["gripper_close_threshold"]
            if self.stage2_enabled
            else robot.get("close_threshold", 0.9)
        )
        self.target_reached_gripper_open_threshold = robot.get(
            "target_reached_gripper_open_threshold",
            self.close_threshold,
        )
        self._gripper_force = robot.get("gripper_force", 70)
        self._gripper_speed = robot.get("gripper_speed", 60)
        self.pre_reset_tcp_pose = robot.get("pre_reset_tcp_pose")
        self.init_tcp_pose = robot["init_tcp_pose"]
        self.init_pose_range = robot.get(
            "init_pose_range",
            robot.get("init_pos_range", [0.03, 0.03, 0.03, 0.0, 0.0, 20.0]),
        )
        self.target_tcp_pose = robot.get("target_tcp_pose")
        self.target_threshold = robot.get("target_threshold")
        has_target_pose = not is_none_config(self.target_tcp_pose)
        has_target_threshold = not is_none_config(self.target_threshold)
        if has_target_pose != has_target_threshold:
            raise ValueError(
                "robot.target_tcp_pose and robot.target_threshold must both be set or both be null."
            )
        self.use_target_success = has_target_pose and has_target_threshold
        self.show_target_error = robot.get("show_target_error", False)
        self.action_fps = robot["action_fps"]
        self.debug = robot.get("debug", False)
        self.control_space = robot.get("control_space", "position")
        self.reference_frame = robot.get("reference_frame", "tcp")
        self.filter_zero_output = robot.get("filter_zero_output", True)
        self.action_idle_hold_enabled = bool(
            robot.get(
                "action_idle_hold_enabled", robot.get("ACTION_IDLE_HOLD_ENABLED", True)
            )
        )
        legacy_idle_threshold = robot.get(
            "action_idle_hold_threshold", robot.get("ACTION_IDLE_HOLD_THRESHOLD")
        )
        self.action_idle_position_threshold = float(
            robot.get(
                "action_idle_position_threshold",
                robot.get(
                    "ACTION_IDLE_POSITION_THRESHOLD",
                    0.0001 if legacy_idle_threshold is None else legacy_idle_threshold,
                ),
            )
        )
        self.action_idle_rotation_threshold = float(
            robot.get(
                "action_idle_rotation_threshold",
                robot.get(
                    "ACTION_IDLE_ROTATION_THRESHOLD",
                    0.001 if legacy_idle_threshold is None else legacy_idle_threshold,
                ),
            )
        )
        self.payload_mass = robot.get("payload", {}).get("mass", 1.601)
        self.payload_cog = robot.get("payload", {}).get("cog", [0.011, -0.002, 0.052])

        position_cfg = robot.get("position_mode", {})
        self._velocity = position_cfg.get("speed", 0.5)
        self._acceleration = position_cfg.get("acceleration", 0.5)
        self._servo_time = position_cfg.get("servo_time", 1.0 / self.action_fps)
        self._lookahead_time = position_cfg.get("lookahead_time", 0.1)
        self._gain = position_cfg.get("gain", 300)

        force_cfg = robot.get("force_mode", {})
        self.continuous_tracking = force_cfg.get("continuous_tracking", False)
        if not isinstance(self.continuous_tracking, bool):
            raise ValueError(
                "robot.force_mode.continuous_tracking must be a YAML boolean"
            )
        if self.continuous_tracking and self.control_space != "force":
            raise ValueError("continuous_tracking requires robot.control_space: force")
        self.kp = force_cfg.get("kp", 2000)
        self.kd = force_cfg.get("kd", 200)
        self.kp_rot = force_cfg.get("kp_rot", 32.0)
        self.kd_rot = force_cfg.get("kd_rot", 6.4)
        self.rtde_freq = force_cfg.get("rtde_freq", 125)
        if not np.isfinite(self.rtde_freq) or self.rtde_freq <= 0:
            raise ValueError("robot.force_mode.rtde_freq must be positive and finite")
        self.select_vector = force_cfg.get("select_vector", [1, 1, 1, 1, 1, 1])
        self.force_limit = force_cfg.get("force_limit", [2, 2, 2, 2, 2, 2])
        self.pos_delta = force_cfg.get("pos_delta", 0.2)
        self.vel_delta = force_cfg.get("vel_delta", 0.2)
        self.gain_scale = force_cfg.get("gain_scale", 1.5)
        self.control_frame_euler_deg = self._validate_control_frame_euler_deg(
            force_cfg.get("control_frame_euler_deg", [0.0, 0.0, 0.0])
        )
        self.control_to_base_rotation = R.from_euler(
            "xyz", self.control_frame_euler_deg, degrees=True
        ).as_matrix()
        self.use_control_frame = not np.allclose(
            self.control_frame_euler_deg, [0.0, 0.0, 0.0]
        )
        self.task_frame = self._make_force_task_frame()
        self.force_type = 2
        # Task config
        task = cfg["task"]
        self.task_description = task["description"]

        # time stamps
        time_str = time.strftime("%Y%m%d-%H%M%S")
        time_path = time.strftime("%Y%m%d")

        # base dir
        base_dir = (
            Path(cfg.get("output_root", "./results")).expanduser()
            / cfg["experiment"]["name"]
            / "openpi-native"
        )
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        # create dir
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # log paths
        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"
        self._timing_path = log_dir / "all_logs" / f"timing_{time_str}.jsonl"

        # video paths
        wrist_video = video_dir / f"wrist_{time_str}.mp4"
        exterior_video = video_dir / f"exterior_{time_str}.mp4"

        # Recorder
        self.recorder = Recorder(
            log_path=log_path,
            video_path=[wrist_video, exterior_video],
            display_fps=self.video_fps,
            visualize=self.visualize,
            save_logs=self.save_logs,
            evaluation_path=(
                Path(cfg.get("output_root", "./results")).expanduser()
                / "runs"
                / f"{time_str}-ur5e-eval"
                / "ur5e-evaluation.json"
            ),
        )

        # create symlink to latest log
        if self.save_logs:
            update_latest_symlink(log_path, latest_path)
            publish_latest(self._timing_path, log_dir / "latest.jsonl")

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.rtde_r = None
        self.rtde_c = None
        self.cameras = None
        self.gripper = None
        self._closed = False
        self._force_controller = None
        self._last_force_report = 0.0
        self._gripper_reader_stop = threading.Event()
        self._gripper_reader_thread = None
        self._last_gripper_position = 1
        self._gripper_position = 1
        self._last_servoj_ts = None
        self._zero_force_lock_active = False
        self._action_idle_hold_pose = None
        self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)
        self._episode_reference_ee_pose = None
        self._ep_start = None
        self._ep_idx = 1
        self._ep_steps = 0
        self._ep_done = False
        self._success_episodes = 0
        self._failed_episodes = 0
        self._reset_requested = threading.Event()
        self._success_requested = threading.Event()
        self._keyboard_listener_stop = threading.Event()
        self._keyboard_listener_thread = None
        self._episode_time_printer_stop = None
        self._episode_time_printer_thread = None

        from rlinf.utils.robot_lifecycle import register_robot_cleanup

        register_robot_cleanup(self.close)

    def close(self):
        """Stop motion and release every connection even when another close fails."""
        if getattr(self, "_closed", False):
            return
        controller = getattr(self, "_force_controller", None)
        if controller is not None:
            try:
                controller.stop()
            except BaseException:
                if controller.running:
                    raise
                logging.exception(
                    "[EXIT] Background force control stopped with an error."
                )
            self._report_force_timing(force=True)
        self._closed = True
        from rlinf.utils.robot_lifecycle import unregister_robot_cleanup

        control = getattr(self, "rtde_c", None)
        receive = getattr(self, "rtde_r", None)
        gripper = getattr(self, "gripper", None)
        stop = getattr(self, "_gripper_reader_stop", None)
        if stop is not None:
            stop.set()
        callbacks = []
        if control is not None:
            for name in ("forceModeStop", "servoStop", "stopScript", "disconnect"):
                callback = getattr(control, name, None)
                if callback is not None:
                    callbacks.append(callback)
        if receive is not None:
            callbacks.append(receive.disconnect)
        thread = getattr(self, "_gripper_reader_thread", None)
        if thread is not None:
            callbacks.append(lambda: thread.join(timeout=1.0))
        serial_port = getattr(gripper, "ser", None)
        if serial_port is not None:
            callbacks.append(serial_port.close)
        for camera in (getattr(self, "cameras", None) or {}).values():
            callbacks.append(camera.disconnect)
        if hasattr(self, "_keyboard_listener_stop"):
            callbacks.append(self._stop_keyboard_episode_control_listener)
        if hasattr(self, "_episode_time_printer_stop"):
            callbacks.append(self._stop_episode_time_printer)
        for callback in callbacks:
            try:
                callback()
            except BaseException:
                logging.exception("[EXIT] Failed to release an inference resource.")
        self.rtde_c = self.rtde_r = self.gripper = self.cameras = None
        unregister_robot_cleanup(self.close)

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to UR5e robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to UR5e robot =====")
            self.rtde_r = RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = RTDEControlInterface(self.robot_ip)
            self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
            if self.control_space == "force":
                self.rtde_c.forceModeSetGainScaling(self.gain_scale)
                logging.info(
                    "[ROBOT] control frame Euler xyz(deg)=%s, task_frame=%s, select_vector=%s",
                    self.control_frame_euler_deg.tolist(),
                    np.round(self.task_frame, 6).tolist(),
                    self.select_vector,
                )

            # Joint positions
            joints = self.rtde_r.getActualQ()
            if joints and len(joints) == 6:
                formatted = [round(j, 4) for j in joints]
                logging.info(f"[ROBOT] Current joint positions: {formatted}")
            else:
                logging.info("[ERROR] Failed to read joint positions.")

            # TCP pose
            tcp_pose = self.rtde_r.getActualTCPPose()
            if tcp_pose and len(tcp_pose) == 6:
                formatted_pose = [round(p, 4) for p in tcp_pose]
                logging.info(f"[ROBOT] Current TCP pose: {formatted_pose}")
                logging.info(
                    f"[ROBOT] Translation (m): x={formatted_pose[0]}, y={formatted_pose[1]}, z={formatted_pose[2]}"
                )
                logging.info(
                    f"[ROBOT] Rotation (rad): rx={formatted_pose[3]}, ry={formatted_pose[4]}, rz={formatted_pose[5]}"
                )
                logging.info("===== [ROBOT] UR5e initialized successfully =====\n")
            else:
                logging.info("[ERROR] Failed to read TCP pose.")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to UR5e robot =====")
            logging.error(f"Exception: {e}\n")
            exit(1)

    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")

            wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.wrist_cam_serial,
                fps=self.cam_fps,
                width=self.cam_width,
                height=self.cam_height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.exterior_cam_serial,
                fps=self.cam_fps,
                width=self.cam_width,
                height=self.cam_height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            camera_config = {"wrist_image": wrist_cfg, "exterior_image": exterior_cfg}
            self.cameras = make_cameras_from_configs(camera_config)

            for name, cam in self.cameras.items():
                cam.connect()
                logging.info(f"[CAM] {name} connected successfully.")

            logging.info("===== [CAM] Cameras initialized successfully =====\n")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.close()
            raise

    # --------------------------- GRIPPER --------------------------- #
    def _create_gripper(self):
        if self.init_gripper:
            return PGE(port=self.gripper_port)

        gripper = PGE.__new__(PGE)
        gripper.ser = serial.Serial(port=self.gripper_port, baudrate=115200)
        gripper.crc16 = crcmod.mkCrcFun(
            0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000
        )
        return gripper

    def connect_gripper(self):
        """Initialize and connect the DH gripper."""
        if self.fix_gripper:
            self.gripper = None
            logging.info("[GRIPPER] fix_gripper enabled; skipping gripper connection.")
            return
        try:
            logging.info("\n===== [GRIPPER] Initializing DH Gripper =====")
            self.gripper = self._create_gripper()
            if self.init_gripper:
                self.gripper.init_feedback()
            else:
                logging.info(
                    "[GRIPPER] Skipped init_state and init_feedback by config."
                )
            self.gripper.set_force(self._gripper_force)
            self.gripper.set_vel(self._gripper_speed)
            logging.info(
                f"[GRIPPER] Force: {self._gripper_force}, speed: {self._gripper_speed}"
            )
            # Start gripper state reader
            self._start_gripper_state_reader()
            logging.info("[GRIPPER] DH Gripper initialized successfully.")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize DH Gripper.")
            logging.error(f"Exception: {e}\n")
            self.close()
            raise

    # --------------------------- GRIPPER THREAD --------------------------- #
    def _start_gripper_state_reader(self):
        self._gripper_reader_stop.clear()
        self._gripper_reader_thread = threading.Thread(
            target=self._read_gripper_state, daemon=True
        )
        self._gripper_reader_thread.start()

    # --------------------------- GRIPPER WAIT --------------------------- #
    def wait_for_gripper_states(self):
        if self.fix_gripper:
            return
        if hasattr(self.gripper, "position"):
            while self.gripper.position is None:
                logging.info("[GRIPPER] Waiting for gripper state to be obtained...")
                time.sleep(0.1)
        else:
            while not hasattr(self.gripper, "position"):
                logging.info("[GRIPPER] Waiting for gripper position to be set...")
                time.sleep(0.1)

    # --------------------------- GRIPPER STATE --------------------------- #
    def _read_gripper_state(self):
        self.gripper.position = None
        while not self._gripper_reader_stop.is_set():
            gripper_position = (
                0.0 if self._gripper_position <= self.close_threshold else 1.0
            )

            if self.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self._last_gripper_position:
                self.gripper.set_pos(val=int(1000 * gripper_position), blocking=False)
                self._last_gripper_position = gripper_position

            gripper_pos = self.gripper.read_pos() / 1000.0
            if self.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self.gripper.position = gripper_pos
            self._gripper_reader_stop.wait(0.01)

    def _open_gripper_before_reset(self):
        if not self.init_gripper or getattr(self, "gripper", None) is None:
            return
        self._gripper_position = 1.0
        self._last_gripper_position = None
        logging.info(
            "[GRIPPER] init_gripper enabled; opening gripper before episode reset."
        )

    # --------------------------- TCP POSE UTILS --------------------------- #
    def _pose_to_transform(self, pose: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    def _transform_to_pose(self, transform: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                transform[:3, 3],
                R.from_matrix(transform[:3, :3]).as_rotvec(),
            )
        )

    def _get_current_tcp_offset(self) -> np.ndarray:
        with self._robot_io():
            return np.asarray(self.rtde_c.getTCPOffset(), dtype=float)

    def _robot_io(self):
        controller = getattr(self, "_force_controller", None)
        return controller.io_lock if controller is not None else nullcontext()

    def tcp_to_ee_pose(
        self, tcp_pose: np.ndarray, tcp_offset: np.ndarray
    ) -> np.ndarray:
        tcp_transform = self._pose_to_transform(np.asarray(tcp_pose, dtype=float))
        offset_transform = self._pose_to_transform(np.asarray(tcp_offset, dtype=float))
        ee_transform = tcp_transform @ np.linalg.inv(offset_transform)
        return self._transform_to_pose(ee_transform)

    def ee_to_tcp_pose(self, ee_pose: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
        ee_transform = self._pose_to_transform(np.asarray(ee_pose, dtype=float))
        offset_transform = self._pose_to_transform(np.asarray(tcp_offset, dtype=float))
        tcp_transform = ee_transform @ offset_transform
        return self._transform_to_pose(tcp_transform)

    def get_ee_pose(self) -> np.ndarray:
        with self._robot_io():
            tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
            tcp_offset = self._get_current_tcp_offset()
        return self.tcp_to_ee_pose(tcp_pose, tcp_offset)

    def _pose_euler(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        return np.concatenate((pose[:3], R.from_rotvec(pose[3:]).as_euler("xyz")))

    def _relative_pose_euler(self, pose: np.ndarray) -> np.ndarray:
        if self._episode_reference_ee_pose is None:
            raise RuntimeError("Episode reference EE pose is not set.")
        reference_transform = self._pose_to_transform(self._episode_reference_ee_pose)
        current_transform = self._pose_to_transform(np.asarray(pose, dtype=float))
        relative_transform = np.linalg.inv(reference_transform) @ current_transform
        return np.concatenate(
            (
                relative_transform[:3, 3],
                R.from_matrix(relative_transform[:3, :3]).as_euler("xyz"),
            )
        )

    def set_episode_reference_pose(self):
        self._episode_reference_ee_pose = self.get_ee_pose()
        logging.info(
            f"[STATE] Set episode reference EE pose: {np.round(self._episode_reference_ee_pose, 6).tolist()}"
        )

    def _validate_tcp_pose(
        self, name: str, pose: list[float] | np.ndarray
    ) -> np.ndarray:
        tcp_pose = np.asarray(pose, dtype=float)
        if tcp_pose.shape != (6,):
            raise ValueError(
                f"robot.{name} must contain 6 values, got {tcp_pose.shape}."
            )
        tcp_pose = tcp_pose.copy()
        tcp_pose[3:] = R.from_euler("xyz", tcp_pose[3:]).as_rotvec()
        return tcp_pose

    def _is_none_pose(self, pose) -> bool:
        return is_none_config(pose)

    def _sample_init_tcp_pose(self) -> np.ndarray:
        init_pose = self._validate_tcp_pose("init_tcp_pose", self.init_tcp_pose)
        random_range = np.abs(np.asarray(self.init_pose_range, dtype=float))
        if random_range.shape == (3,):
            random_range = np.concatenate((random_range, [0.0, 0.0, 20.0]))
        elif random_range.shape != (6,):
            raise ValueError(
                f"robot.init_pose_range must contain 3 or 6 values, got {random_range.shape}."
            )
        target_pose = init_pose.copy()
        target_pose[:3] += np.random.uniform(-random_range[:3], random_range[:3])

        init_euler = R.from_rotvec(target_pose[3:]).as_euler("xyz", degrees=True)
        delta_euler_deg = np.random.uniform(-random_range[3:], random_range[3:])
        target_pose[3:] = R.from_euler(
            "xyz", init_euler + delta_euler_deg, degrees=True
        ).as_rotvec()
        return target_pose

    def move_to_tcp_pose(self, tcp_pose: np.ndarray):
        tcp_pose = np.asarray(tcp_pose, dtype=float)
        self._clear_action_idle_hold()
        if not self.debug:
            self.stop_force()
            self.rtde_c.servoStop()
            self.rtde_c.moveL(tcp_pose.tolist(), self._velocity, self._acceleration)

    def _reapply_payload_and_zero_ft_sensor(self):
        """Reapply gravity compensation, then zero the F/T sensor."""
        payload_result = self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
        if isinstance(payload_result, (bool, np.bool_)) and not bool(payload_result):
            raise RuntimeError("RTDE setPayload returned False after episode reset")
        time.sleep(PAYLOAD_SETTLE_TIME_S)
        zero_result = self.rtde_c.zeroFtSensor()
        if isinstance(zero_result, (bool, np.bool_)) and not bool(zero_result):
            raise RuntimeError("RTDE zeroFtSensor returned False after episode reset")
        time.sleep(FT_ZERO_SETTLE_TIME_S)
        logging.info(
            "[RESET] Reapplied payload and zeroed F/T sensor: payload_mass=%s "
            "payload_cog=%s settle_time=%.2f s",
            self.payload_mass,
            self.payload_cog,
            PAYLOAD_SETTLE_TIME_S + FT_ZERO_SETTLE_TIME_S,
        )

    def reset_episode(self):
        if getattr(self, "_critical_switch", None) is not None:
            self._critical_switch.reset()
        if getattr(self, "continuous_tracking", False):
            self.stop_force()
        self._clear_action_idle_hold()
        self._open_gripper_before_reset()

        if self._is_none_pose(self.pre_reset_tcp_pose):
            logging.info(
                "[RESET] pre_reset_tcp_pose is None; skipping pre-reset TCP pose."
            )
        else:
            logging.info("[RESET] Moving to pre-reset TCP pose.")
            self.move_to_tcp_pose(
                self._validate_tcp_pose("pre_reset_tcp_pose", self.pre_reset_tcp_pose)
            )

        logging.info("[RESET] Moving to randomized initial TCP pose.")
        self.move_to_tcp_pose(self._sample_init_tcp_pose())
        self._reapply_payload_and_zero_ft_sensor()
        self.set_episode_reference_pose()

    def _has_success_gripper_state(self) -> bool:
        if self.fix_gripper:
            return True
        if not self.init_gripper:
            return True

        gripper_position = getattr(getattr(self, "gripper", None), "position", None)
        if gripper_position is None:
            if self.show_target_error:
                logging.info(
                    "[STATE] Target pose reached, waiting for gripper position."
                )
            return False

        gripper_position = float(gripper_position)
        gripper_open = gripper_position > self.target_reached_gripper_open_threshold
        if not gripper_open:
            logging.info(
                "[STATE] Target pose reached, but gripper is not open yet: "
                f"position={gripper_position:.3f}, threshold>{self.target_reached_gripper_open_threshold:.3f}"
            )
        return gripper_open

    def has_reached_target(self) -> bool:
        target_pose = self._validate_tcp_pose("target_tcp_pose", self.target_tcp_pose)
        threshold = np.asarray(self.target_threshold, dtype=float)
        if threshold.size == 1:
            threshold = np.repeat(threshold.item(), 6)
        if threshold.shape != (6,):
            raise ValueError(
                f"robot.target_threshold must contain 1 or 6 values, got {threshold.shape}."
            )

        current_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        position_error = current_pose[:3] - target_pose[:3]
        rotation_error = (
            R.from_rotvec(target_pose[3:]) * R.from_rotvec(current_pose[3:]).inv()
        ).as_euler("xyz")
        pose_error = np.abs(np.concatenate((position_error, rotation_error)))
        if self.show_target_error:
            logging.info(
                f"[STATE] Target error [x, y, z, rx, ry, rz]: {np.round(pose_error, 5).tolist()}"
            )
        reached = bool(np.all(pose_error <= threshold))
        if reached and not self._has_success_gripper_state():
            return False
        if reached:
            logging.info(
                f"[STATE] Target reached, pose error: {np.round(pose_error, 5).tolist()}"
            )
        return reached

    def _action_motion_mask(self, action: np.ndarray) -> np.ndarray:
        """Return per-axis translation activity and one shared rotation state."""
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")
        if not self.action_idle_hold_enabled:
            return np.ones(6, dtype=bool)
        position_threshold = float(self.action_idle_position_threshold)
        rotation_threshold = float(self.action_idle_rotation_threshold)
        translation_mask = (
            np.ones(3, dtype=bool)
            if position_threshold < 0
            else np.abs(action[:3]) > position_threshold
        )
        rotation_active = (
            True
            if rotation_threshold < 0
            else np.linalg.norm(action[3:6]) > rotation_threshold
        )
        return np.concatenate((translation_mask, np.repeat(rotation_active, 3)))

    @staticmethod
    def _base_motion_delta_from_action(
        action: np.ndarray,
        current_rotation: np.ndarray,
        reference_frame: str,
    ) -> np.ndarray:
        """Express a 6D Cartesian action in the base frame used by hold anchors."""
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")
        current_rot = R.from_matrix(np.asarray(current_rotation, dtype=float))
        delta_rot = R.from_euler("xyz", action[3:6])
        if reference_frame == "base":
            delta_pos_base = action[:3]
            delta_rot_base = delta_rot
        elif reference_frame == "tcp":
            delta_pos_base = current_rot.apply(action[:3])
            delta_rot_base = current_rot * delta_rot * current_rot.inv()
        else:
            raise ValueError(f"Unsupported reference_frame: {reference_frame}")
        return np.concatenate((delta_pos_base, delta_rot_base.as_rotvec()))

    def _clear_action_idle_hold(self):
        self._action_idle_hold_pose = None
        self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)

    def _target_pose_from_action(
        self, action: np.ndarray, current_tcp_pose: np.ndarray | None = None
    ) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")

        if current_tcp_pose is None:
            current_tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        else:
            current_tcp_pose = np.asarray(current_tcp_pose, dtype=float)
        if current_tcp_pose.shape != (6,):
            raise ValueError(
                f"current TCP pose must contain 6 values, got {current_tcp_pose.shape}"
            )

        tcp_offset = self._get_current_tcp_offset()
        current_ee_pose = self.tcp_to_ee_pose(current_tcp_pose, tcp_offset)
        current_position = current_ee_pose[:3]
        current_rotation = R.from_rotvec(current_ee_pose[3:]).as_matrix()

        delta_position = np.asarray(action[:3], dtype=float)
        delta_rotation = R.from_euler(
            "xyz", np.asarray(action[3:6], dtype=float)
        ).as_matrix()

        if self.reference_frame == "base":
            target_position = current_position + delta_position
            target_rotation = delta_rotation @ current_rotation
        elif self.reference_frame == "tcp":
            target_position = current_position + current_rotation @ delta_position
            target_rotation = current_rotation @ delta_rotation
        else:
            raise ValueError(f"Unsupported reference_frame: {self.reference_frame}")

        target_ee_transform = np.eye(4)
        target_ee_transform[:3, :3] = target_rotation
        target_ee_transform[:3, 3] = target_position
        target_ee_pose = self._transform_to_pose(target_ee_transform)
        action_target_pose = self.ee_to_tcp_pose(target_ee_pose, tcp_offset)

        if not self.action_idle_hold_enabled or (
            self.action_idle_position_threshold < 0
            and self.action_idle_rotation_threshold < 0
        ):
            self._clear_action_idle_hold()
            return action_target_pose

        if self._action_idle_hold_pose is None:
            self._action_idle_hold_pose = current_tcp_pose.copy()

        base_motion_delta = self._base_motion_delta_from_action(
            action,
            current_rotation,
            self.reference_frame,
        )
        translation_mask = (
            np.ones(3, dtype=bool)
            if self.action_idle_position_threshold < 0
            else np.abs(base_motion_delta[:3]) > self.action_idle_position_threshold
        )
        rotation_active = (
            True
            if self.action_idle_rotation_threshold < 0
            else np.linalg.norm(base_motion_delta[3:])
            > self.action_idle_rotation_threshold
        )
        motion_mask = np.concatenate((translation_mask, np.repeat(rotation_active, 3)))
        stopped_mask = self._action_idle_prev_motion_mask & ~motion_mask
        self._action_idle_hold_pose[stopped_mask] = current_tcp_pose[stopped_mask]

        target_pose = self._action_idle_hold_pose.copy()
        target_pose[motion_mask] = action_target_pose[motion_mask]
        # While an axis is moving, keep its future hold baseline aligned with
        # the actual pose. On the first idle sample it is latched once above.
        self._action_idle_hold_pose[motion_mask] = current_tcp_pose[motion_mask]
        self._action_idle_prev_motion_mask = motion_mask.copy()
        return target_pose

    def _calculate_force(
        self, target_pose: np.ndarray, current_pose: np.ndarray, current_vel: np.ndarray
    ) -> np.ndarray:
        diff_p = np.clip(
            target_pose[:3] - current_pose[:3], -self.pos_delta, self.pos_delta
        )
        diff_d = np.clip(-current_vel[:3], -self.vel_delta, self.vel_delta)
        force_pos = self.kp * diff_p + self.kd * diff_d

        target_rot = R.from_rotvec(target_pose[3:]).as_matrix()
        current_rot = R.from_rotvec(current_pose[3:]).as_matrix()
        rot_err = R.from_matrix(target_rot @ current_rot.T).as_rotvec()
        torque = self.kp_rot * rot_err - self.kd_rot * current_vel[3:]
        return self._wrench_base_to_task(np.concatenate((force_pos, torque)))

    @staticmethod
    def _validate_control_frame_euler_deg(euler_deg) -> np.ndarray:
        euler_deg = np.asarray(euler_deg, dtype=float)
        if euler_deg.shape != (3,) or not np.all(np.isfinite(euler_deg)):
            raise ValueError(
                "robot.force_mode.control_frame_euler_deg must contain three finite degree values."
            )
        return euler_deg

    def _make_force_task_frame(self) -> list[float]:
        if not self.use_control_frame:
            return [0.0] * 6
        rotvec = R.from_euler(
            "xyz", self.control_frame_euler_deg, degrees=True
        ).as_rotvec()
        return [0.0, 0.0, 0.0, *rotvec.tolist()]

    def _wrench_base_to_task(self, wrench_base: np.ndarray) -> np.ndarray:
        wrench_base = np.asarray(wrench_base, dtype=float)
        if wrench_base.shape != (6,):
            raise ValueError(f"Base-frame wrench must be 6D, got {wrench_base.shape}.")
        if not self.use_control_frame:
            return wrench_base
        base_to_control = self.control_to_base_rotation.T
        return np.concatenate(
            (
                base_to_control @ wrench_base[:3],
                base_to_control @ wrench_base[3:],
            )
        )

    def _is_zero_output_action(self, action: np.ndarray) -> bool:
        return (
            self.filter_zero_output
            and np.any(self._action_motion_mask(action))
            and np.allclose(action[:6], 0.0, atol=1e-3)
        )

    def _calculate_ft_target(
        self, action: np.ndarray, timing: dict[str, float] | None = None
    ) -> np.ndarray:
        started = time.perf_counter()
        current_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        current_vel = np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=float)
        state_read = time.perf_counter()
        target_pose = self._target_pose_from_action(
            action, current_tcp_pose=current_pose
        )
        target_ready = time.perf_counter()
        wrench = self._calculate_force(target_pose, current_pose, current_vel)
        if timing is not None:
            timing["state_read"] = state_read - started
            timing["target"] = target_ready - state_read
            timing["impedance"] = time.perf_counter() - target_ready
        return wrench

    def _pace_action_target(self):
        """Space target submissions without catching up after an overrun."""
        if not hasattr(self, "_last_action_target"):
            return
        started = time.perf_counter()
        previous = self._last_action_target
        self._action_preparation_sec = started - self._action_preparation_started
        if previous is not None:
            time.sleep(max(0.0, previous + 1.0 / self.action_fps - started))
        sent = time.perf_counter()
        if self.action_timing_enabled:
            self._action_pace_wait += sent - started
            self._target_interval = None if previous is None else sent - previous
        self._last_action_target = sent

    def _send_force_action(
        self, action: np.ndarray, timing: dict[str, float] | None = None
    ):
        if getattr(self, "continuous_tracking", False):
            self._send_tracking_action(action, timing)
            return
        if self._is_zero_output_action(action):
            if self._zero_force_lock_active:
                self._pace_action_target()
                return
            if self.show_target_error:
                logging.info(
                    f"[ACTION] Zero output detected. Sending zero forceMode once. action={action.tolist()}"
                )
            self._pace_action_target()
            send_started = time.perf_counter()
            self.rtde_c.forceMode(
                self.task_frame,
                [0, 0, 0, 0, 0, 0],
                np.zeros(6),
                self.force_type,
                self.force_limit,
            )
            if timing is not None:
                timing["send"] = time.perf_counter() - send_started
            self._zero_force_lock_active = True
            return

        self._zero_force_lock_active = False
        ft_target = self._calculate_ft_target(action, timing)
        self._pace_action_target()
        send_started = time.perf_counter()
        self.rtde_c.forceMode(
            self.task_frame,
            self.select_vector,
            ft_target,
            self.force_type,
            self.force_limit,
        )
        if timing is not None:
            timing["send"] = time.perf_counter() - send_started

    def stop_force(self):
        if getattr(self, "continuous_tracking", False):
            controller = self._force_controller
            if controller is not None:
                controller.stop()
                self._report_force_timing(force=True)
            elif self.rtde_c is not None and not self.debug:
                _ContinuousForceController._require_success(
                    self.rtde_c.forceModeStop(), "forceModeStop"
                )
            return
        if self.rtde_c is not None and self.control_space == "force":
            self.rtde_c.forceMode(
                self.task_frame,
                [0, 0, 0, 0, 0, 0],
                np.zeros(6),
                self.force_type,
                self.force_limit,
            )
            if hasattr(self.rtde_c, "forceModeStop"):
                self.rtde_c.forceModeStop()

    def _send_tracking_action(self, action, timing=None):
        controller = self._force_controller
        if controller is None:
            raise RuntimeError(
                "Start an episode before sending continuous force targets"
            )
        controller._check()
        started = time.perf_counter()
        if self._is_zero_output_action(action):
            target = None
            state_read = target_ready = started
        else:
            with controller.io_lock:
                pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
                state_read = time.perf_counter()
                target = self._target_pose_from_action(action, current_tcp_pose=pose)
                target_ready = time.perf_counter()
        self._pace_action_target()
        send_started = time.perf_counter()
        controller.update_target(target)
        if timing is not None:
            timing["state_read"] = state_read - started
            timing["target"] = target_ready - state_read
            timing["send"] = time.perf_counter() - send_started

    def _report_force_timing(self, *, force=False):
        controller = getattr(self, "_force_controller", None)
        if controller is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_force_report < 1.0:
            return
        self._last_force_report = now
        timing = controller.drain_timing()
        if timing is not None:
            self._record_timing("controller", **timing)

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Transfer raw TCP-reference observation state to UR5e policy format."""

        state_parts = [
            np.asarray(obs["tcp_pose"], dtype=np.float32),
            np.asarray(obs["tcp_speed"], dtype=np.float32),
            np.asarray(obs["tcp_force"], dtype=np.float32),
        ]
        if not self.fix_gripper:
            state_parts.append(np.asarray([obs["gripper_position"]], dtype=np.float32))
        state = np.concatenate(state_parts)

        ur5e_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return ur5e_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> dict[str, Any]:
        """Return current TCP-reference observation from robot."""
        obs = {}

        if self.rtde_r:
            ee_pose = self.get_ee_pose()
            if self.reference_frame == "base":
                obs["tcp_pose"] = self._pose_euler(ee_pose)
            elif self.reference_frame == "tcp":
                obs["tcp_pose"] = self._relative_pose_euler(ee_pose)
            else:
                raise ValueError(f"Unsupported reference_frame: {self.reference_frame}")
            with self._robot_io():
                obs["tcp_speed"] = self.rtde_r.getActualTCPSpeed()
                obs["tcp_force"] = self.rtde_r.getActualTCPForce()

        if self.cameras:
            for name, cam in self.cameras.items():
                obs[name] = cam.read()

        if self.task_description:
            obs["prompt"] = self.task_description

        if not self.fix_gripper and self.gripper:
            obs["gripper_position"] = self.gripper.position

        return self._transfer_obs_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute model actions as TCP-frame end-effector deltas."""
        if self.rtde_c is None:
            logging.error(
                "[ERROR] Robot controller not connected. Cannot execute actions."
            )
            return

        if block:
            logging.info("[STATE] Moving robot to TCP pose...")
            self.move_to_tcp_pose(actions)
            logging.info("[STATE] Robot reached TCP pose.")
            return
        self.fps_action.reset()
        self._last_action_target = None
        chunk = np.asarray(actions)
        for action_index, action in enumerate(chunk, start=1):
            self._action_pace_wait = 0.0
            self._target_interval = None
            start_time = time.perf_counter()
            if self._last_action_target is not None:
                ready = self._last_action_target + 1.0 / self.action_fps
                time.sleep(
                    max(
                        0.0,
                        ready
                        - getattr(self, "_action_preparation_sec", 0.0)
                        - start_time,
                    )
                )
            self._action_preparation_started = time.perf_counter()
            if self.action_timing_enabled and self._last_action_target is not None:
                self._action_pace_wait += self._action_preparation_started - start_time
            timing = (
                dict.fromkeys(("state_read", "target", "impedance", "send"), 0.0)
                if self.action_timing_enabled
                else None
            )
            action = np.asarray(action, dtype=float)
            expected_action_dim = 6 if self.fix_gripper else 7
            if action.ndim != 1 or action.size < expected_action_dim:
                raise ValueError(
                    f"Each model action must contain at least {expected_action_dim} values, got {action.shape}."
                )
            action = action[:expected_action_dim]

            if not self.debug:
                if self.control_space == "position":
                    state_started = time.perf_counter()
                    current_pose = np.asarray(
                        self.rtde_r.getActualTCPPose(), dtype=float
                    )
                    state_read = time.perf_counter()
                    target_pose = self._target_pose_from_action(
                        action, current_tcp_pose=current_pose
                    )
                    target_ready = time.perf_counter()
                    self._pace_action_target()
                    send_started = time.perf_counter()
                    if timing is not None:
                        timing["state_read"] = state_read - state_started
                        timing["target"] = target_ready - state_read
                    # t_start = self.rtde_c.initPeriod()
                    self.rtde_c.servoL(
                        target_pose.tolist(),
                        self._velocity,
                        self._acceleration,
                        self._servo_time,
                        self._lookahead_time,
                        self._gain,
                    )
                    if timing is not None:
                        timing["send"] = time.perf_counter() - send_started
                    # self.rtde_c.waitPeriod(t_start)
                elif self.control_space == "force":
                    # t_start = self.rtde_c.initPeriod()
                    self._send_force_action(action, timing)
                    # self.rtde_c.waitPeriod(t_start)
                else:
                    raise ValueError(f"Unsupported control_space: {self.control_space}")

            if self.debug:
                self._pace_action_target()
            gripper_started = time.perf_counter()
            if not self.fix_gripper:
                self._gripper_position = float(action[6])

            processing_finished = time.perf_counter()
            elapsed = processing_finished - start_time
            if action_index == len(chunk):
                # Preserve the final target's execution period before the next observation.
                last_target = self._last_action_target or start_time
                time.sleep(
                    max(0.0, last_target + 1.0 / self.action_fps - time.perf_counter())
                )
            cycle_finished = time.perf_counter()
            wait_elapsed = self._action_pace_wait + cycle_finished - processing_finished
            elapsed -= self._action_pace_wait
            self.fps_action.update(show=False)
            if timing is not None:
                gripper = processing_finished - gripper_started
                other = max(0.0, elapsed - sum(timing.values()) - gripper)
                cycle = cycle_finished - start_time
                self._record_timing(
                    "action",
                    infer_index=getattr(self, "_infer_index", None),
                    action_index=action_index,
                    target_hz=self.action_fps,
                    control_hz=1.0 / cycle,
                    control_sec=cycle,
                    command_sec=elapsed,
                    state_read_sec=timing["state_read"],
                    target_sec=timing["target"],
                    impedance_sec=timing["impedance"],
                    send_sec=timing["send"],
                    gripper_target_sec=gripper,
                    other_sec=other,
                    pace_wait_sec=wait_elapsed,
                    target_interval_sec=self._target_interval,
                    control_overrun_sec=max(
                        0.0, (self._target_interval or 0.0) - 1.0 / self.action_fps
                    ),
                )
        self._report_force_timing()

    def _record_timing(self, event: str, **values: Any) -> None:
        """Save routine timings and optionally detailed action measurements."""
        if event == "action" and not self.action_timing_enabled:
            return
        if not self.save_logs:
            return
        with self._timing_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"event": event, "timestamp": time.time(), **values}) + "\n"
            )

    # --------------------------- KEYBOARD EPISODE CONTROL --------------------------- #
    def _keyboard_episode_control_listener(self):
        """Read double-press controls without buffering bytes past select readiness."""
        last_enter_time = None
        last_space_time = None
        old_termios = None
        input_fd = sys.stdin.fileno()
        if sys.stdin.isatty():
            old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        try:
            while not self._keyboard_listener_stop.is_set():
                ready, _, _ = select.select([input_fd], [], [], 0.05)
                if not ready:
                    continue
                raw = os.read(input_fd, 1)
                if not raw:
                    break
                char = raw.decode("ascii", errors="ignore")
                now = time.perf_counter()
                if char == "." and getattr(self, "stage2_enabled", False):
                    if self._critical_switch.press(now):
                        logging.info("[RLT] MLP selected for the next inference chunk.")
                    continue
                if char in ("\n", "\r"):
                    if last_enter_time is not None and now - last_enter_time <= 0.5:
                        self._reset_requested.set()
                        last_enter_time = None
                        logging.info(
                            "[RESET] Double Enter detected. Reset requested; current episode marked failed."
                        )
                    else:
                        last_enter_time = now
                    continue
                if char != " ":
                    continue
                if last_space_time is not None and now - last_space_time <= 0.5:
                    self._success_requested.set()
                    last_space_time = None
                    logging.info("[EPISODE] Double Space detected. Success requested.")
                else:
                    last_space_time = now
        finally:
            if old_termios is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)

    def _start_keyboard_episode_control_listener(self):
        self._reset_requested.clear()
        self._success_requested.clear()
        self._keyboard_listener_stop.clear()
        self._keyboard_listener_thread = threading.Thread(
            target=self._keyboard_episode_control_listener, daemon=True
        )
        self._keyboard_listener_thread.start()
        if getattr(self, "stage2_enabled", False):
            logging.info(
                "[RLT] Executing VLA actions. Double '.' within 0.5s to select MLP; reset restores VLA."
            )
        if self.use_target_success:
            logging.info(
                "[EPISODE] Press Enter twice within 0.5s to fail/reset, or Space twice within 0.5s to request target-checked success."
            )
        else:
            logging.info(
                "[EPISODE] Press Enter twice within 0.5s to fail/reset, or Space twice within 0.5s to mark success."
            )

    def _stop_keyboard_episode_control_listener(self):
        self._keyboard_listener_stop.set()
        if self._keyboard_listener_thread is not None:
            self._keyboard_listener_thread.join(timeout=0.2)
            self._keyboard_listener_thread = None

    def _wait_for_next_episode_start(self):
        input(f"Press Enter to start episode {self._ep_idx}...")

    # --------------------------- EPISODE TIME DISPLAY --------------------------- #
    def _print_episode_time_every_second(
        self,
        episode_index: int,
        started_at: float,
        stop_event: threading.Event,
    ):
        next_elapsed_sec = 1
        while True:
            wait_sec = started_at + next_elapsed_sec - time.perf_counter()
            if stop_event.wait(max(wait_sec, 0.0)):
                return
            elapsed_sec = int(time.perf_counter() - started_at)
            if elapsed_sec <= 0:
                next_elapsed_sec = 1
                continue
            self._record_timing(
                "episode_progress", episode_index=episode_index, elapsed_sec=elapsed_sec
            )
            next_elapsed_sec = elapsed_sec + 1

    def _start_episode_time_printer(self):
        if self._ep_start is None:
            return

        self._stop_episode_time_printer()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._print_episode_time_every_second,
            args=(self._ep_idx, self._ep_start, stop_event),
            daemon=True,
        )
        self._episode_time_printer_stop = stop_event
        self._episode_time_printer_thread = thread
        thread.start()

    def _stop_episode_time_printer(self):
        stop_event = self._episode_time_printer_stop
        thread = self._episode_time_printer_thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._episode_time_printer_stop = None
        self._episode_time_printer_thread = None

    # --------------------------- PIPELINE --------------------------- #
    def _create_openpi_policy(self):
        from .policy import create_policy

        return create_policy(self._native_config, timing_recorder=self._record_timing)

    def _prepare_inference(self):
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        self.connect_gripper()
        self.wait_for_gripper_states()
        self.reset_episode()

        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation state: {obs.keys()}")
        obs_values = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim <= 1:
                obs_values[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                obs_values[key] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                }
            else:
                obs_values[key] = value
        logging.info(f"[STATE] Observation values: {obs_values}")
        policy = self._create_openpi_policy()

        from .policy import warmup_policy

        warmup_policy(
            policy,
            obs,
            timing_recorder=self._record_timing,
        )

        if self.fix_gripper:
            logging.info(
                "[GRIPPER] fix_gripper enabled; skipping manual pre-inference close."
            )
        elif not self.init_gripper:
            input("Press Enter to close the gripper...")
            self._gripper_position = 0.0
            logging.info("[GRIPPER] Gripper close target set.")
        else:
            logging.info(
                "[GRIPPER] init_gripper enabled; skipping manual pre-inference close."
            )
        input("Press Enter to continue inference...")
        return policy

    def _start_episode(self):
        self._stop_episode_time_printer()
        if getattr(self, "continuous_tracking", False) and not self.debug:
            if self._force_controller is None:
                self._force_controller = _ContinuousForceController(self)
            self._force_controller.start()
        self._ep_start = time.perf_counter()
        self._ep_steps = 0
        self._ep_done = False
        self._zero_force_lock_active = False
        self._start_episode_time_printer()

    def _submit_step(self, result, obs, infer_idx: int):
        self._infer_index = infer_idx
        action_dim = 6 if self.fix_gripper else 7
        robot_actions = np.asarray(result["actions"])[..., :action_dim]
        self.execute_actions(robot_actions)
        self._ep_steps += 1
        self.recorder.submit_actions(
            robot_actions,
            infer_idx,
            obs["prompt"],
            state=obs["observation/state"],
            episode_index=self._ep_idx,
            episode_action_count=self._ep_steps,
            action_horizon=len(robot_actions),
            controller=result.get("controller", "vla"),
        )
        self.recorder.submit_obs(obs)

    def _submit_episode(self, status: str, completed: bool, **extra):
        if getattr(self, "continuous_tracking", False):
            self.stop_force()
        duration = time.perf_counter() - self._ep_start
        self.recorder.submit_episode_result(
            episode_index=self._ep_idx,
            duration_sec=duration,
            status=status,
            action_batches=self._ep_steps,
            completed=completed,
            **extra,
        )
        if completed:
            self._success_episodes += 1
        else:
            self._failed_episodes += 1

        total_ended = self._success_episodes + self._failed_episodes
        success_rate = self._success_episodes / total_ended
        failure_rate = self._failed_episodes / total_ended
        logging.info(
            f"[EPISODE] {self._ep_idx}: {status}, {duration:.1f}s; "
            f"success {self._success_episodes}/{total_ended} ({success_rate:.1%}), "
            f"failure {self._failed_episodes}/{total_ended} ({failure_rate:.1%})"
        )
        self._ep_done = True
        self._stop_episode_time_printer()

    def _finish_episode_if_needed(self) -> bool:
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._success_requested.clear()
            logging.info("[RESET] Restarting current episode from initial pose.")
            self._submit_episode("keyboard_reset", completed=False, reason="enter")
        elif self._success_requested.is_set():
            self._success_requested.clear()
            if not self.use_target_success:
                self._submit_episode(
                    "manual_success", completed=True, reason="double_space"
                )
            elif self.has_reached_target():
                self._submit_episode(
                    "target_reached", completed=True, reason="double_space"
                )
            else:
                logging.info(
                    "[EPISODE] Double Space ignored because target pose threshold is not satisfied."
                )
                return False
        elif time.perf_counter() - self._ep_start >= self.ep_timeout:
            logging.info(
                f"[RESET] Episode timeout ({self.ep_timeout}s). Restarting current episode."
            )
            self._submit_episode("timeout", completed=False, reason="episode_timeout")
        elif self.use_target_success and self.has_reached_target():
            self._submit_episode("target_reached", completed=True)
        else:
            return False

        if self._ep_idx >= self.num_episodes:
            self._ep_idx += 1
            return True

        self._ep_idx += 1
        self._stop_keyboard_episode_control_listener()
        self.reset_episode()
        self._wait_for_next_episode_start()
        self._start_keyboard_episode_control_listener()
        self._start_episode()
        return True

    def _close_episode_on_exit(self):
        if self._ep_steps > 0 and not self._ep_done:
            duration = time.perf_counter() - self._ep_start
            logging.info(
                f"[EPISODE] Episode {self._ep_idx} interrupted: "
                f"{self.recorder._format_duration(duration)}"
            )

    def _reset_before_exit(self):
        if self.rtde_r is None or self.rtde_c is None:
            return
        try:
            logging.info("[RESET] Resetting robot before exit.")
            self.reset_episode()
        except Exception as e:
            logging.error(f"[ERROR] Failed to reset robot before exit: {e}")

    def _ask_save_video(self):
        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

    def run(self):
        """Main pipeline: connect robot, cameras, and run inference."""
        completed = False
        try:
            policy = self._prepare_inference()
            infer_idx = 1
            self._start_keyboard_episode_control_listener()
            self._start_episode()
            logging.info("========== Starting Inference Loop ==========")
            while self._ep_idx <= self.num_episodes:
                loop_start = time.perf_counter()
                obs_start = time.perf_counter()
                obs = self.get_obs_state()
                obs_elapsed = time.perf_counter() - obs_start
                infer_start = time.perf_counter()
                if getattr(self, "stage2_enabled", False):
                    obs["rlt_actor_enabled"] = self._critical_switch.active
                result = policy.infer(obs)
                infer_elapsed = time.perf_counter() - infer_start
                submit_start = time.perf_counter()
                self._submit_step(result, obs, infer_idx)
                submit_elapsed = time.perf_counter() - submit_start
                loop_elapsed = time.perf_counter() - loop_start
                self._record_timing(
                    "step",
                    infer_index=infer_idx,
                    loop_sec=loop_elapsed,
                    loop_hz=1.0 / loop_elapsed,
                    observation_sec=obs_elapsed,
                    infer_sec=infer_elapsed,
                    action_and_record_sec=submit_elapsed,
                )
                self._finish_episode_if_needed()
                infer_idx += 1
            completed = True
            logging.info(f"[INFO] Finished {self.num_episodes} inference episodes.")
        except KeyboardInterrupt:
            logging.info(
                "[INFO] KeyboardInterrupt detected. Saving recorded videos before exiting..."
            )
        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")
        finally:
            try:
                if completed:
                    self._reset_before_exit()
            finally:
                self.close()
            self._close_episode_on_exit()
            self.recorder.submit_episode_summary(
                action_horizon=self.action_chunk,
                description=self.task_description,
            )

        self._ask_save_video()


# --------------------------- MAIN --------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run VLA-Precision UR inference with an RLinf checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "vla_precision_ur.yaml",
        help="Path to the VLA-Precision UR inference YAML config.",
    )
    args = parser.parse_args()
    inference = Inference(args.config)
    inference.run()


# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
