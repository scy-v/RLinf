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

"""UR5e Cartesian targets over the Universal Robots RTDE interfaces."""

import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.robotics.parts.arms.base import Arm, BaseArm
from rlinf.robotics.parts.base import Action, Features
from rlinf.robotics.parts.claims import DeviceClaim


@dataclass
class UR5eRobotState:
    """Base-frame TCP state; pose uses metres and an XYZW quaternion."""

    tcp_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arm_joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tcp_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tcp_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_position: float = 0.0
    gripper_open: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot under RLinf's canonical arm field names."""
        return asdict(self)


@dataclass(frozen=True)
class UR5eImpedanceConfig:
    """UR force-mode gains and clipping, matching VLA-Precision's running controller.

    Position errors are metres; velocity errors are m/s. Rotation-error
    clipping is degrees. The reference divides rotational PD torque by Hz.
    """

    frequency: float = 100.0
    kp: float = 1200.0
    kd: float = 600.0
    kp_rotation: float = 2000.0
    kd_rotation: float = 400.0
    position_error_clip: float = 0.048
    velocity_error_clip: float = 4.0
    rotation_error_clip_degrees: float = 20.0
    control_frame_euler_deg: tuple[float, ...] = (0.0, 0.0, 0.0)
    selection_vector: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    force_limits: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    force_type: int = 2
    force_gain_scale: float = 1.5
    payload_mass: float | None = None
    payload_cog: tuple[float, ...] = (0.0, 0.0, 0.0)
    command_timeout: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid settings before connecting any device."""
        positive = (
            self.frequency,
            self.position_error_clip,
            self.velocity_error_clip,
            self.rotation_error_clip_degrees,
            self.command_timeout,
        )
        gains = (self.kp, self.kd, self.kp_rotation, self.kd_rotation)
        if not all(np.isfinite(v) and v > 0 for v in positive) or not all(
            np.isfinite(v) and v >= 0 for v in gains
        ):
            raise ValueError(
                "Expected finite positive impedance frequency/clips/timeout and nonnegative gains."
            )
        if self.force_type != 2 or not 0 <= self.force_gain_scale <= 2:
            raise ValueError(
                "Expected fixed task-frame force_type=2 and force_gain_scale in [0, 2]."
            )
        for name, size in (
            ("control_frame_euler_deg", 3),
            ("selection_vector", 6),
            ("force_limits", 6),
            ("payload_cog", 3),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(
                    f"Expected {name} to contain {size} finite values, got {value}."
                )
        if not all(v in (0, 1) for v in self.selection_vector) or not all(
            v > 0 for v in self.force_limits
        ):
            raise ValueError(
                "Expected binary selection_vector and positive force_limits."
            )
        if self.payload_mass is not None and (
            not np.isfinite(self.payload_mass) or self.payload_mass < 0
        ):
            raise ValueError(
                "Expected payload_mass to be None or a nonnegative finite mass."
            )


@Arm.register("ur_rtde")
class UR5eArm(BaseArm):
    """Track absolute TCP targets with a continuously running UR force-mode loop.

    The environment owns delta composition. The driver holds the most recent
    target between policy calls and reports controller failures to callers.
    """

    STATE_FIELDS = (
        "tcp_pose",
        "tcp_vel",
        "arm_joint_position",
        "arm_joint_velocity",
        "tcp_force",
        "tcp_torque",
    )

    def __init__(
        self,
        robot_ip: str,
        speed: float = 0.1,
        acceleration: float = 0.2,
        impedance: Mapping[str, Any] | None = None,
    ) -> None:
        if not robot_ip or not 0 < speed <= 1 or not 0 < acceleration <= 2:
            raise ValueError(
                "Expected robot_ip, reset speed in (0, 1] and acceleration in (0, 2]."
            )
        self.robot_ip = robot_ip
        self.speed = speed
        self.acceleration = acceleration
        self.impedance = UR5eImpedanceConfig(**(impedance or {}))
        self._claim = DeviceClaim(f"ur-rtde:{robot_ip}", type(self).__name__)

    @property
    def action_features(self) -> Features:
        """Declare an absolute base-frame XYZ + XYZW TCP target."""
        return {"tcp_pose": {"shape": (7,), "dtype": "float32"}}

    def _open(self) -> Any:
        import rtde_control
        import rtde_receive

        self._claim.acquire()
        control = receive = None
        self._io_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop_error: BaseException | None = None
        self._sequence = 0
        self._applied_sequence = -1
        self._control_rotation = Rotation.from_euler(
            "xyz", self.impedance.control_frame_euler_deg, degrees=True
        ).as_matrix()
        self._task_frame = [
            0.0,
            0.0,
            0.0,
            *Rotation.from_matrix(self._control_rotation).as_rotvec().tolist(),
        ]
        try:
            receive = rtde_receive.RTDEReceiveInterface(
                self.robot_ip, self.impedance.frequency
            )
            control = rtde_control.RTDEControlInterface(
                self.robot_ip, self.impedance.frequency
            )
            if not control.isConnected() or not receive.isConnected():
                raise ConnectionError(
                    f"Expected connected RTDE interfaces at {self.robot_ip}."
                )
            self._require_success(
                control.forceModeSetGainScaling(self.impedance.force_gain_scale),
                "forceModeSetGainScaling",
            )
            if self.impedance.payload_mass is not None:
                self._require_success(
                    control.setPayload(
                        self.impedance.payload_mass, list(self.impedance.payload_cog)
                    ),
                    "setPayload",
                )
            return control, receive
        except BaseException:
            try:
                if control is not None:
                    try:
                        control.stopScript()
                    finally:
                        control.disconnect()
            finally:
                try:
                    if receive is not None:
                        receive.disconnect()
                finally:
                    self._claim.release()
            raise

    @staticmethod
    def _require_success(result: Any, operation: str) -> None:
        if isinstance(result, (bool, np.bool_)) and not result:
            raise RuntimeError(f"UR5e {operation} did not complete successfully.")

    def _opened(self) -> None:
        self._start_loop()

    def _start_loop(self) -> None:
        self._stop_event.clear()
        self._loop_error = None
        self._sequence = 0
        self._applied_sequence = -1
        self._target = self._read_state().tcp_pose
        self._thread = threading.Thread(
            target=self._run_controller, name="UR5e-force-mode", daemon=True
        )
        self._thread.start()
        with self._condition:
            self._wait_applied(0)

    def _wait_applied(self, sequence: int) -> None:
        ready = self._condition.wait_for(
            lambda: (
                self._applied_sequence >= sequence
                or self._loop_error is not None
                or self._stop_event.is_set()
            ),
            timeout=self.impedance.command_timeout,
        )
        self._check_controller()
        if not ready or self._applied_sequence < sequence:
            self._stop_event.set()
            raise RuntimeError(
                "UR5e force-mode controller did not apply the target before command_timeout."
            )

    def _check_controller(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError(
                "UR5e force-mode controller failed."
            ) from self._loop_error
        if self._stop_event.is_set():
            raise RuntimeError(
                "Expected a running UR5e force-mode controller, found stopped."
            )

    def _compute_wrench(
        self, target: np.ndarray, current: UR5eRobotState
    ) -> np.ndarray:
        cfg = self.impedance
        position_error = np.clip(
            target[:3] - current.tcp_pose[:3],
            -cfg.position_error_clip,
            cfg.position_error_clip,
        )
        velocity_error = np.clip(
            -current.tcp_vel[:3], -cfg.velocity_error_clip, cfg.velocity_error_clip
        )
        force = cfg.kp * position_error + cfg.kd * velocity_error
        rotation_error = (
            Rotation.from_quat(target[3:])
            * Rotation.from_quat(current.tcp_pose[3:]).inv()
        ).as_rotvec()
        angle = np.linalg.norm(rotation_error)
        rotation_error *= min(
            1.0, np.deg2rad(cfg.rotation_error_clip_degrees) / max(angle, 1e-8)
        )
        angular_velocity_error = (
            Rotation.from_rotvec(current.tcp_vel[3:]).inv().as_rotvec()
        )
        torque = (
            cfg.kp_rotation * rotation_error + cfg.kd_rotation * angular_velocity_error
        ) / cfg.frequency
        return np.concatenate(
            (self._control_rotation.T @ force, self._control_rotation.T @ torque)
        )

    def _run_controller(self) -> None:
        control, _ = self._device
        try:
            while not self._stop_event.is_set():
                cycle_start = time.monotonic()
                with self._condition:
                    target, sequence = self._target.copy(), self._sequence
                with self._io_lock:
                    state = self._read_state()
                    wrench = self._compute_wrench(target, state)
                    self._require_success(
                        control.forceMode(
                            self._task_frame,
                            list(self.impedance.selection_vector),
                            wrench.tolist(),
                            self.impedance.force_type,
                            list(self.impedance.force_limits),
                        ),
                        "forceMode",
                    )
                with self._condition:
                    self._applied_sequence = sequence
                    self._condition.notify_all()
                self._stop_event.wait(
                    max(
                        0.0,
                        1.0 / self.impedance.frequency
                        - (time.monotonic() - cycle_start),
                    )
                )
        except BaseException as error:
            with self._condition:
                self._loop_error = error
                self._stop_event.set()
                self._condition.notify_all()
        finally:
            try:
                with self._io_lock:
                    self._require_success(control.forceModeStop(), "forceModeStop")
            except BaseException as error:
                with self._condition:
                    self._loop_error = self._loop_error or error
                    self._condition.notify_all()

    def _stop_loop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self.impedance.command_timeout)
            if self._thread.is_alive():
                raise RuntimeError(
                    "UR5e force-mode thread did not stop before command_timeout."
                )

    def _release(self, device: Any) -> None:
        control, receive = device
        try:
            try:
                self._stop_loop()
            finally:
                try:
                    self._require_success(control.forceModeStop(), "forceModeStop")
                finally:
                    control.stopScript()
        finally:
            try:
                control.disconnect()
            finally:
                try:
                    receive.disconnect()
                finally:
                    self._claim.release()

    def _read_state(self) -> UR5eRobotState:
        _, receive = self._device
        with self._io_lock:
            pose = np.asarray(receive.getActualTCPPose(), dtype=np.float64)
            speed = np.asarray(receive.getActualTCPSpeed(), dtype=np.float64)
            wrench = np.asarray(receive.getActualTCPForce(), dtype=np.float64)
            joints = np.asarray(receive.getActualQ(), dtype=np.float64)
            joint_speed = np.asarray(receive.getActualQd(), dtype=np.float64)
        if (
            pose.shape != (6,)
            or speed.shape != (6,)
            or not np.isfinite(np.concatenate((pose, speed))).all()
        ):
            raise RuntimeError(
                "Expected finite six-value UR5e TCP pose and speed readings."
            )
        return UR5eRobotState(
            tcp_pose=np.concatenate(
                (pose[:3], Rotation.from_rotvec(pose[3:]).as_quat())
            ),
            tcp_vel=speed,
            arm_joint_position=joints,
            arm_joint_velocity=joint_speed,
            tcp_force=wrench[:3],
            tcp_torque=wrench[3:],
        )

    def get_state(self) -> UR5eRobotState:
        """Read measured state and surface background control failures."""
        self._check_controller()
        return self._read_state()

    def send_action(self, action: Action) -> Action:
        """Replace the held TCP target and wait for its first force-mode update."""
        if set(action) != {"tcp_pose"}:
            raise KeyError(f"Expected only tcp_pose, got {sorted(action)}.")
        pose = np.asarray(action["tcp_pose"], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"Expected finite TCP pose shape (7,), got {pose}.")
        rotation = Rotation.from_quat(pose[3:])
        pose = np.concatenate((pose[:3], rotation.as_quat()))
        target = np.concatenate((pose[:3], rotation.as_rotvec())).tolist()
        with self._command_lock:
            self._check_controller()
            control, _ = self._device
            with self._io_lock:
                if not control.isPoseWithinSafetyLimits(target):
                    raise ValueError(
                        f"UR5e TCP target is outside robot safety limits: {target}."
                    )
            with self._condition:
                self._target = pose.copy()
                self._sequence += 1
                self._wait_applied(self._sequence)
        return {"tcp_pose": pose.astype(np.float32)}

    def reset_joint(self, positions: list[float]) -> None:
        """Leave force mode, reset six joints, then hold the measured TCP pose."""
        joints = np.asarray(positions, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError(f"Expected six finite UR5e joint angles, got {joints}.")
        with self._command_lock:
            self._check_controller()
            self._stop_loop()
            control, _ = self._device
            with self._io_lock:
                self._require_success(
                    control.moveJ(joints.tolist(), self.speed, self.acceleration),
                    "moveJ",
                )
            self._start_loop()
