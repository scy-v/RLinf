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

"""Record executed single-step actions from synchronous model chunks."""

import copy
import sys
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from tqdm import tqdm

from rlinf.utils.logging import get_logger

from .terminal import color_status

logger = get_logger()

ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper_position",
)


@dataclass(frozen=True)
class ControlState:
    """Operator state; an outcome is latched until the episode is saved."""

    outcome: bool | None = None
    discard: bool = False
    stop: bool = False
    finish_requested: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    frames: int
    discarded: bool = False
    stop: bool = False
    needs_review: bool = False


def rollout_features(robot: Any) -> dict:
    """Extend the original UR schema with outcome and action-source flags."""
    from lerobot.datasets.utils import hw_to_dataset_features

    features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(
            robot.observation_features, "observation", use_video=True
        ),
    }
    for name in ("is_success", "intervene_flag"):
        features[name] = {"dtype": "bool", "shape": (1,), "names": [name]}
    return features


def save_episode(dataset: Any, result: EpisodeResult) -> None:
    """Stamp the outcome before committing; interrupted recordings are discarded."""
    if result.needs_review:
        raise ValueError("Expected a reviewed outcome before saving an episode.")
    if result.discarded:
        dataset.clear_episode_buffer()
        return
    size = int(dataset.episode_buffer["size"])
    if size != result.frames or size == 0:
        raise ValueError(f"Expected {result.frames} recorded frames, found {size}.")
    dataset.episode_buffer["is_success"] = [
        np.array([result.success], dtype=np.bool_) for _ in range(size)
    ]
    dataset.save_episode()


def review_episode(result: EpisodeResult, controls: Any) -> EpisodeResult:
    """Wait for a label without executing or recording any robot actions."""
    if not result.needs_review:
        return result
    controls.begin_review()
    try:
        while True:
            control = controls.snapshot()
            if control.stop or control.discard:
                return replace(
                    result,
                    success=False,
                    discarded=True,
                    stop=control.stop,
                    needs_review=False,
                )
            if control.outcome is not None:
                return replace(result, success=control.outcome, needs_review=False)
            time.sleep(0.02)
    finally:
        controls.idle()


def manual_reset(
    robot: Any,
    teleop: Any,
    controls: Any,
    *,
    fps: float,
    duration: float,
    display: bool = False,
    step_sizes: tuple[float, float] | None = None,
) -> None:
    """Retain the original unrecorded keyboard phase before automatic reset."""
    # Reset observations need a temporary origin; recording sets a fresh one later.
    robot.set_episode_reference_pose()
    teleop.set_input_enabled(True)
    controls.begin_reset()
    original_sizes = None
    if step_sizes is not None:
        original_sizes = (
            teleop.default_step_size,
            teleop.default_rot_step_size,
            teleop.alternate_step_size,
            teleop.alternate_rot_step_size,
        )
        teleop.default_step_size, teleop.default_rot_step_size = step_sizes
        teleop.alternate_step_size = teleop.alternate_rot_step_size = None
    teleop.reset_step_size()
    teleop.gripper_action = float(getattr(robot, "_gripper_position", 1.0))
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration:
            if controls.snapshot().stop or controls.reset_finished():
                break
            tick = time.monotonic()
            observation = robot.get_observation()
            action = robot.send_action(teleop.get_action())
            if display:
                from lerobot.utils.visualization_utils import log_rerun_data

                log_rerun_data(observation=observation, action=action)
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - tick)))
    finally:
        if original_sizes is not None:
            (
                teleop.default_step_size,
                teleop.default_rot_step_size,
                teleop.alternate_step_size,
                teleop.alternate_rot_step_size,
            ) = original_sizes
            teleop.reset_step_size()
        teleop.set_input_enabled(False)
        controls.idle()
        robot.stop_force()


def record_rollout_episode(
    robot: Any,
    teleop: Any,
    policy: Any,
    controls: Any,
    dataset: Any,
    *,
    task: str,
    chunk_size: int,
    fps: float,
    max_chunks: int,
    close_threshold: float = 0.7,
    display: bool = False,
) -> EpisodeResult:
    """Finish each chunk before a manual stop or the episode chunk limit.

    Intervention replaces only the current action slot. The chunk cursor always
    advances, so releasing control resumes the original remaining model actions.
    Manual stops require a separate result review; the chunk limit means failure.
    Escape aborts immediately and discards the unsaved episode.
    """
    from lerobot.datasets.utils import build_dataset_frame

    if chunk_size < 1 or fps <= 0 or max_chunks < 1:
        raise ValueError("chunk_size, fps and max_chunks must be positive.")
    if not 0 < close_threshold <= 1:
        raise ValueError("close_threshold must be in (0, 1].")
    frames = 0
    chunks = 0
    previously_intervening = False
    last_gripper = float(getattr(robot, "_gripper_position", 1.0))
    with tqdm(
        total=max_chunks,
        desc=color_status("[采集] 已完成 chunk"),
        unit="chunk",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
        mininterval=0.1,
    ) as progress:
        while True:
            control = controls.snapshot()
            if control.stop or control.discard:
                return EpisodeResult(False, frames, discarded=True, stop=control.stop)
            if control.finish_requested:
                return EpisodeResult(
                    False, frames, discarded=frames == 0, needs_review=frames > 0
                )
            observation = robot.get_observation()
            actions = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < chunk_size:
                raise ValueError(
                    f"Expected at least ({chunk_size}, 7) actions, got {actions.shape}."
                )
            if not np.isfinite(actions[:chunk_size]).all():
                raise ValueError("Expected finite model actions.")
            for index in range(chunk_size):
                started = time.monotonic()
                control = controls.snapshot()
                if control.stop or control.discard:
                    return EpisodeResult(
                        False, frames, discarded=True, stop=control.stop
                    )
                # Inference and operator actions may take time; record a fresh observation.
                observation = robot.get_observation()
                if not previously_intervening:
                    teleop.gripper_action = last_gripper
                human_action = teleop.get_action()
                intervening = teleop.intervention_active
                progress.set_postfix_str(
                    "人工接管" if intervening else "模型控制", refresh=False
                )
                progress.update(0)
                selected = (
                    human_action
                    if intervening
                    else dict(zip(ACTION_NAMES, actions[index].tolist(), strict=True))
                )
                action = {name: float(selected[name]) for name in ACTION_NAMES}
                # Store logical open/close commands; hardware reversal stays in the driver.
                action["gripper_position"] = float(
                    action["gripper_position"] >= close_threshold
                )
                observation_frame = build_dataset_frame(
                    dataset.features, observation, "observation"
                )
                # Snapshot before dispatch in case a device mutates its input dict.
                sent = robot.send_action(copy.deepcopy(action))
                executed = action if sent is None else sent
                last_gripper = float(executed["gripper_position"])
                action_frame = build_dataset_frame(dataset.features, executed, "action")
                dataset.add_frame(
                    {
                        **observation_frame,
                        **action_frame,
                        "task": task,
                        "is_success": np.array([False], dtype=np.bool_),
                        "intervene_flag": np.array([intervening], dtype=np.bool_),
                    }
                )
                frames += 1
                previously_intervening = intervening
                if display:
                    from lerobot.utils.visualization_utils import log_rerun_data

                    log_rerun_data(observation=observation, action=executed)
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))
            control = controls.snapshot()
            if control.stop or control.discard:
                return EpisodeResult(False, frames, discarded=True, stop=control.stop)
            chunks += 1
            progress.update(1)
            if control.finish_requested:
                return EpisodeResult(False, frames, needs_review=True)
            if chunks >= max_chunks:
                logger.info(
                    color_status(
                        "[采集] 已执行 %s/%s 个 chunk（%s 帧），按达到上限失败保存。"
                    ),
                    chunks,
                    max_chunks,
                    frames,
                )
                return EpisodeResult(False, frames)
