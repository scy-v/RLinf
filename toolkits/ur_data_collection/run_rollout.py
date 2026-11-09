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

"""Collect synchronous PI05 UR rollouts using the original keyboard robot."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

from .controls import RolloutControls
from .policy import URRolloutPolicy
from .rollout_loop import (
    manual_reset,
    record_rollout_episode,
    review_episode,
    rollout_features,
    save_episode,
)
from .terminal import (
    color_status,
    configure_console_logging,
    format_collection_summary,
    quiet_keyboard,
    wait_for_enter,
)

logger = get_logger()


T = TypeVar("T")


def format_elapsed(seconds: float) -> str:
    """Round elapsed time to whole seconds, using minutes when needed."""
    minutes, remainder = divmod(int(seconds + 0.5), 60)
    return f"{minutes}min {remainder}s" if minutes else f"{remainder}s"


def write_collection_metadata(
    path: Path,
    cfg: dict[str, Any],
    counts: list[list[int]],
    timing: dict[str, Any] | None,
) -> None:
    """Atomically save configuration, dataset totals and optional session timing."""

    def totals(success: int, failure: int) -> dict[str, int]:
        return {"success": success, "failure": failure, "total": success + failure}

    payload = {
        **cfg,
        "collection_summary": {
            "scope": "current_dataset",
            "autonomous": totals(*counts[0]),
            "intervened": totals(*counts[1]),
            "total": totals(counts[0][0] + counts[1][0], counts[0][1] + counts[1][1]),
        },
    }
    if timing is not None:
        payload["timing"] = timing
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class CollectionTiming:
    """Measure one collection session, including interrupted phase calls."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.started_at = datetime.now().astimezone().isoformat()
        self.started = time.perf_counter()
        self.seconds = dict.fromkeys(("record", "reset", "wait", "review", "save"), 0.0)

    def call(
        self, phase: str, function: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        if not self.enabled:
            return function(*args, **kwargs)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.seconds[phase] += time.perf_counter() - started

    def finish(self) -> dict[str, Any]:
        total = time.perf_counter() - self.started
        return {
            "started_at": self.started_at,
            "ended_at": datetime.now().astimezone().isoformat(),
            "total_duration": format_elapsed(total),
            **{
                f"{phase}_duration": format_elapsed(value)
                for phase, value in self.seconds.items()
            },
            "other_duration": format_elapsed(
                max(0.0, total - sum(self.seconds.values()))
            ),
        }


def saved_episode_counts(root: Path) -> list[list[int]]:
    """Read saved episode outcomes and any-frame intervention flags for resume."""
    import pyarrow.parquet as pq

    episodes = {}
    for path in sorted(root.glob("data/**/*.parquet")):
        rows = pq.read_table(
            path, columns=["episode_index", "is_success", "intervene_flag"]
        ).to_pylist()
        for row in rows:
            success = bool(
                row["is_success"][0]
                if isinstance(row["is_success"], list)
                else row["is_success"]
            )
            intervened = bool(
                row["intervene_flag"][0]
                if isinstance(row["intervene_flag"], list)
                else row["intervene_flag"]
            )
            key = row["episode_index"]
            previous = episodes.get(key, (success, False))
            episodes[key] = (success, previous[1] or intervened)
    counts = [[0, 0], [0, 0]]
    for success, intervened in episodes.values():
        counts[int(intervened)][int(not success)] += 1
    return counts


def load_config(path: Path) -> dict[str, Any]:
    """Load the local UR rollout configuration."""
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    record = cfg["record"]
    if record["robot"]["debug"] or not record["cameras"]["enabled"]:
        raise ValueError(
            "Real rollout recording requires robot.debug=false and cameras.enabled=true."
        )
    if not record["robot"]["use_gripper"]:
        raise ValueError(
            "The PI05 UR checkpoint requires a seven-dimensional action with gripper."
        )
    if record["robot"]["reference_frame"] != "tcp":
        raise ValueError(
            "The current PI05 checkpoint requires robot.reference_frame=tcp."
        )
    if record["storage"].get("push_to_hub", False):
        raise ValueError(
            "Rollout recording saves locally; set storage.push_to_hub=false."
        )
    if cfg["rollout"]["max_chunks"] < 1 or record["fps"] <= 0:
        raise ValueError("rollout.max_chunks and record.fps must be positive.")
    return cfg


def inference_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reuse the standalone PI05 loader with its local synchronous contract."""
    return {
        "model": cfg["model"],
        "policy": {"location": "local"},
        "robot": {"fix_gripper": False},
    }


def open_dataset(cfg: dict[str, Any], features: dict[str, Any]) -> Any:
    """Create a local v3 writer or resume a matching existing dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    record = cfg["record"]
    storage = record["storage"]
    if storage.get("root"):
        root = Path(storage["root"]).expanduser()
    elif record["task"].get("resume", False):
        raise ValueError(
            "Resuming requires record.storage.root pointing to an existing dataset."
        )
    else:
        root = Path(storage["output_dir"]).expanduser() / datetime.now().strftime(
            "%Y%m%d-%H%M%S-%f"
        )
    if record["task"].get("resume", False):
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(f"Expected an existing local dataset at {root}.")
        dataset = LeRobotDataset(record["repo_id"], root=root)
        if dataset.fps != record["fps"]:
            raise ValueError(
                f"Dataset FPS {dataset.fps} differs from configured {record['fps']}."
            )
        for name, feature in features.items():
            actual = dataset.features.get(name)
            if (
                actual is None
                or actual["dtype"] != feature["dtype"]
                or tuple(actual["shape"]) != tuple(feature["shape"])
                or actual.get("names") != feature.get("names")
            ):
                raise ValueError(
                    f"Dataset feature {name!r} differs from rollout schema: {actual}."
                )
        dataset.start_image_writer(num_threads=4)
    else:
        dataset = LeRobotDataset.create(
            record["repo_id"],
            root=root,
            fps=record["fps"],
            robot_type="ur5e",
            features=features,
            use_videos=True,
            image_writer_threads=4,
        )
    logger.info("Rollout dataset: %s", root)
    dataset.meta.metadata_buffer_size = 1
    return dataset


def run(cfg: dict[str, Any]) -> None:
    """Warm up PI05, collect labeled episodes, then release all resources."""
    from evaluations.realworld.vla_precision_ur.policy import (
        create_policy,
        warmup_policy,
    )

    from .recording import RecordConfig, make_hardware

    record_cfg = RecordConfig(cfg["record"])
    robot = teleop = dataset = None
    timing = invocation = None
    controls = RolloutControls()
    try:
        # Load and compile before connecting to the robot.
        policy = create_policy(inference_config(cfg))
        import numpy as np

        warmup_policy(
            policy,
            {
                "observation/state": np.zeros(19, dtype=np.float32),
                "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                "prompt": record_cfg.task_description,
            },
        )
        robot, teleop = make_hardware(record_cfg)
        features = rollout_features(robot)
        dataset = open_dataset(cfg, features)
        counts = saved_episode_counts(Path(dataset.root))
        policy = URRolloutPolicy(
            policy,
            features["observation.state"]["names"],
            cfg["model"]["network"]["openpi_data"]["obs_indices"],
            record_cfg.task_description,
        )
        if record_cfg.display:
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(session_name="ur-recap-rollout")
        # Episode controls own Escape so the collector can discard and clean up.
        teleop.disconnect_on_esc = False
        teleop.set_input_enabled(False)
        teleop.connect()
        controls.start()
        robot.connect()
        root = Path(dataset.root)
        # Keep each invocation's checkpoint and control settings alongside the data.
        invocation = (
            root
            / "meta"
            / f"rollout_from_episode_{dataset.meta.total_episodes:06d}.json"
        )
        invocation.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
        timing = CollectionTiming(bool(cfg["rollout"].get("record_timing", True)))
        saved = 0
        initial_reset = True
        while saved < record_cfg.num_episodes and not controls.snapshot().stop:
            if initial_reset:
                if not timing.call(
                    "wait",
                    wait_for_enter,
                    "[初始复位] 确认可安全移动后，按回车自动复位",
                    controls,
                ):
                    break
                initial_reset = False
            else:
                if not timing.call(
                    "wait", wait_for_enter, "[下一条] 按回车启用键盘手动复位", controls
                ):
                    break
                timing.call(
                    "reset",
                    manual_reset,
                    robot,
                    teleop,
                    controls,
                    fps=record_cfg.fps,
                    duration=record_cfg.reset_time_sec,
                    display=record_cfg.display,
                    step_sizes=(
                        cfg["rollout"]["reset_teleop"][
                            f"{record_cfg.control_space}_step_size"
                        ],
                        cfg["rollout"]["reset_teleop"][
                            f"{record_cfg.control_space}_rot_step_size"
                        ],
                    ),
                )
            if controls.snapshot().stop:
                break
            timing.call(
                "reset",
                robot.reset_to_init_pose,
                record_cfg.init_pose,
                record_cfg.init_pose_range,
            )
            logger.info(color_status("[复位] 自动复位完成。"))
            logger.info("\n%s", format_collection_summary(counts))
            if not timing.call(
                "wait", wait_for_enter, "[开始采集] 布置好场景后按回车开始", controls
            ):
                break
            if controls.snapshot().stop:
                break
            teleop.set_input_enabled(True)
            controls.begin_episode()
            teleop.reset_step_size()
            robot.set_episode_reference_pose()
            try:
                result = timing.call(
                    "record",
                    record_rollout_episode,
                    robot,
                    teleop,
                    policy,
                    controls,
                    dataset,
                    task=record_cfg.task_description,
                    chunk_size=int(cfg["model"]["network"]["openpi"]["action_chunk"]),
                    fps=record_cfg.fps,
                    max_chunks=int(cfg["rollout"]["max_chunks"]),
                    close_threshold=record_cfg.close_threshold,
                    display=record_cfg.display,
                )
            finally:
                teleop.set_input_enabled(False)
                controls.idle()
                robot.stop_force()
            result = timing.call("review", review_episode, result, controls)
            intervened = (
                any(
                    bool(np.asarray(flag).item())
                    for flag in dataset.episode_buffer["intervene_flag"]
                )
                if not result.discarded
                else False
            )
            timing.call("save", save_episode, dataset, result)
            if result.discarded:
                logger.info(color_status("[采集] 当前轨迹已丢弃，未写入数据集。"))
            else:
                saved += 1
                counts[int(intervened)][int(not result.success)] += 1
                logger.info(
                    color_status("[采集] 已保存 episode %s：success=%s，frames=%s。"),
                    dataset.meta.total_episodes - 1,
                    result.success,
                    result.frames,
                )
            if result.stop:
                break
    finally:
        # Every resource is closed even when a different resource fails cleanup.
        import sys

        failure = sys.exc_info()[1]
        errors = []
        if timing is not None:
            timing_result = timing.finish() if timing.enabled else None
            try:
                write_collection_metadata(invocation, cfg, counts, timing_result)
            except Exception as exc:
                logger.error("Failed to save collection metadata: %s", exc)
                errors.append(exc)
        for close in (
            controls.close,
            (
                lambda: (
                    teleop.disconnect()
                    if teleop is not None and teleop.is_connected
                    else None
                )
            ),
            (lambda: robot.disconnect() if robot is not None else None),
            (
                lambda: (
                    dataset.clear_episode_buffer()
                    if dataset is not None
                    and dataset.episode_buffer
                    and dataset.episode_buffer.get("size", 0)
                    else None
                )
            ),
            (lambda: dataset.finalize() if dataset is not None else None),
        ):
            try:
                close()
            except Exception as exc:
                logger.error("Rollout cleanup failed: %s", exc)
                errors.append(exc)
        if (failure is not None or errors) and dataset is not None:
            from .recording import handle_incomplete_dataset

            logger.warning(
                color_status(
                    "[异常退出] %s；设备清理已执行。下面询问的是整个数据集（包含已保存轨迹）：%s"
                ),
                "Ctrl+C"
                if isinstance(failure, KeyboardInterrupt)
                else str(failure or errors[0]),
                dataset.root,
            )
            try:
                with quiet_keyboard(echo=True):
                    handle_incomplete_dataset(Path(dataset.root))
            except (EOFError, KeyboardInterrupt):
                logger.info(
                    color_status("[保留] 未确认移入回收站，保留数据集：%s"),
                    dataset.root,
                )
            except Exception as exc:
                logger.error(color_status("[回收站] 操作失败，未执行永久删除：%s"), exc)
        if errors and failure is None:
            raise errors[0]


def main() -> None:
    """Run collection or the hardware-free configuration check."""
    configure_console_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config/ur5e_recap_rollout.yaml",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration and imports without connecting hardware.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

        from evaluations.realworld.vla_precision_ur.policy import model_config_from_yaml

        from .recording import RecordConfig
        from .robot import UR5e

        RecordConfig(cfg["record"])
        model_config_from_yaml(inference_config(cfg))
        if CODEBASE_VERSION != "v3.0":
            raise RuntimeError(f"Expected LeRobot v3.0, got {CODEBASE_VERSION}.")
        logger.info(
            "Rollout config/imports passed: %s, robot=%s; hardware not connected.",
            CODEBASE_VERSION,
            UR5e.name,
        )
    else:
        try:
            with quiet_keyboard():
                run(cfg)
        except KeyboardInterrupt:
            logger.info(color_status("[退出] Ctrl+C 处理完成。"))


if __name__ == "__main__":
    main()
