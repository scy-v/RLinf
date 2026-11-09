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

"""Standalone UR rollout contracts using fake hardware and real LeRobot I/O."""

import json

import numpy as np
import pytest

from toolkits.ur_data_collection.controls import RolloutControls
from toolkits.ur_data_collection.rollout_loop import (
    ACTION_NAMES,
    ControlState,
    record_rollout_episode,
    review_episode,
    rollout_features,
    save_episode,
)


class FakeRobot:
    action_features = dict.fromkeys(ACTION_NAMES, float)
    observation_features = {
        **dict.fromkeys([f"state_{i}" for i in range(19)], float),
        "exterior_image": (16, 16, 3),
        "wrist_image": (16, 16, 3),
    }

    def __init__(self):
        self.actions = []
        self._gripper_position = 1.0

    def get_observation(self):
        return {
            **{f"state_{i}": float(len(self.actions)) for i in range(19)},
            "exterior_image": np.full((16, 16, 3), len(self.actions), dtype=np.uint8),
            "wrist_image": np.zeros((16, 16, 3), dtype=np.uint8),
        }

    def send_action(self, action):
        self.actions.append(action.copy())
        return action


class FakeTeleop:
    gripper_action = 1.0

    def __init__(self, robot=None):
        self.robot = robot
        self.intervention_active = False

    def set_input_enabled(self, enabled):
        self.input_enabled = enabled

    def get_action(self):
        self.intervention_active = self.robot is not None and len(
            self.robot.actions
        ) in (1, 2)
        return {
            **dict.fromkeys(ACTION_NAMES[:-1], -0.1),
            "gripper_position": self.gripper_action,
        }


class FakePolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, observation):
        self.calls += 1
        actions = np.zeros((6, 7), dtype=np.float32)
        actions[:, 0] = np.arange(1, 7) / 100
        actions[:, -1] = 1
        return {"actions": actions}


class ScheduledControls:
    def __init__(self, robot, *, outcome=True, outcome_step=3, abort=False):
        self.robot = robot
        self.outcome = outcome
        self.outcome_step = outcome_step
        self.abort = abort

    def snapshot(self):
        step = len(self.robot.actions)
        return ControlState(
            finish_requested=self.outcome is not None and step >= self.outcome_step,
            discard=self.abort and step >= self.outcome_step,
        )


def create_dataset(root, robot, *, video=False):
    module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    if module.CODEBASE_VERSION != "v3.0":
        pytest.skip("UR collection requires the pinned LeRobot v3 runtime.")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = rollout_features(robot)
    if not video:
        for feature in features.values():
            if feature["dtype"] == "video":
                feature["dtype"] = "image"
    return LeRobotDataset.create(
        "test/ur",
        root=root,
        fps=15,
        robot_type="ur5e",
        features=features,
        use_videos=video,
    )


def press_key(controls, key):
    controls.handle(key, True)
    controls.handle(key, False)


def label_result(result, outcome):
    class Operator(RolloutControls):
        def begin_review(self):
            super().begin_review()
            key = "space" if outcome else "f"
            press_key(self, key)
            press_key(self, key)

    return review_episode(result, Operator())


@pytest.mark.parametrize("outcome", [True, False])
def test_ur_rollout_records_actual_actions_and_finishes_chunk(tmp_path, outcome):
    import pyarrow.parquet as pq

    robot, policy = FakeRobot(), FakePolicy()
    dataset = create_dataset(tmp_path / "data", robot)
    result = record_rollout_episode(
        robot,
        FakeTeleop(robot),
        policy,
        ScheduledControls(robot, outcome=outcome),
        dataset,
        task="insert rack",
        chunk_size=6,
        fps=10000,
        max_chunks=5,
    )
    assert result.needs_review
    with pytest.raises(ValueError, match="reviewed outcome"):
        save_episode(dataset, result)
    result = label_result(result, outcome)
    assert result.frames == 6
    assert result.success is outcome
    assert policy.calls == 1
    np.testing.assert_allclose(
        [a["delta_x"] for a in robot.actions], [0.01, -0.1, -0.1, 0.04, 0.05, 0.06]
    )
    save_episode(dataset, result)
    dataset.finalize()
    table = pq.read_table(next((tmp_path / "data/data").rglob("*.parquet"))).to_pydict()
    np.testing.assert_allclose(
        np.asarray(table["action"])[:, 0],
        [a["delta_x"] for a in robot.actions],
    )
    assert table["is_success"] == [outcome] * 6
    assert table["intervene_flag"] == [False, True, True, False, False, False]
    assert [x[0] for x in table["observation.state"]] == list(range(6))


@pytest.mark.parametrize("chunk_size", [3, 6])
def test_ur_rollout_limit_counts_executed_chunks(tmp_path, chunk_size):
    robot, policy = FakeRobot(), FakePolicy()
    dataset = create_dataset(tmp_path / "data", robot)
    result = record_rollout_episode(
        robot,
        FakeTeleop(robot),
        policy,
        ScheduledControls(robot, outcome=None),
        dataset,
        task="insert rack",
        chunk_size=chunk_size,
        fps=10000,
        max_chunks=2,
    )
    assert result.frames == 2 * chunk_size and not result.success
    assert policy.calls == 2
    save_episode(dataset, result)
    dataset.finalize()


def test_ur_rollout_abort_discards_partial_episode(tmp_path):
    robot = FakeRobot()
    dataset = create_dataset(tmp_path / "data", robot)
    result = record_rollout_episode(
        robot,
        FakeTeleop(robot),
        FakePolicy(),
        ScheduledControls(robot, abort=True),
        dataset,
        task="insert rack",
        chunk_size=6,
        fps=10000,
        max_chunks=5,
    )
    assert result.discarded and result.frames == 3
    save_episode(dataset, result)
    assert dataset.meta.total_episodes == 0
    assert dataset.episode_buffer["size"] == 0
    dataset.finalize()


@pytest.mark.parametrize(
    "key, success, discarded",
    [("space", True, False), ("f", False, False), ("d", None, True)],
)
def test_ur_rollout_review_labels_are_phase_scoped(key, success, discarded):
    controls = RolloutControls()
    for begin in (controls.idle, controls.begin_reset, controls.begin_episode):
        begin()
        press_key(controls, key)
        press_key(controls, key)
        assert controls.snapshot().outcome is None
        assert not controls.snapshot().discard
    controls.begin_review()
    press_key(controls, key)
    assert controls.snapshot().outcome is None
    assert not controls.snapshot().discard
    press_key(controls, key)
    assert controls.snapshot().outcome is success
    assert controls.snapshot().discard is discarded
    for other in ("space", "d", "f"):
        press_key(controls, other)
        press_key(controls, other)
    assert controls.snapshot().outcome is success
    assert controls.snapshot().discard is discarded


@pytest.mark.parametrize(
    "key, phase",
    [(".", "rollout"), ("space", "review"), ("f", "review"), ("d", "review")],
)
@pytest.mark.parametrize("interval, accepted", [(0.5, True), (0.501, False)])
def test_ur_rollout_double_press_interval(monkeypatch, key, phase, interval, accepted):
    from toolkits.ur_data_collection import controls as module

    ticks = iter([1.0, 1.0 + interval])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    controls = RolloutControls()
    (controls.begin_episode if phase == "rollout" else controls.begin_review)()
    press_key(controls, key)
    press_key(controls, key)
    state = controls.snapshot()
    assert (
        state.finish_requested or state.outcome is not None or state.discard
    ) is accepted


def test_ur_rollout_keys_do_not_carry_across_phases():
    controls = RolloutControls()
    controls.begin_reset()
    press_key(controls, ".")
    press_key(controls, ".")
    assert not controls.snapshot().finish_requested
    controls.begin_episode()
    controls.handle(".", True)
    controls.handle(".", True)
    assert not controls.snapshot().finish_requested
    controls.handle(".", False)
    press_key(controls, ".")
    assert controls.snapshot().finish_requested
    press_key(controls, "right")
    assert not controls.reset_finished()
    # A rollout Space must never count as half of a review success gesture.
    controls.handle("space", True)
    controls.begin_review()
    controls.handle("space", True)
    assert controls.snapshot().outcome is None
    controls.handle("space", False)
    press_key(controls, "space")
    assert controls.snapshot().outcome is None
    press_key(controls, "space")
    assert controls.snapshot().outcome is True
    controls.begin_reset()
    assert controls.snapshot().outcome is None
    press_key(controls, "right")
    assert controls.reset_finished()
    press_key(controls, "esc")
    controls.begin_episode()
    assert controls.snapshot().stop


def test_ur_rollout_v3_video_returns_and_sft_chunks(tmp_path):
    import pyarrow.parquet as pq

    pytest.importorskip("lerobot.datasets.utils")
    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        process_dataset,
    )

    root = tmp_path / "video_data"
    robot = FakeRobot()
    dataset = create_dataset(root, robot, video=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for outcome in (True, False):
        robot.actions.clear()
        result = record_rollout_episode(
            robot,
            FakeTeleop(robot),
            FakePolicy(),
            ScheduledControls(robot, outcome=outcome),
            dataset,
            task="insert black rack",
            chunk_size=6,
            fps=10000,
            max_chunks=5,
        )
        result = label_result(result, outcome)
        save_episode(dataset, result)
    dataset.finalize()
    info_before = json.loads((root / "meta/info.json").read_text())
    assert info_before["codebase_version"] == "v3.0"
    process_dataset(
        root,
        output_path=None,
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        num_workers=2,
        tag="test",
    )
    table = pq.read_table(root / "meta/returns_test.parquet").to_pydict()
    assert table["return"][:6] == [-5, -4, -3, -2, -1, 0]
    assert table["return"][6:] == [-305, -304, -303, -302, -301, -300]
    assert set(table["prompt"]) == {"insert black rack"}
    assert (
        json.loads((root / "meta/info.json").read_text())["features"]
        == info_before["features"]
    )
    loaded = LeRobotDataset(
        "test/ur", root=root, delta_timestamps={"action": [i / 15 for i in range(20)]}
    )
    item = loaded[4]
    assert item["action"].shape == (20, 7)
    assert item["action_is_pad"].tolist() == [False, False] + [True] * 18
    assert item["observation.images.exterior_image"].shape == (3, 16, 16)


def test_ur_returns_preserve_episodes_across_v3_files(tmp_path):
    pytest.importorskip("lerobot.datasets.utils")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pytest.importorskip("lerobot.datasets.utils")
    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        process_dataset,
    )

    root = tmp_path / "split"
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "insert rack"}\n')
    for file_id, frames in enumerate(([0, 1], [2, 3])):
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0, 0],
                    "frame_index": frames,
                    "is_success": [[False], [False]],
                    "task_index": [0, 0],
                }
            ),
            root / f"data/chunk-000/file-{file_id:03d}.parquet",
        )
    process_dataset(
        root,
        output_path=None,
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        num_workers=2,
        tag="test",
    )
    assert pq.read_table(root / "meta/returns_test.parquet")["return"].to_pylist() == [
        -303,
        -302,
        -301,
        -300,
    ]


def test_ur_rollout_policy_uses_sft_state_order_and_two_images():
    from toolkits.ur_data_collection.policy import URRolloutPolicy

    class Inference:
        def infer(self, obs):
            self.obs = obs
            return {"actions": np.zeros((10, 7))}

    inner = Inference()
    names = [f"state_{i}" for i in range(20)]
    adapter = URRolloutPolicy(inner, names, list(range(2, 21)), "insert rack")
    observation = {name: float(i) for i, name in enumerate(names)}
    observation.update(
        exterior_image=np.zeros((16, 24, 3), dtype=np.uint8),
        wrist_image=np.zeros((16, 24, 3), dtype=np.uint8),
    )
    assert adapter.infer(observation)["actions"].shape == (10, 7)
    np.testing.assert_equal(inner.obs["observation/state"], np.arange(1, 20))
    assert inner.obs["observation/image"].shape == (224, 224, 3)
    assert inner.obs["prompt"] == "insert rack"


def test_ur_rollout_resume_preserves_previous_episodes(tmp_path):
    module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    if module.CODEBASE_VERSION != "v3.0":
        pytest.skip("UR collection requires the pinned LeRobot v3 runtime.")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from toolkits.ur_data_collection.run_rollout import open_dataset

    root = tmp_path / "resume"
    robot = FakeRobot()
    features = rollout_features(robot)
    cfg = {
        "record": {
            "repo_id": "test/ur",
            "fps": 15,
            "storage": {"root": str(root)},
            "task": {"resume": False},
        }
    }
    for resume, success in ((False, True), (True, False)):
        cfg["record"]["task"]["resume"] = resume
        dataset = open_dataset(cfg, features)
        robot.actions.clear()
        result = record_rollout_episode(
            robot,
            FakeTeleop(robot),
            FakePolicy(),
            ScheduledControls(robot, outcome=success),
            dataset,
            task="insert rack",
            chunk_size=6,
            fps=10000,
            max_chunks=1,
        )
        result = label_result(result, success)
        save_episode(dataset, result)
        dataset.finalize()
    loaded = LeRobotDataset("test/ur", root=root)
    assert loaded.meta.total_episodes == 2
    assert len(loaded) == 12
    assert bool(loaded[0]["is_success"])
    assert not bool(loaded[6]["is_success"])


def test_ur_rollout_manual_reset_uses_human_actions_without_recording():
    from toolkits.ur_data_collection.rollout_loop import manual_reset

    class Robot(FakeRobot):
        def set_episode_reference_pose(self):
            self.reference_action_count = len(self.actions)

        def get_observation(self):
            assert self.reference_action_count == 0
            return super().get_observation()

        def stop_force(self):
            self.stopped = True

    class Teleop(FakeTeleop):
        default_step_size = 0.03
        default_rot_step_size = 0.05
        alternate_step_size = 0.006
        alternate_rot_step_size = 0.03

        def reset_step_size(self):
            self.step_size = self.default_step_size
            self.rot_step_size = self.default_rot_step_size

        def get_action(self):
            assert self.step_size == 0.045
            assert self.rot_step_size == 0.075
            assert self.alternate_step_size is None
            return super().get_action()

    class Controls:
        def idle(self):
            pass

        def begin_reset(self):
            pass

        def snapshot(self):
            return ControlState()

        def reset_finished(self):
            return len(robot.actions) == 3

    robot = Robot()
    teleop = Teleop()
    manual_reset(
        robot, teleop, Controls(), fps=10000, duration=5, step_sizes=(0.045, 0.075)
    )
    assert (teleop.step_size, teleop.rot_step_size) == (0.03, 0.05)
    assert (teleop.alternate_step_size, teleop.alternate_rot_step_size) == (0.006, 0.03)
    assert len(robot.actions) == 3
    assert robot.stopped
    assert [a["delta_x"] for a in robot.actions] == [-0.1] * 3


def test_ur_robot_disconnect_cleans_partial_rtde_connection(monkeypatch):
    pytest.importorskip("lerobot.datasets.utils")
    pytest.importorskip("rtde_control")
    pytest.importorskip("pyDHgripper")
    from toolkits.ur_data_collection.robot import ur5e
    from toolkits.ur_data_collection.robot.config_ur5e import UR5eConfig

    class Receiver:
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    receiver = Receiver()
    monkeypatch.setattr(ur5e, "RTDEReceiveInterface", lambda ip: receiver)

    def fail_control(ip):
        raise RuntimeError("controller unavailable")

    monkeypatch.setattr(ur5e, "RTDEControlInterface", fail_control)
    robot = ur5e.UR5e(UR5eConfig(use_gripper=False, cameras={}))
    with pytest.raises(RuntimeError, match="controller unavailable"):
        robot.connect()
    robot.disconnect()
    robot.disconnect()
    assert receiver.disconnected
    assert not robot.is_connected


def test_ur_teleop_motion_keys_do_not_leak_from_review(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("lerobot.teleoperators.teleoperator")
    from toolkits.ur_data_collection.teleop import teleop as module
    from toolkits.ur_data_collection.teleop.config_teleop import UR5eTeleopConfig

    class Listener:
        def is_alive(self):
            return True

    # Replace only the OS keyboard boundary; run the actual motion-key mapping.
    monkeypatch.setattr(module, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(
        module,
        "keyboard",
        SimpleNamespace(
            Key=SimpleNamespace(esc="esc"),
            KeyCode=SimpleNamespace(from_char=lambda value: value),
            Listener=Listener,
        ),
    )
    teleop = module.UR5eTeleop(UR5eTeleopConfig())
    teleop.listener = Listener()
    teleop._on_press("d")
    assert teleop.get_action()["delta_y"] > 0
    assert teleop.intervention_active
    teleop._on_release("d")
    teleop.get_action()
    assert not teleop.intervention_active
    teleop._on_press("/")
    teleop.get_action()
    assert not teleop.intervention_active
    teleop._on_release("/")
    teleop._on_press("o")
    teleop.get_action()
    assert teleop.intervention_active
    teleop._on_release("o")
    teleop._on_press("d")
    teleop._on_press("a")
    assert teleop.get_action()["delta_y"] == 0
    assert teleop.intervention_active
    teleop._on_release("a")
    teleop.set_input_enabled(False)
    teleop._on_release("d")
    teleop._on_press("f")
    teleop._on_release("f")
    teleop._on_press("f")  # Second review press is still held during transition.
    teleop.set_input_enabled(True)
    teleop._on_press("f")  # Auto-repeat must not become a reset motion.
    assert all(teleop.get_action()[name] == 0 for name in ACTION_NAMES[:-1])
    teleop._on_release("f")
    teleop._on_press("f")
    assert teleop.get_action()["delta_ry"] < 0


@pytest.mark.parametrize("key", ["d", "esc"])
def test_ur_rollout_review_discard_never_saves_or_executes_more_actions(tmp_path, key):
    robot = FakeRobot()
    dataset = create_dataset(tmp_path / "data", robot)
    policy = FakePolicy()
    result = record_rollout_episode(
        robot,
        FakeTeleop(robot),
        policy,
        ScheduledControls(robot),
        dataset,
        task="insert rack",
        chunk_size=6,
        fps=10000,
        max_chunks=1,
    )
    assert result.needs_review  # Manual request wins over the simultaneous cap.
    assert dataset.meta.total_episodes == 0

    class Operator(RolloutControls):
        def begin_review(self):
            super().begin_review()
            press_key(self, key)
            press_key(self, key)

    result = review_episode(result, Operator())
    assert result.discarded and not result.needs_review
    assert result.stop == (key == "esc")
    save_episode(dataset, result)
    assert dataset.meta.total_episodes == 0
    assert dataset.episode_buffer["size"] == 0
    assert len(robot.actions) == 6 and policy.calls == 1
    dataset.finalize()


def test_ur_rollout_phase_prompts_and_confirmation(caplog):
    import logging

    caplog.set_level(logging.INFO)
    controls = RolloutControls()
    controls.begin_episode()
    press_key(controls, ".")
    press_key(controls, ".")
    controls.begin_review()
    press_key(controls, "f")
    press_key(controls, "f")
    controls.begin_reset()
    assert "连按两次句号键" in caplog.text
    assert "已收到结束请求" in caplog.text
    assert "空格=成功，F=失败，D=丢弃" in caplog.text
    assert "已选择：失败" in caplog.text
    assert "右方向键进入自动复位" in caplog.text


def test_ur_rollout_timestamped_batches_require_explicit_resume_root(tmp_path):
    module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    if module.CODEBASE_VERSION != "v3.0":
        pytest.skip("UR collection requires the pinned LeRobot v3 runtime.")
    from toolkits.ur_data_collection.run_rollout import open_dataset

    cfg = {
        "record": {
            "repo_id": "test/ur",
            "fps": 15,
            "storage": {"root": None, "output_dir": str(tmp_path / "round_001")},
            "task": {"resume": False},
        }
    }
    roots = []
    for _ in range(2):
        dataset = open_dataset(cfg, rollout_features(FakeRobot()))
        roots.append(dataset.root)
        assert (dataset.root / "meta/info.json").is_file()
        dataset.finalize()
    assert roots[0] != roots[1]
    assert all(root.parent == tmp_path / "round_001" for root in roots)
    assert cfg["record"]["storage"]["root"] is None
    cfg["record"]["task"]["resume"] = True
    with pytest.raises(ValueError, match="Resuming requires"):
        open_dataset(cfg, rollout_features(FakeRobot()))


def test_ur_terminal_wait_escape_and_restore_echo(monkeypatch):
    import os
    import pty
    import termios

    from toolkits.ur_data_collection import terminal

    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, "r") as stream:
            monkeypatch.setattr(terminal.sys, "stdin", stream)
            original = termios.tcgetattr(stream)
            controls = RolloutControls()
            with pytest.raises(KeyboardInterrupt), terminal.quiet_keyboard():
                assert not termios.tcgetattr(stream)[3] & termios.ECHO
                with pytest.raises(EOFError), terminal.quiet_keyboard(echo=True):
                    assert termios.tcgetattr(stream)[3] & termios.ECHO
                    raise EOFError
                assert not termios.tcgetattr(stream)[3] & termios.ECHO
                controls.handle("esc", True)
                assert terminal.wait_for_enter("等待复位", controls) is False
                raise KeyboardInterrupt
            assert termios.tcgetattr(stream) == original
    finally:
        os.close(master)


def test_ur_terminal_wait_accepts_enter(monkeypatch):
    import os

    from toolkits.ur_data_collection import terminal

    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "r") as stream:
            monkeypatch.setattr(terminal.sys, "stdin", stream)
            os.write(write_fd, b"\n")
            assert terminal.wait_for_enter("开始采集", RolloutControls())
    finally:
        os.close(write_fd)


def test_ur_terminal_colors_only_status_titles(monkeypatch):
    from toolkits.ur_data_collection import terminal

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(terminal.sys.stderr, "isatty", lambda: True)
    message = "[采集] 已保存 episode 1"
    assert (
        terminal.color_status(message) == "\033[38;5;117m[采集]\033[0m 已保存 episode 1"
    )
    assert terminal.color_status("普通消息") == "普通消息"
    monkeypatch.setenv("NO_COLOR", "1")
    assert terminal.color_status(message) == message
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setattr(terminal.sys.stderr, "isatty", lambda: False)
    assert terminal.color_status(message) == message


@pytest.mark.parametrize("answer,removed", [("y", True), ("n", False), ("", False)])
def test_ur_abnormal_exit_dataset_trash_requires_confirmation(
    tmp_path, monkeypatch, answer, removed
):
    from toolkits.ur_data_collection import recording

    root = tmp_path / "dataset"
    root.mkdir()
    (root / "saved_episode").write_text("preserve until confirmed")
    calls = []
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    # The OS trash service is the boundary; never trash real user data in tests.
    monkeypatch.setattr(recording, "send2trash", lambda path: calls.append(path))
    keep = recording.handle_incomplete_dataset(root)
    assert keep is not removed
    assert calls == ([str(root)] if removed else [])
    assert (root / "saved_episode").read_text() == "preserve until confirmed"


@pytest.mark.parametrize("threshold", [0.5, 0.75])
def test_ur_rollout_binarizes_gripper_before_dispatch_and_recording(
    tmp_path, threshold
):
    import pyarrow.parquet as pq

    class Policy(FakePolicy):
        def infer(self, observation):
            result = super().infer(observation)
            result["actions"] = result["actions"].astype(np.float64)
            result["actions"][:, -1] = [
                -0.03,
                threshold - 0.01,
                threshold,
                0.8,
                1.01,
                0.1,
            ]
            return result

    robot = FakeRobot()
    dataset = create_dataset(tmp_path / "binary", robot)
    result = record_rollout_episode(
        robot,
        FakeTeleop(),
        Policy(),
        ScheduledControls(robot, outcome=None),
        dataset,
        task="insert rack",
        chunk_size=6,
        fps=10000,
        max_chunks=1,
        close_threshold=threshold,
    )
    expected = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert [a["gripper_position"] for a in robot.actions] == expected
    save_episode(dataset, result)
    dataset.finalize()
    table = pq.read_table(next((tmp_path / "binary/data").rglob("*.parquet")))
    assert np.array(table["action"].to_pylist())[:, -1].tolist() == expected


@pytest.mark.parametrize(
    "counts", [[[0, 0], [0, 0]], [[1, 1], [1, 0]], [[999, 12], [1000, 0]]]
)
def test_ur_collection_summary_alignment(counts, monkeypatch):
    from wcwidth import wcswidth

    from toolkits.ur_data_collection.terminal import format_collection_summary

    monkeypatch.setenv("NO_COLOR", "1")
    lines = format_collection_summary(counts).splitlines()
    assert len({wcswidth(line) for line in lines[1:]}) == 1
    assert str(sum(map(sum, counts))) in lines[-2]
    assert "自主 rollout" in lines[4]
    assert "人工干预" in lines[5]


@pytest.mark.parametrize("scalar", [False, True])
def test_ur_saved_episode_counts_across_files(tmp_path, scalar):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from toolkits.ur_data_collection.run_rollout import saved_episode_counts

    assert saved_episode_counts(tmp_path) == [[0, 0], [0, 0]]
    data = tmp_path / "data/chunk-000"
    data.mkdir(parents=True)
    rows = [
        {"episode_index": 0, "is_success": [True], "intervene_flag": [False]},
        {"episode_index": 1, "is_success": [False], "intervene_flag": [False]},
        {"episode_index": 2, "is_success": [True], "intervene_flag": [False]},
        {"episode_index": 2, "is_success": [True], "intervene_flag": [True]},
        {"episode_index": 2, "is_success": [True], "intervene_flag": [False]},
    ]
    if scalar:
        for row in rows:
            row["is_success"] = row["is_success"][0]
            row["intervene_flag"] = row["intervene_flag"][0]
    pq.write_table(pa.Table.from_pylist(rows[:3]), data / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[3:]), data / "file-001.parquet")
    assert saved_episode_counts(tmp_path) == [[1, 1], [1, 0]]


@pytest.mark.parametrize("enabled", [True, False])
def test_ur_collection_timing_interruption_and_disabled(enabled):
    import time
    from datetime import datetime

    from toolkits.ur_data_collection.run_rollout import CollectionTiming

    timing = CollectionTiming(enabled)

    def interrupted():
        time.sleep(0.001)
        raise KeyboardInterrupt

    assert timing.call("wait", lambda: 42) == 42
    with pytest.raises(KeyboardInterrupt):
        timing.call("record", interrupted)
    result = timing.finish()
    assert datetime.fromisoformat(result["started_at"]).tzinfo is not None
    assert datetime.fromisoformat(result["ended_at"]).tzinfo is not None
    assert timing.seconds["record"] > 0 if enabled else timing.seconds["record"] == 0
    assert result["record_duration"].endswith("s")
    assert result["total_duration"].endswith("s")


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0s"),
        (22.2, "22s"),
        (59.9, "1min 0s"),
        (2005, "33min 25s"),
        (8091.35, "134min 51s"),
    ],
)
def test_ur_elapsed_duration_format(seconds, expected):
    from toolkits.ur_data_collection.run_rollout import format_elapsed

    assert format_elapsed(seconds) == expected


@pytest.mark.parametrize(
    "timing",
    [None, {"started_at": "2026-11-09T19:04:15+08:00", "total_duration": "134min 51s"}],
)
def test_ur_collection_metadata_persists_counts_with_optional_timing(tmp_path, timing):
    from toolkits.ur_data_collection.run_rollout import write_collection_metadata

    path = tmp_path / "rollout_from_episode_000000.json"
    cfg = {"rollout": {"record_timing": timing is not None}}
    path.write_text(json.dumps(cfg))
    write_collection_metadata(path, cfg, [[15, 28], [52, 5]], timing)
    result = json.loads(path.read_text())
    assert result["collection_summary"]["autonomous"] == {
        "success": 15,
        "failure": 28,
        "total": 43,
    }
    assert result["collection_summary"]["intervened"] == {
        "success": 52,
        "failure": 5,
        "total": 57,
    }
    assert result["collection_summary"]["total"] == {
        "success": 67,
        "failure": 33,
        "total": 100,
    }
    assert result.get("timing") == timing
    assert ("timing" in result) == (timing is not None)
    assert cfg == {"rollout": {"record_timing": timing is not None}}
