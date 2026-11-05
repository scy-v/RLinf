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

"""Shared helpers: metrics, entropy aggregation, checkpoint paths, and resume."""

from __future__ import annotations

import importlib.util
import io
import math
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.utils import compute_entropy_loss
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.utils.metric_utils import compute_evaluate_metrics, compute_rollout_metrics


@pytest.mark.parametrize(
    "previous, current, save, ended",
    [
        (2490, 2520, True, False),
        (2500, 2500, False, False),
        (2490, 2499, False, False),
        (7990, 8050, True, True),
    ],
)
def test_check_progress_crossed_training_rounds(previous, current, save, ended):
    from rlinf.utils.runner_utils import check_progress

    validate, save_model, is_end = check_progress(
        current, 8000, -1, 2500, 1.0, previous_step=previous
    )
    assert not validate
    assert save_model == save
    assert is_end == ended


def test_check_progress_preserves_call_based_contract():
    from rlinf.utils.runner_utils import check_progress

    assert check_progress(2500, 8000, -1, 2500, 1.0) == (False, True, False)
    assert check_progress(2520, 8000, -1, 2500, 1.0) == (False, False, False)


@pytest.mark.parametrize("fail", [False, True])
def test_startup_timing_nested_stages_and_failure_restore_scope(fail):
    from rlinf.utils.startup_timing import StartupTiming, startup_section

    records, barriers = [], []

    def record(event, **fields):
        records.append({"event": event, **fields})

    def operation():
        with StartupTiming(
            "load",
            lambda message: None,
            record=record,
            synchronize=lambda: barriers.append(True),
        ):
            with startup_section("outer"):
                with startup_section("inner"):
                    torch.ones(4).sum()
                if fail:
                    raise ValueError("load failed")

    if fail:
        with pytest.raises(ValueError, match="load failed"):
            operation()
    else:
        operation()
    stages = [row for row in records if row["event"] == "startup_stage"]
    assert [row["stage"] for row in stages] == ["outer/inner", "outer"]
    assert [row["depth"] for row in stages] == [1, 0]
    assert stages[1]["duration_sec"] >= stages[0]["duration_sec"]
    assert stages[1]["status"] == ("failed" if fail else "complete")
    assert len(barriers) == (1 if fail else 2)
    summary = next(row for row in records if row["event"] == "startup_summary")
    assert summary["duration_sec"] >= stages[1]["duration_sec"]
    assert summary["unattributed_sec"] >= 0
    assert summary["major_faults"] >= 0
    assert summary["minor_faults"] >= 0
    count = len(records)
    with startup_section("ordinary_rollout"):
        pass
    assert len(records) == count


def test_disabled_startup_timing_does_not_probe_or_synchronize(monkeypatch):
    import resource

    from rlinf.utils.startup_timing import StartupTiming, startup_section

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled startup timing performed work")

    monkeypatch.setattr(resource, "getrusage", forbidden)
    with StartupTiming(
        "load", forbidden, record=forbidden, synchronize=forbidden, enabled=False
    ):
        with startup_section("model"):
            pass


def test_startup_timing_reports_compiler_work_without_resetting_cache():
    from rlinf.utils.startup_timing import StartupTiming, startup_section

    records = []

    def record(event, **fields):
        records.append({"event": event, **fields})

    def forward(value):
        return value.sin() + 0.1234567

    compiled = torch.compile(forward, backend="eager")
    x = torch.ones(5)
    with StartupTiming("warmup", lambda message: None, record=record):
        with startup_section("sample"):
            actual = compiled(x)
    torch.testing.assert_close(actual, forward(x))
    assert any(row["event"] == "startup_compile" for row in records)
    records.clear()
    with StartupTiming("warmup", lambda message: None, record=record):
        compiled(x)
    assert not any(row["event"] == "startup_compile" for row in records)


@pytest.mark.parametrize("color", [False, True])
def test_stage_log_keeps_plain_file_and_colors_terminal(tmp_path, color):
    from rlinf.utils.progress import stream_run_log

    source = io.StringIO(
        "\033[36m[LOAD] Loading checkpoint: 1s\033[0m\n[RLT] Press Enter to start.\n"
        "UserWarning: example warning\nRuntimeError: example error\nordinary line\n"
    )
    terminal = io.StringIO()
    log = tmp_path / "run.log"
    stream_run_log(source, terminal, str(log), color=color)
    plain = log.read_text()
    assert "\033" not in plain
    assert (
        plain == "[LOAD] Loading checkpoint: 1s\n[RLT] Press Enter to start.\n"
        "UserWarning: example warning\nRuntimeError: example error\nordinary line\n"
    )
    if color:
        assert "\033[1;33m[RLT] Press Enter" in terminal.getvalue()
        assert "\033[1;36m[LOAD]" in terminal.getvalue()
        assert "\033[38;5;174mUserWarning:" in terminal.getvalue()
        assert "\033[1;31mRuntimeError:" in terminal.getvalue()
    else:
        assert terminal.getvalue() == plain


@pytest.mark.parametrize("interactive", [False, True])
def test_compact_stage_log_updates_without_scrolling_and_keeps_details(
    tmp_path, interactive
):
    from rlinf.utils.progress import stream_run_log

    lines = ["{large config here}\n", "[INFO] FlexiblePlacementStrategy details\n"]
    lines += [
        f"(Rollout pid=123) [LOAD] Loading model: {i}.0s elapsed\n" for i in range(20)
    ]
    lines += ["[LOAD] Loading model: Completed after 20.0s\n"]
    lines += ["[COLLECT] Waiting: 12/150\n"] * 10
    lines += [
        "[RLT] Reset complete. Press Enter to start.\n",
        "[RLT] Critical phase selected.\n",
    ]
    lines += [
        "Traceback (most recent call last):\n",
        "  File example.py, line 1\n",
        "ValueError: bad input\n",
    ]
    source = "".join(lines)
    terminal = io.StringIO()
    path = tmp_path / "compact.log"
    stream_run_log(
        io.StringIO(source), terminal, str(path), compact=True, interactive=interactive
    )
    assert path.read_text() == source
    output = terminal.getvalue()
    assert "large config" not in output and "FlexiblePlacement" not in output
    assert output.count("\n") <= 8
    assert "Completed after 20.0s" in output
    assert "Press Enter" in output and "Critical phase selected" in output
    assert "File example.py" in output and "ValueError: bad input" in output
    if interactive:
        assert "\r" in output and "19.0s elapsed" in output
    else:
        assert "19.0s elapsed" not in output


@pytest.mark.parametrize("phase", ["LOAD", "WARMUP"])
@pytest.mark.parametrize("fail", [False, True])
def test_elapsed_time_reports_blocking_work_and_joins_on_exit(phase, fail, monkeypatch):
    from rlinf.utils.progress import ElapsedTime

    tick = threading.Event()

    class Stream(io.StringIO):
        def write(self, value):
            if threading.current_thread().name == f"ur-{phase.lower()}-timer":
                tick.set()
            return super().write(value)

    stream = Stream()
    monkeypatch.setattr(sys, "stderr", stream)

    def operation():
        with ElapsedTime(phase, "Model setup") as timer:
            assert tick.wait(3), "Progress was not displayed while the operation waited"
            if fail:
                raise ValueError("setup failed")
        return timer

    if fail:
        with pytest.raises(ValueError, match="setup failed"):
            operation()
    else:
        assert operation().duration >= 1.0
    assert stream.getvalue().count("\n") == 1
    assert ("Stopped before completion" if fail else "Completed") in stream.getvalue()
    assert not any(t.name == f"ur-{phase.lower()}-timer" for t in threading.enumerate())


def test_ur_run_logs_publish_matching_latest_files(tmp_path):
    import json

    import yaml

    from rlinf.utils.ur_logging import append_record, prepare_run_logs

    for run_id in ("first", "second"):
        yaml_path, timing_path = prepare_run_logs(tmp_path, run_id)
        prediction = {
            "episode_index": 1,
            "infer_index": 7,
            "state": [0.1, -0.2, 0.3],
            "actions": [[0.0] * 7],
        }
        append_record(yaml_path, prediction)
        append_record(timing_path, {"event": "step", "infer_index": 7})
        assert (tmp_path / "latest.yaml").resolve() == yaml_path
        assert (tmp_path / "latest.jsonl").resolve() == timing_path
        yaml_text = (tmp_path / "latest.yaml").read_text()
        record = yaml.safe_load(yaml_text.split(" " + "-" * 60)[0])
        assert "state: [0.1, -0.2, 0.3]" in yaml_text
        assert "actions:\n- [" in yaml_text
        assert record == prediction
        assert list(record) == list(prediction)
        timing = json.loads((tmp_path / "latest.jsonl").read_text())
        assert record["infer_index"] == timing["infer_index"] == 7
    assert (tmp_path / "all_logs" / "log_first.yaml").exists()


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


@pytest.fixture
def single_rank_reduction(monkeypatch):
    from rlinf.scheduler.worker.worker import Worker

    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


def test_compute_rollout_metrics_reports_loss_mask_fraction(single_rank_reduction):
    metrics = compute_rollout_metrics(
        {
            "loss_mask": torch.tensor([[[True], [False]], [[True], [True]]]),
            "rewards": torch.tensor([[[1.0], [8.0]], [[2.0], [3.0]]]),
        }
    )

    assert metrics["loss_mask_fraction"] == pytest.approx(0.75)
    assert metrics["rewards"] == pytest.approx(2.0)


def test_compute_rollout_metrics_omits_loss_mask_fraction_without_mask(
    single_rank_reduction,
):
    metrics = compute_rollout_metrics({"rewards": torch.tensor([[[1.0], [3.0]]])})

    assert "loss_mask_fraction" not in metrics
    assert metrics["rewards"] == pytest.approx(2.0)


# The embodied actor's entropy bonus aggregation. The shapes below are the ones
# the shipped models actually emit: openpi, lingbotvla,
# dexbotic_pi, dexbotic_dm0 and flow_policy all reduce entropy to [bsz, 1],
# cnn_policy returns [bsz, action_dim], openvla_oft returns [bsz, seq_len], and
# the StarVLA action heads return [bsz, num_action_chunks, action_dim].
# loss_mask is [bsz, 1] under reward_type: chunk_level and
# [bsz, num_action_chunks] otherwise.


def _entropy(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=generator) + 0.5


@pytest.mark.parametrize("batch_size", [4, 16, 64, 500])
def test_chunk_level_entropy_does_not_scale_with_batch_size(batch_size):
    """The bug signature: entropy_loss came out multiplied by the micro-batch size."""
    entropy = _entropy(batch_size, 1)
    loss_mask = torch.ones(batch_size, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_chunk_level_entropy_averages_only_the_valid_rows():
    entropy = _entropy(16, 1)
    loss_mask = torch.zeros(16, 1, dtype=torch.bool)
    loss_mask[:6] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:6].mean()), rel=1e-6)


def test_chunk_level_sums_a_wide_entropy_before_averaging():
    # cnn_policy shape: one entropy per action dimension.
    entropy = _entropy(12, 4)
    loss_mask = torch.zeros(12, 1, dtype=torch.bool)
    loss_mask[:5] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:5].sum(dim=-1).mean()), rel=1e-6)


def test_token_level_averages_over_every_valid_element():
    # openvla_oft shape: entropy per token, mask per chunk step.
    entropy = _entropy(10, 7)
    loss_mask = torch.zeros(10, 1, dtype=torch.bool)
    loss_mask[:4] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:4].mean()), rel=1e-6)


def test_action_level_sums_action_dim_then_averages():
    entropy = _entropy(6, 3 * 7)
    loss_mask = torch.zeros(6, 1, dtype=torch.bool)
    loss_mask[:2] = True

    got = compute_entropy_loss(
        entropy, "action_level", loss_mask, action_dim=7, batch_size=6
    )

    per_chunk = entropy.reshape(6, 3, 7).sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:2].mean()), rel=1e-6)


def test_a_wider_mask_than_entropy_still_weights_by_valid_steps():
    # lingbotvla: entropy is [bsz, 1] while reward_type != chunk_level keeps the
    # mask at [bsz, num_action_chunks]. Each sample is weighted by its valid steps.
    entropy = _entropy(5, 1)
    loss_mask = torch.zeros(5, 4, dtype=torch.bool)
    loss_mask[0, :4] = True
    loss_mask[1, :1] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    expected = (entropy[0, 0] * 4 + entropy[1, 0] * 1) / 5
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_three_dim_entropy_reduces_to_the_mask_rank():
    # StarVLA action heads return [bsz, num_action_chunks, action_dim]; the
    # chunk_level sum already lands on the mask's rank, so nothing is unsqueezed.
    entropy = _entropy(4, 8, 7)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:, :3].mean()), rel=1e-6)


def test_three_dim_entropy_is_right_when_batch_equals_num_chunks():
    # Same shape family with bsz == num_action_chunks, where a rank mismatch
    # broadcasts successfully instead of raising and would go unnoticed.
    entropy = _entropy(8, 8, 7)
    loss_mask = torch.zeros(8, 8, dtype=torch.bool)
    loss_mask[:3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:3].mean()), rel=1e-6)


def test_no_mask_averages_everything():
    entropy = _entropy(9, 1)

    got = compute_entropy_loss(entropy, "chunk_level", None)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_a_fully_masked_batch_contributes_zero():
    entropy = _entropy(8, 1)
    loss_mask = torch.zeros(8, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(0.0)


def test_entropy_loss_keeps_the_gradient_path():
    entropy = _entropy(8, 1).requires_grad_(True)
    loss_mask = torch.ones(8, 1, dtype=torch.bool)

    compute_entropy_loss(entropy, "chunk_level", loss_mask).backward()

    # A correct mean spreads 1/8 of the gradient onto each row; the outer-product
    # bug put 1.0 on each instead.
    assert torch.allclose(entropy.grad, torch.full((8, 1), 1 / 8))


def test_entropy_loss_rejects_a_model_that_computes_no_entropy():
    # gr00t, abot_m0 and evo1 return entropy=None; pairing one with a non-zero
    # entropy_bonus is a config error, not a zero bonus.
    with pytest.raises(ValueError, match="algorithm.entropy_bonus"):
        compute_entropy_loss(None, "chunk_level", torch.ones(4, 1, dtype=torch.bool))


def _load_checkpoint_utils():
    module_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "checkpoint.py"
    )
    assert module_path.exists(), "checkpoint path utilities are not implemented"
    spec = importlib.util.spec_from_file_location(
        "_rlinf_utils_checkpoint_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/global_step_30",
        "/tmp/checkpoints/global_step_30/",
        "/tmp/checkpoints/global_step_30///",
    ],
)
def test_parse_global_step_accepts_trailing_slashes(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    assert (
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path) == 30
    )


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/step_30",
        "/tmp/checkpoints/global_step_latest/",
        "/tmp/checkpoints/global_step_30/actor",
    ],
)
def test_parse_global_step_rejects_invalid_checkpoint_directories(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    with pytest.raises(ValueError, match="global_step_<step>"):
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path)


class _StubRunner:
    """Expose only the checkpoint helpers and state used by these tests."""

    def __init__(self, critic=None):
        self.critic = critic

    _is_complete_checkpoint = ReasoningRunner._is_complete_checkpoint


class _ImmediateHandle:
    def wait(self):
        return None


class _Actor:
    def save_checkpoint(self, path: str, _step: int):
        os.makedirs(path, exist_ok=True)
        return _ImmediateHandle()


class _Dataloader:
    def state_dict(self):
        return {"offset": 3}


def _write_checkpoint(
    root: Path, step: int, *, complete: bool, with_critic: bool = False
) -> Path:
    checkpoint_dir = root / f"global_step_{step}"
    (checkpoint_dir / "actor").mkdir(parents=True)
    if with_critic:
        (checkpoint_dir / "critic").mkdir()
    if complete:
        data_dir = checkpoint_dir / "data"
        data_dir.mkdir()
        (data_dir / "data.pt").write_bytes(b"dataloader-state")
    return checkpoint_dir


def _resolve_auto_resume(log_path: Path, *, critic=None) -> str | None:
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": "auto", "logger": {"log_path": str(log_path)}}}
    )
    runner = _StubRunner(critic=critic)
    runner.cfg = cfg
    runner.init_rollout_workers = lambda: None
    runner.init_actor_critic_workers = lambda: None

    ReasoningRunner.init_workers(runner)
    return cfg.runner.resume_dir


def _saving_runner(tmp_path: Path) -> _StubRunner:
    runner = _StubRunner()
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "output_dir": str(tmp_path),
                "experiment_name": "experiment",
            }
        }
    )
    runner.global_steps = 8
    runner.actor = _Actor()
    runner.train_dataloader = _Dataloader()
    return runner


@pytest.mark.parametrize(
    "completeness,expected_step",
    [
        pytest.param({40: True, 80: False}, 40, id="skips-the-incomplete-newest"),
        pytest.param({40: True, 80: True}, 80, id="takes-the-newest-complete"),
        pytest.param({40: False}, None, id="starts-fresh-when-none-is-complete"),
    ],
)
def test_auto_resume_selects_the_newest_complete_checkpoint(
    tmp_path, completeness, expected_step
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for step, complete in completeness.items():
        _write_checkpoint(checkpoints_dir, step, complete=complete)

    expected = (
        None
        if expected_step is None
        else str(checkpoints_dir / f"global_step_{expected_step}")
    )
    assert _resolve_auto_resume(tmp_path) == expected


def test_checkpoint_requires_the_critic_only_when_configured(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint = _write_checkpoint(checkpoints_dir, 40, complete=True)

    assert _StubRunner()._is_complete_checkpoint(str(checkpoint))
    assert not _StubRunner(critic=object())._is_complete_checkpoint(str(checkpoint))


def test_dataloader_state_is_published_atomically(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)
    written_paths = []

    def save(_state, path):
        written_paths.append(path)
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", save)

    ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert written_paths == [f"{final_path}.tmp"]
    assert final_path.read_bytes() == b"complete"
    assert not Path(f"{final_path}.tmp").exists()
    assert runner._is_complete_checkpoint(str(checkpoint))


def test_interrupted_dataloader_save_does_not_publish_completion(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)

    def interrupted_save(_state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", interrupted_save)

    with pytest.raises(RuntimeError, match="interrupted"):
        ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert not final_path.exists()
    assert not Path(f"{final_path}.tmp").exists()
    assert not runner._is_complete_checkpoint(str(checkpoint))


_REDIRECTED_ENTRYPOINT = textwrap.dedent(
    """
    import hydra

    from rlinf.scheduler import Cluster
    from rlinf.utils.utils import output_redirector


    @hydra.main(version_base="1.1", config_path=None)
    @output_redirector
    def main(cfg):
        Cluster(num_nodes=1)
        print("entrypoint ran")
        if cfg.outcome == "raise":
            raise RuntimeError("entrypoint failed")


    main()
    """
)


def _run_redirected_entrypoint(tmp_path, outcome):
    script = tmp_path / "entrypoint.py"
    script.write_text(_REDIRECTED_ENTRYPOINT)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            f"+outcome={outcome}",
            f"+runner.output_dir={tmp_path}",
            "+runner.experiment_name=exp",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_redirected_entrypoint_exits_zero_and_keeps_its_log(tmp_path):
    result = _run_redirected_entrypoint(tmp_path, "return")

    assert result.returncode == 0, result.stderr
    assert "entrypoint ran" in (tmp_path / "exp" / "log" / "main.log").read_text()


def test_redirected_entrypoint_failure_is_not_reported_as_success(tmp_path):
    # Hydra catches the exception and calls sys.exit(1) itself, so the failure
    # never reaches sys.excepthook.
    result = _run_redirected_entrypoint(tmp_path, "raise")

    assert result.returncode == 1, result.stderr
    assert "entrypoint failed" in (tmp_path / "exp" / "log" / "main.log").read_text()


def _flash_availability(monkeypatch, **installed):
    """Pretend transformers reports exactly the given flash variants."""
    import transformers.utils as transformers_utils

    for version in (2, 3, 4):
        name = f"is_flash_attn_{version}_available"
        monkeypatch.setattr(
            transformers_utils,
            name,
            lambda version=version: installed.get(f"fa{version}", False),
            raising=False,
        )


def test_attn_implementation_keeps_the_preferred_variant(monkeypatch):
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch, fa2=True)
    assert resolve_attn_implementation() == "flash_attention_2"


def test_attn_implementation_falls_back_to_the_installed_variant(monkeypatch):
    """An sm90+ image ships FA4 instead of FA2, so FA2 must not be requested."""
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch, fa4=True)
    assert resolve_attn_implementation("flash_attention_2") == "flash_attention_4"


def test_attn_implementation_falls_back_to_sdpa_without_flash_attn(monkeypatch):
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch)
    assert resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_attn_implementation_survives_a_probe_that_raises(monkeypatch):
    """A probe can import kernels and fail on platforms without flash-attn."""
    import transformers.utils as transformers_utils

    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch)

    def explode():
        raise RuntimeError("no flash-attn kernels for this platform")

    monkeypatch.setattr(
        transformers_utils, "is_flash_attn_2_available", explode, raising=False
    )
    assert resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_attn_implementation_passes_through_non_flash_choices(monkeypatch):
    from rlinf.utils.attention import resolve_attn_implementation

    _flash_availability(monkeypatch)
    assert resolve_attn_implementation("eager") == "eager"


@pytest.mark.parametrize("exit_kind", ["normal", "error", "interrupt", "sigterm"])
def test_robot_process_exit_runs_cleanup_before_termination(tmp_path, exit_kind):
    marker = tmp_path / "closed"
    code = textwrap.dedent("""\
        import os
        import signal
        from pathlib import Path
        from rlinf.utils.robot_lifecycle import register_robot_cleanup
        marker = Path(os.environ["TEST_ROBOT_CLEANUP_MARKER"])
        register_robot_cleanup(lambda: marker.write_text("stopped"))
        exit_kind = os.environ["TEST_ROBOT_EXIT_KIND"]
        if exit_kind == "error":
            raise ValueError("test exit")
        if exit_kind == "interrupt":
            os.kill(os.getpid(), signal.SIGINT)
        if exit_kind == "sigterm":
            os.kill(os.getpid(), signal.SIGTERM)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "TEST_ROBOT_CLEANUP_MARKER": str(marker),
            "TEST_ROBOT_EXIT_KIND": exit_kind,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert marker.read_text() == "stopped", result.stderr
    assert (result.returncode == 0) == (exit_kind == "normal")


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM", "SIGUSR1"])
def test_robot_entry_signals_run_finally_before_worker_shutdown(tmp_path, signal_name):
    marker = tmp_path / "closed"
    code = textwrap.dedent("""\
        import os
        import signal
        from contextlib import ExitStack
        from pathlib import Path
        from rlinf.utils.robot_lifecycle import termination_signals
        marker = Path(os.environ["TEST_ROBOT_CLEANUP_MARKER"])
        # Emulate Cluster's handler that otherwise kills workers immediately.
        signal.signal(signal.SIGUSR1, lambda *_: os._exit(99))
        with termination_signals(), ExitStack() as cleanup:
            cleanup.callback(lambda: marker.write_text("stopped"))
            os.kill(os.getpid(), getattr(signal, os.environ["TEST_ROBOT_EXIT_SIGNAL"]))
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "TEST_ROBOT_CLEANUP_MARKER": str(marker),
            "TEST_ROBOT_EXIT_SIGNAL": signal_name,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert marker.read_text() == "stopped", result.stderr
    assert result.returncode not in (0, 99)


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM"])
def test_worker_exit_closes_robots_registered_on_background_thread(
    tmp_path, signal_name
):
    marker = tmp_path / "closed"
    code = textwrap.dedent("""\
        import os
        import signal
        import threading
        from pathlib import Path
        from rlinf.utils.robot_lifecycle import (
            install_robot_cleanup_handlers,
            register_robot_cleanup,
        )
        # Worker can install a handler that exits immediately after printing a trace.
        selected = getattr(signal, os.environ["TEST_ROBOT_EXIT_SIGNAL"])
        signal.signal(selected, lambda *_: os._exit(91))
        install_robot_cleanup_handlers()
        marker = Path(os.environ["TEST_ROBOT_CLEANUP_MARKER"])
        thread = threading.Thread(
            target=lambda: register_robot_cleanup(lambda: marker.write_text("stopped"))
        )
        thread.start()
        thread.join()
        os.kill(os.getpid(), selected)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "TEST_ROBOT_CLEANUP_MARKER": str(marker),
            "TEST_ROBOT_EXIT_SIGNAL": signal_name,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 91, result.stderr
    assert marker.read_text() == "stopped", result.stderr


@pytest.mark.parametrize("resume", [False, True])
def test_embodied_runner_defers_environment_until_models_ready(tmp_path, resume):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    calls = []

    class RemoteGroup:
        def __init__(self, name):
            self.name = name

        def init_worker(self):
            calls.append(self.name + ".init")
            return SimpleNamespace(wait=lambda: calls.append(self.name + ".ready"))

        def load_checkpoint(self, path):
            calls.append(self.name + ".load")
            return SimpleNamespace(wait=lambda: calls.append(self.name + ".restored"))

    checkpoint = tmp_path / "global_step_1000"
    (checkpoint / "actor").mkdir(parents=True)
    runner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "runner": {
                    "env_init_after_models": True,
                    "resume_dir": str(checkpoint) if resume else None,
                }
            }
        ),
        rollout=RemoteGroup("rollout"),
        actor=RemoteGroup("actor"),
        env=RemoteGroup("env"),
        reward=None,
        logger=__import__("logging").getLogger(__name__),
    )
    EmbodiedRunner.init_workers(runner)
    assert calls.index("rollout.ready") < calls.index("env.init")
    assert calls.index("actor.ready") < calls.index("env.init")
    if resume:
        assert calls.index("actor.restored") < calls.index("env.init")


@pytest.mark.parametrize("interactive", [False, True])
def test_training_progress_completes_and_preserves_timing_in_file(
    tmp_path, interactive
):
    from rlinf.utils.progress import TrainingProgress, stream_run_log

    records = []
    progress = TrainingProgress("Warmup updates", 150, records.append, initial=148)
    progress.update(148, 160, force=True)
    progress.update(149, 180, force=True)
    progress.update(150, 190)
    progress.update(150, 190)
    assert sum("Completed" in record for record in records) == 1
    records += [
        "[BUFFER] Critical chunks: 190",
        "[TIMING] Learner update: work_ms=12.00",
    ]
    records += [
        "[BUFFER] Critical chunks: 210",
        "[TIMING] Learner update: work_ms=14.00",
        "[PD] UR background: measured_hz=99.80",
    ]
    records += ["[RLT] Reset complete. Press Enter to start."]
    source = "\n".join(records) + "\n"
    terminal = io.StringIO()
    path = tmp_path / "training.log"
    stream_run_log(
        io.StringIO(source),
        terminal,
        str(path),
        compact=True,
        color=True,
        interactive=interactive,
    )
    assert path.read_text() == source
    output = terminal.getvalue()
    assert "100%" in output and "150/150" in output
    assert "Starting Warmup updates" in output and "Completed Warmup updates" in output
    assert "\033[1;34m[TRAIN]" in output
    assert "Press Enter" in output
    assert "[TIMING]" not in output and "[PD]" not in output
    if interactive:
        assert "Critical chunks: 210" in output
        assert "\033[" in output and "A" in output
    else:
        assert "work_ms=14.00" not in output


@pytest.mark.parametrize("enabled", [False, True])
def test_stage_timing_never_synchronizes_and_reports_when_enabled(enabled):
    from rlinf.utils.progress import StageTiming

    records, syncs = [], []
    timing = StageTiming(
        "Learner update", enabled, records.append, lambda: syncs.append(True)
    )
    with timing.measure():
        value = sum(range(100))
    assert value == 4950
    assert not syncs
    assert bool(records) == enabled
    if enabled:
        assert any("work_ms=" in row and "work_hz=" in row for row in records)


@pytest.mark.parametrize("chunk_size", [1, 3, 10])
def test_chunk_horizons_override_legacy_steps_without_mutating_them(chunk_size):
    from rlinf.config import (
        get_episode_step_limit,
        get_rollout_chunk_count,
        resolve_chunk_limits,
    )

    cfg = OmegaConf.create(
        {
            "runner": {"timing_mode": "detailed"},
            "rollout": {},
            "env": {
                "train": {
                    "env_type": "real",
                    "init_params": {"id": "UR5eEnv-v1"},
                    "max_episode_chunks": 200,
                    "max_chunks_per_rollout_epoch": 20,
                    "max_episode_steps": 7,
                    "max_steps_per_rollout_epoch": 7,
                }
            },
        }
    )
    resolve_chunk_limits(cfg, chunk_size)
    assert get_episode_step_limit(cfg.env.train) == 200 * chunk_size
    assert get_rollout_chunk_count(cfg.env.train, chunk_size) == 20
    assert (
        cfg.env.train.max_episode_steps
        == cfg.env.train.max_steps_per_rollout_epoch
        == 7
    )
    assert cfg.env.train.online_timing_enabled
    legacy = OmegaConf.create(
        {"max_episode_steps": 300, "max_steps_per_rollout_epoch": 300}
    )
    assert get_episode_step_limit(legacy) == 300
    assert get_rollout_chunk_count(legacy, chunk_size) == 300 // chunk_size


@pytest.mark.parametrize(
    "field", ["max_episode_chunks", "max_chunks_per_rollout_epoch"]
)
@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_chunk_horizons_fail_before_robot_start(field, value):
    from rlinf.config import resolve_chunk_limits

    cfg = OmegaConf.create(
        {
            "runner": {},
            "rollout": {},
            "env": {"train": {"env_type": "real", field: value}},
        }
    )
    with pytest.raises(ValueError, match="positive integer"):
        resolve_chunk_limits(cfg, 3)


@pytest.mark.parametrize("enabled", [False, True])
def test_episode_replay_selects_episode_horizon_and_validates_contract(enabled):
    from rlinf.config import get_rollout_chunk_count, validate_episode_replay_cfg

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "loss_type": "rlt_ac",
                "rlt_transition_replay": True,
                "rlt_schedule": {"enable": True},
            },
            "actor": {"model": {"num_action_chunks": 3}},
            "rollout": {
                "pipeline_stage_num": 1,
                "generation_backend": "huggingface",
                "collect_transitions": True,
            },
            "env": {
                "train": {
                    "flush_replay_on_episode_end": enabled,
                    "env_type": "real",
                    "init_params": {"id": "UR5eEnv-v1"},
                    "total_num_envs": 1,
                    "rollout_epoch": 1,
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "max_episode_chunks": 150,
                    "max_chunks_per_rollout_epoch": 50,
                }
            },
        }
    )
    validate_episode_replay_cfg(cfg)
    assert get_rollout_chunk_count(cfg.env.train, 3) == (150 if enabled else 50)
    assert cfg.env.train.max_chunks_per_rollout_epoch == 50
    for key, value in (
        ("env.train.auto_reset", False),
        ("env.train.total_num_envs", 2),
        ("env.train.rollout_epoch", 2),
        ("rollout.pipeline_stage_num", 2),
        ("algorithm.loss_type", "ppo"),
    ):
        previous = OmegaConf.select(cfg, key)
        OmegaConf.update(cfg, key, value)
        if enabled:
            with pytest.raises(
                ValueError, match="flush_replay_on_episode_end requires"
            ):
                validate_episode_replay_cfg(cfg)
        else:
            validate_episode_replay_cfg(cfg)
        OmegaConf.update(cfg, key, previous)


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("color", [False, True])
def test_compact_episode_output_keeps_result_and_prompt_only(
    tmp_path, interactive, color
):
    from rlinf.utils.progress import stream_run_log

    details = [
        "[RLT] Success confirmed; ending episode with reward 1.",
        "[RESET] Opening gripper and moving UR to reset pose.",
        "[RESET] Completed forceModeStop -> servoStop -> moveL -> configured payload -> zeroFtSensor; settle_time=0.75 s; cleared action hold and refreshed reference pose.",
        "[RLT] Episode reset; critical_phase=False. Switch: Key.dot (double press within 0.50 s).",
        "[RLT] Success: Key.space (double press within 0.50 s).",
        "[RLT] Enter confirmed; ready for the first policy action.",
    ]
    result = "[EPISODE] Stopped after 422 actions: reward=1.0, terminated=True, truncated=False."
    prompt = "[RLT] Reset complete. Press Enter to start this episode."
    lines = [details[0], result, *details[1:-1], prompt, details[-1]]
    source = "\n".join(f"(EnvGroup pid=123) {line}" for line in lines) + "\n"
    output = io.StringIO()
    path = tmp_path / "episode.log"
    stream_run_log(
        io.StringIO(source),
        output,
        str(path),
        compact=True,
        interactive=interactive,
        color=color,
    )
    assert path.read_text() == source
    rendered = output.getvalue()
    assert rendered.count("\n") == 2
    assert result in rendered and prompt in rendered
    assert all(detail not in rendered for detail in details)
    if color:
        assert "\033[1;97m[EPISODE]" in rendered
        assert "\033[1;33m[RLT]" in rendered
    else:
        assert "\033" not in rendered


@pytest.mark.parametrize("interactive", [False, True])
def test_compact_live_categories_keep_distinct_colors(tmp_path, interactive):
    from rlinf.utils.progress import stream_run_log

    lines = [
        "[LOAD] Loading model: 1.0s elapsed",
        "[COLLECT] Waiting: 12/150",
        "[BUFFER] Critical chunks: 150 available; 150 collected total",
        "[PROGRESS] Warmup updates: 10/150 elapsed=2.000 initial=0 buffer=150",
        "[PROGRESS] Online RL runner rounds: 274/8000 elapsed=74.000 initial=19 buffer=150",
        "[TIMING] Learner update: work_ms=12.00",
        "[RLT] Critical phase selected.",
        "[RLT] Reset complete. Press Enter to start this episode.",
        "UserWarning: example warning",
        "[ERROR] [RESET] Completed forceModeStop -> cleanup error",
    ]
    source = "\n".join(lines) + "\n"
    output = io.StringIO()
    path = tmp_path / "colors.log"
    stream_run_log(
        io.StringIO(source),
        output,
        str(path),
        compact=True,
        color=True,
        interactive=interactive,
    )
    assert path.read_text() == source
    rendered = output.getvalue()
    for code, label in [
        ("1;36", "[LOAD]"),
        ("1;35", "[COLLECT]"),
        ("1;35", "[BUFFER]"),
        ("1;34", "Warmup updates:"),
        ("1;32", "Online RL runner rounds:"),
        ("1;35", "[RLT] Critical phase selected."),
        ("1;33", "[RLT] Reset complete."),
        ("38;5;174", "UserWarning: example warning"),
        ("1;31", "[RESET] Completed forceModeStop -> cleanup error"),
    ]:
        assert f"\033[{code}m{label}" in rendered


@pytest.mark.parametrize("interactive", [False, True])
def test_episode_chunk_progress_refreshes_and_resets_in_place(tmp_path, interactive):
    from rlinf.utils.progress import stream_run_log

    source = "".join(
        f"[EXEC] Episode chunks: {count}/200\n" for count in (0, 1, 20, 37, 200, 0)
    )
    output = io.StringIO()
    path = tmp_path / "chunks.log"
    stream_run_log(
        io.StringIO(source),
        output,
        str(path),
        compact=True,
        interactive=interactive,
        color=True,
    )
    assert path.read_text() == source
    rendered = output.getvalue()
    assert "\033[38;5;214mEpisode chunks:" in rendered
    if interactive:
        assert "37/200" in rendered and "200/200" in rendered
        assert rendered.count("\n") == 0
        assert rendered.rindex("0/200") > rendered.index("200/200")
    else:
        assert rendered.count("\n") == 1


@pytest.mark.parametrize("interactive", [False, True])
def test_online_progress_replaces_standalone_buffer_row(tmp_path, interactive):
    from rlinf.utils.progress import stream_run_log

    source = (
        "[BUFFER] Critical chunks: 150 available; 150 collected total\n"
        "[PROGRESS] Online RL runner rounds: 274/8000 elapsed=74.000 initial=19 buffer=281\n"
        "[BUFFER] Critical chunks: 999 available; 999 collected total\n"
        "[RLT] Critical phase selected.\n"
    )
    output = io.StringIO()
    path = tmp_path / "buffer.log"
    stream_run_log(
        io.StringIO(source),
        output,
        str(path),
        compact=True,
        interactive=interactive,
        color=True,
    )
    assert path.read_text() == source
    rendered = output.getvalue()
    assert "buffer=281" in rendered
    assert "999" not in rendered
    after_online = rendered[rendered.index("Online RL runner rounds:") :]
    assert "[BUFFER]" not in after_online


@pytest.mark.parametrize("enabled", [False, True])
def test_structured_timing_file_records_each_operation(tmp_path, enabled):
    import csv
    import json

    from rlinf.utils.progress import StageTiming, emit_timing_record, stream_run_log

    messages = []
    timing = StageTiming("Learner update", enabled, messages.append)
    for update in (41, 42):
        timing.record(0.25, context={"rank": 0, "update": update}, target_interval=0.05)
    if enabled:
        emit_timing_record(
            messages.append,
            "UR background",
            episode=2,
            chunk=3,
            samples=1,
            intervals=0,
            measured_hz=None,
        )
    terminal = io.StringIO()
    path = tmp_path / "run_embodiment.log"
    stream = "".join(f"(Worker pid=123) {message}\n" for message in messages)
    stream_run_log(io.StringIO(stream), terminal, str(path), compact=True)
    assert "TIMING_RECORD" not in terminal.getvalue()
    assert path.read_text() == stream
    timing_path = tmp_path / "timing.tsv"
    table_path = tmp_path / "timing.log"
    assert table_path.exists() == enabled
    assert timing_path.exists() == enabled
    if not enabled:
        return
    # Reopening a run appends rows without another header.
    stream_run_log(io.StringIO(stream), terminal, str(path), compact=True)
    with timing_path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    assert len(rows) == 6
    table = table_path.read_text()
    assert table.count("Timing | Learner update |") == 4
    assert table.count("Timing | UR background |") == 2
    assert table.count("Raw numeric records: timing.tsv.") == 1
    assert "update=41" in table and "work_ms=250.000" in table
    assert "measured_hz=n/a" in table
    assert "Duration (ms)" in table and "Cadence / PD" in table
    assert "\t" not in table
    assert [row["update"] for row in rows[:2]] == ["41", "42"]
    assert [row["sample"] for row in rows[:2]] == ["1", "2"]
    assert float(rows[0]["target_interval_ms"]) == 50
    assert "target_interval_ms=50.000" in table
    assert float(rows[0]["work_ms"]) == 250
    assert float(rows[0]["work_hz"]) == 4
    assert rows[2]["measured_hz"] == ""
    assert rows[2]["episode"] == "2" and rows[2]["chunk"] == "3"
    records = [
        json.loads(row.split("[TIMING_RECORD] ")[1])
        for row in messages
        if "[TIMING_RECORD] " in row
    ]
    assert len(records) == 3


@pytest.mark.parametrize("enabled", [False, True])
def test_learning_sections_and_runner_step_are_written_to_timing_files(
    tmp_path, enabled
):
    import csv
    import json

    from rlinf.utils.progress import StageTiming, stream_run_log

    messages, syncs = [], []
    learner = StageTiming(
        "Learner update", enabled, messages.append, lambda: syncs.append(1)
    )
    for update, actor_updated in [(1, True), (2, False)]:
        with learner.measure(update=update, actor_updated=actor_updated) as measurement:
            with measurement.measure("critic"):
                sum(range(100))
            if actor_updated:
                with measurement.measure("actor"):
                    sum(range(100))
    runner = StageTiming("Runner step", enabled, messages.append)
    runner.record(0.1, context={"step": 7, "critic_updates": 8, "actor_updates": 2})
    log = tmp_path / "run_embodiment.log"
    terminal = io.StringIO()
    stream_run_log(
        io.StringIO("\n".join(messages) + "\n"), terminal, str(log), compact=True
    )
    assert not terminal.getvalue()
    assert not syncs
    if not enabled:
        assert not log.with_name("timing.tsv").exists()
        return
    rows = list(csv.DictReader(log.with_name("timing.tsv").open(), delimiter="\t"))
    assert len(rows) == 3
    assert rows[0]["source"] == "Learner update"
    assert rows[0]["actor_updated"] == "True"
    assert float(rows[0]["critic_ms"]) > 0
    assert float(rows[0]["actor_ms"]) > 0
    assert rows[1]["actor_updated"] == "False" and rows[1]["actor_ms"] == ""
    assert rows[2]["source"] == "Runner step"
    assert rows[2]["step"] == "7"
    assert rows[2]["critic_updates"] == "8" and rows[2]["actor_updates"] == "2"
    records = [
        json.loads(row.split("[TIMING_RECORD] ")[1])
        for row in messages
        if "[TIMING_RECORD]" in row
    ]
    assert records[0]["work_ms"] == pytest.approx(
        records[0]["critic_ms"] + records[0]["actor_ms"] + records[0]["other_ms"]
    )
    rendered = log.with_name("timing.log").read_text()
    assert "critic_ms=" in rendered and "actor_ms=" in rendered
    assert "Timing | Runner step" in rendered and "step=7" in rendered


@pytest.mark.parametrize("mode", ["disabled", "basic", "detailed"])
def test_worker_timing_modes_partition_local_work_without_sync(
    tmp_path, monkeypatch, mode
):
    import json
    from itertools import count

    import rlinf.utils.progress as progress

    calls = []
    ticks = count()

    def clock():
        calls.append(True)
        if mode == "disabled":
            raise AssertionError("Disabled timing read a clock")
        return next(ticks) / 1000

    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=clock, time=clock))
    records = []
    actual = torch.arange(4, dtype=torch.float32)
    with progress.WorkerStepTiming(mode, "Rollout step", records.append):
        # Upstream waiting has no section and must not contribute to work_ms.
        with progress.timing_section("input", major=True):
            actual = actual + 1
        with progress.timing_section("inference", major=True):
            with progress.timing_section("prefix"):
                actual = actual.square()
        with progress.timing_section("action_send", major=True):
            actual = actual.clone()
        with progress.timing_section("data_submit", major=True):
            payload = actual.tolist()
    assert payload == [1, 4, 9, 16]
    assert progress.current_worker_timing() is None
    if mode == "disabled":
        assert not calls and not records
        return
    record = json.loads(records[0].split("[TIMING_RECORD] ")[1])
    assert record["work_ms"] == pytest.approx(
        sum(
            record[k + "_ms"]
            for k in ["input", "inference", "action_send", "data_submit"]
        )
    )
    assert ("prefix_ms" in record) == (mode == "detailed")
    assert record["clock"] == "host_elapsed"
    assert record["timing_mode"] == mode
    output = io.StringIO()
    log = tmp_path / "run.log"
    progress.stream_run_log(
        io.StringIO(records[0] + "\n"), output, str(log), compact=True
    )
    assert log.with_name("timing.tsv").exists()
    assert "Invalid timing" not in output.getvalue()


@pytest.mark.parametrize("mode", ["basic", "detailed"])
def test_deduplicated_timing_preserves_records_and_other_output(tmp_path, mode):
    import csv
    import json

    from rlinf.utils.progress import stream_run_log

    record = {
        "timestamp": 1793526641.8338847,
        "source": "Learner update",
        "timing_mode": mode,
        "clock": "host_elapsed",
        "rank": 0,
        "update": 182014,
        "sample": 182014,
        "actor_updated": False,
        "work_ms": 7.774195168167353,
        "target_ms": 0.09318580850958824,
        "action_interval_hz": None,
    }
    if mode == "detailed":
        record.update(episode=81, chunk=60, prefix_ms=52.432123456789)
    timing_line = (
        "(ActorGroup(rank=0) pid=3914163) [INFO 17:50:41 ActorGroup-Rank-0]"
        "[progress.py:419] [TIMING_RECORD] " + json.dumps(record) + "\n"
    )
    ordinary = "[RESET] Completed\nUserWarning: stream mismatch\n"
    malformed = '[TIMING_RECORD] {"timestamp": broken}\n'
    unknown = '[TIMING_RECORD] {"timestamp": 1, "future_field": 7}\n'
    traceback = "Traceback (most recent call last):\n  call()\nRuntimeError: failed\n"
    stream = ordinary + timing_line + malformed + unknown + traceback
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for directory, deduplicate in [(old, False), (new, True)]:
        for _ in range(2):
            stream_run_log(
                io.StringIO(stream),
                io.StringIO(),
                str(directory / "run.log"),
                compact=True,
                deduplicate_timing=deduplicate,
            )
    assert (old / "run.log").read_text() == stream * 2
    assert (new / "run.log").read_text() == (
        ordinary + malformed + unknown + traceback
    ) * 2
    original = [
        line
        for line in (old / "run.log").read_text().splitlines(keepends=True)
        if line == timing_line
    ]
    preserved = [
        line
        for line in (new / "timing.log").read_text().splitlines(keepends=True)
        if "[TIMING_RECORD] " in line
    ]
    assert preserved == original == [timing_line, timing_line]
    with (new / "timing.tsv").open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    assert len(rows) == 2
    run_bytes = (new / "run.log").read_bytes()
    restored = bytearray()
    offset = 0
    for row, line in zip(rows, preserved):
        next_offset = int(row["run_log_offset"])
        restored.extend(run_bytes[offset:next_offset])
        restored.extend(line.encode())
        offset = next_offset
    restored.extend(run_bytes[offset:])
    assert restored.decode() == (old / "run.log").read_text()
    for row in rows:
        assert float(row["work_ms"]) == record["work_ms"]
        assert float(row["target_ms"]) == record["target_ms"]
        assert row["update"] == "182014"
        assert row["actor_updated"] == "False"
    assert (new / "timing.log").stat().st_size < (old / "timing.log").stat().st_size


def test_compact_metrics_preserve_every_value_and_progress(
    tmp_path, monkeypatch, capsys
):
    import json

    import numpy as np

    from rlinf.utils import metric_utils

    monkeypatch.setattr(metric_utils.time, "time", lambda: 1700000123.125)
    metrics = {
        "train/actor/very_long_metric_name_that_tables_truncate": 1.2345678901234567,
        "train/sac/critic_loss": np.float32(1.1420744e-6),
        "train/rlt/update_step": np.int64(182006),
        "time/step": torch.tensor(0.10556268692016602, dtype=torch.float64),
        "env/success_once": True,
        "env/note": "接管\n释放",
        "custom": {"missing": None, "values": np.array([1.0, 2.0])},
    }
    for step in (22500, 22501):
        metric_utils.print_metrics_table(
            step, 80000, 1700000000.0, metrics, 100, str(tmp_path), compact=True
        )
    assert capsys.readouterr().out == ""
    records = [
        json.loads(line) for line in (tmp_path / "metrics.log").read_text().splitlines()
    ]
    assert len(records) == 2
    for step, record in zip((22501, 22502), records):
        assert record["step"] == step
        assert record["total_steps"] == 80000
        assert record["start_step"] == 100
        assert record["start_time"] == 1700000000.0
        assert record["timestamp"] == 1700000123.125
        assert record["elapsed_seconds"] == 123.125
        assert record["average_step_seconds"] == 123.125 / (step - 100)
        assert record["eta_seconds"] == 123.125 / (step - 100) * (80000 - step)
        assert record["metrics"] == {
            **metrics,
            "train/sac/critic_loss": float(metrics["train/sac/critic_loss"]),
            "custom": {"missing": None, "values": [1.0, 2.0]},
        }


def test_compact_metrics_can_reproduce_legacy_table(tmp_path, monkeypatch, capsys):
    import json

    from rlinf.utils import metric_utils

    monkeypatch.setattr(metric_utils.time, "time", lambda: 1234.56789)
    metrics = {
        "time/step": 0.12567890123456,
        "train/sac/actor_loss": -0.008408108726143837,
        "train/sac/critic_loss": 1.1420744385759463e-6,
        "train/actor/weighted_bc": 0.00011898841330548748,
        "train/actor/weighted_q": 0.008527097292244434,
        "train/rlt/update_step": 182006,
        "env/success_once": True,
    }
    legacy = tmp_path / "legacy"
    compact = tmp_path / "compact"
    metric_utils.print_metrics_table(22500, 80000, 1000, metrics, 100, str(legacy))
    expected = capsys.readouterr().out
    assert (legacy / "metrics.log").read_text() == expected
    metric_utils.print_metrics_table(
        22500, 80000, 1000, metrics, 100, str(compact), compact=True
    )
    record = json.loads((compact / "metrics.log").read_text())
    metric_utils.print_metrics_table(
        record["step"] - 1,
        record["total_steps"],
        record["start_time"],
        record["metrics"],
        record["start_step"],
    )
    assert capsys.readouterr().out == expected
    assert (compact / "metrics.log").stat().st_size < (
        legacy / "metrics.log"
    ).stat().st_size

    metric_utils.print_metrics_table(
        22500, 80000, 1000, metrics, 100, str(legacy), compact=True
    )
    assert capsys.readouterr().out == expected
    assert (legacy / "metrics.log").read_text() == expected * 2


def test_deduplicated_timing_keeps_existing_legacy_run_readable(tmp_path):
    import csv
    import json

    from rlinf.utils.progress import stream_run_log

    line = (
        "[TIMING_RECORD] "
        + json.dumps(
            {
                "timestamp": 1793526641.8338847,
                "source": "Rollout step",
                "work_ms": 80.12,
            }
        )
        + "\n"
    )
    path = tmp_path / "run.log"
    for deduplicate in (False, True):
        stream_run_log(
            io.StringIO(line),
            io.StringIO(),
            str(path),
            compact=True,
            deduplicate_timing=deduplicate,
        )
    assert path.read_text() == line * 2
    with path.with_name("timing.tsv").open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    assert len(rows) == 2
    assert all(None not in row for row in rows)
    assert path.with_name("timing.log").read_text().count("Timing | Rollout step") == 2


def test_disabled_additional_timers_do_not_start_threads_or_read_clocks(monkeypatch):
    import rlinf.utils.progress as progress

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled timer performed measurement work")

    monkeypatch.setattr(
        progress, "time", SimpleNamespace(monotonic=forbidden, time=forbidden)
    )
    monkeypatch.setattr(progress.threading, "Thread", forbidden)
    monkeypatch.setattr(progress.threading, "Event", forbidden)
    with progress.ElapsedTime("LOAD", "load", log=forbidden, enabled=False):
        pass
    stage = progress.StageTiming("Learner update", False, forbidden, forbidden)
    with stage.measure() as measurement:
        with measurement.measure("critic"):
            pass
    progress.TrainingProgress("updates", 150, forbidden, enabled=False).update(150, 150)


@pytest.mark.parametrize("value", ["off", "unknown", True, False, 1])
def test_timing_mode_rejects_invalid_values(value):
    from rlinf.utils.progress import get_timing_mode

    with pytest.raises(ValueError, match="timing_mode"):
        get_timing_mode({"runner": {"timing_mode": value}})


def test_timing_mode_defaults_to_disabled():
    from rlinf.utils.progress import get_timing_mode

    assert get_timing_mode({}) == "disabled"
    assert get_timing_mode({"runner": {}}) == "disabled"


@pytest.mark.parametrize("mode", ["disabled", "basic", "detailed"])
def test_timing_mode_controls_activation(mode):
    from rlinf.utils.progress import get_timing_mode, timing_active, timing_detailed

    cfg = {"runner": {"timing_mode": mode}}
    assert get_timing_mode(cfg) == mode
    assert timing_active(cfg) == (mode != "disabled")
    assert timing_detailed(cfg) == (mode == "detailed")
