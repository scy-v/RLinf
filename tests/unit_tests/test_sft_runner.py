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

"""SFT runner terminal output and persistent metrics."""

import re

import pytest
from omegaconf import OmegaConf

from rlinf.runners.sft_runner import SFTRunner


class CompletedCall:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value

    def consume_duration(self):
        return 0.01


class LocalSFTActor:
    """In-process stand-in for the remote SFT worker interface."""

    def get_max_steps_per_epoch(self):
        return CompletedCall([100])

    def run_training(self):
        return CompletedCall([{"loss": 0.5}])


def runner_config(tmp_path, max_steps):
    return OmegaConf.create(
        {
            "runner": {
                "max_steps": max_steps,
                "val_check_interval": -1,
                "save_interval": -1,
                "logger": {
                    "log_path": str(tmp_path),
                    "logger_backends": ["tensorboard"],
                },
            }
        }
    )


@pytest.mark.parametrize(
    ("interval", "start_step", "max_steps", "expected"),
    [
        (None, 0, 23, [0, 10, 20, 23]),
        (None, 7, 23, [7, 10, 20, 23]),
        (4, 0, 11, [0, 4, 8, 11]),
        (1, 0, 3, [0, 1, 2, 3]),
    ],
)
def test_terminal_interval_preserves_every_tensorboard_step(
    tmp_path, capsys, interval, start_step, max_steps, expected
):
    cfg = runner_config(tmp_path, max_steps)
    if interval is not None:
        cfg.runner.log_interval = interval
    runner = SFTRunner(cfg, LocalSFTActor())
    runner.global_step = start_step
    runner.run()

    output = capsys.readouterr().err
    displayed_steps = [
        int(step) for step in re.findall(rf"\b(\d+)/{max_steps}\b", output)
    ]
    assert list(dict.fromkeys(displayed_steps)) == expected

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    for tag in ("train/loss", "time/step", "time/training"):
        assert [event.step for event in events.Scalars(tag)] == list(
            range(start_step, max_steps)
        )


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_reject_invalid_log_interval(tmp_path, interval):
    cfg = runner_config(tmp_path, 1)
    cfg.runner.log_interval = interval
    with pytest.raises(
        ValueError, match="runner.log_interval must be a positive integer"
    ):
        SFTRunner(cfg, LocalSFTActor())
