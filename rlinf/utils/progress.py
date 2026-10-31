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

"""Worker-local timing and training progress reporting."""

import csv
import json
import sys
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from types import TracebackType


def get_timing_mode(cfg) -> str:
    """Resolve the additional timing mode; unspecified configurations stay untimed."""
    runner = cfg.get("runner", cfg)
    mode = runner.get("timing_mode", "disabled")
    if mode not in ("disabled", "basic", "detailed"):
        raise ValueError(f"Expected timing_mode disabled/basic/detailed, got {mode!r}.")
    return mode


def timing_active(cfg) -> bool:
    """Return whether additional learner or worker timing is enabled."""
    return get_timing_mode(cfg) != "disabled"


def timing_detailed(cfg) -> bool:
    """Return whether submodule and diagnostic timing is enabled."""
    return get_timing_mode(cfg) == "detailed"


_WORKER_TIMING_SEQUENCES = {}
_ACTIVE_WORKER_TIMING = ContextVar("rlinf_worker_timing", default=None)
_NO_TIMING = nullcontext()


def current_worker_timing():
    """Return the active local step measurement, if any."""
    return _ACTIVE_WORKER_TIMING.get()


def timing_section(name: str, *, major: bool = False):
    """Measure a named local section; disabled sections read no clocks."""
    measurement = current_worker_timing()
    if measurement is None or (not major and not measurement.detailed):
        return _NO_TIMING
    return measurement.section(name, major=major)


class WorkerStepTiming:
    """Partition one step into major sections, excluding upstream input waits.

    work_ms is the sum of the major sections. Nested detailed sections are
    explanatory and are never added to work_ms a second time. Values measure
    host elapsed time, including business-required waits; no device sync is added.
    """

    def __init__(self, mode: str, source: str, log, **context):
        self.enabled = mode != "disabled"
        self.detailed = mode == "detailed"
        self.mode, self.source, self.log = mode, source, log
        self.context = context
        names = (
            ("input", "execute", "publish")
            if source == "Env step"
            else ("input", "inference", "action_send", "data_submit")
        )
        self.major = dict.fromkeys(names, 0.0) if self.enabled else {}
        self.details = {}
        self.emit = True

    def __enter__(self):
        self.token = _ACTIVE_WORKER_TIMING.set(self if self.enabled else None)
        return self

    @contextmanager
    def section(self, name: str, *, major: bool = False):
        if not self.enabled or (not major and not self.detailed):
            yield
            return
        token = _ACTIVE_WORKER_TIMING.set(self)
        started = time.monotonic()
        try:
            yield
        finally:
            seconds = time.monotonic() - started
            target = self.major if major else self.details
            target[name] = target.get(name, 0.0) + seconds
            _ACTIVE_WORKER_TIMING.reset(token)

    def add_detail(self, name: str, seconds: float) -> None:
        if self.detailed:
            self.details[name] = self.details.get(name, 0.0) + seconds

    def __exit__(self, exc_type, exc, traceback):
        _ACTIVE_WORKER_TIMING.reset(self.token)
        if self.enabled and self.emit and exc_type is None:
            components = {
                f"{key}_ms": value * 1000 for key, value in self.major.items()
            }
            components.update(
                {f"{key}_ms": value * 1000 for key, value in self.details.items()}
            )
            key = (self.source, self.context.get("rank"), self.context.get("stage_id"))
            sequence = _WORKER_TIMING_SEQUENCES.get(key, 0) + 1
            _WORKER_TIMING_SEQUENCES[key] = sequence
            emit_timing_record(
                self.log,
                self.source,
                sample=sequence,
                timing_mode=self.mode,
                clock="host_elapsed",
                work_ms=sum(self.major.values()) * 1000,
                samples=1,
                **self.context,
                **components,
            )


class ElapsedTime:
    """Report a blocking operation once per second and stop on success or failure.

    Args:
        label: Short phase name displayed in brackets.
        description: Operation being timed.
        log: Logger callback; when omitted, update stderr in place.
    """

    def __init__(
        self,
        label: str,
        description: str,
        log: Callable[[str], None] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.label = label
        self.description = description
        self.log = log
        self.duration = 0.0
        self._stopped = threading.Event() if enabled else None

    def _report(self, message: str, *, final: bool = False) -> None:
        message = f"[{self.label}] {message}"
        if self.log is not None:
            self.log(message)
        else:
            self._stream.write("\r" + message.ljust(100) + ("\n" if final else ""))
            self._stream.flush()

    def _update(self) -> None:
        while not self._stopped.wait(1.0):
            self._report(
                f"{self.description}: {time.monotonic() - self._started:.1f}s elapsed"
            )

    def __enter__(self) -> "ElapsedTime":
        if not self.enabled:
            return self
        self._stream = sys.stderr
        self._started = time.monotonic()
        self._report(f"{self.description}: 0.0s elapsed")
        self._thread = threading.Thread(
            target=self._update, name=f"ur-{self.label.lower()}-timer", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self.enabled:
            return
        self._stopped.set()
        self._thread.join()
        self.duration = time.monotonic() - self._started
        status = "Completed" if exc_type is None else "Stopped before completion"
        self._report(
            f"{self.description}: {status} after {self.duration:.2f}s", final=True
        )


class TrainingProgress:
    """Send bounded progress events for the driver's single terminal renderer."""

    def __init__(
        self,
        label: str,
        total: int,
        log: Callable[[str], None],
        initial: int = 0,
        *,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.label, self.total, self.log = label, total, log
        self.initial = initial
        self.complete = False
        if not enabled:
            return
        self.started = time.monotonic()
        self.last_report = float("-inf")
        self.log(f"[TRAIN] Starting {label}: {initial}/{total}.")

    def update(self, current: int, buffer_size: int, *, force: bool = False) -> None:
        """Report at most once per second, always reporting phase completion."""
        if not self.enabled or self.complete:
            return
        now = time.monotonic()
        if not force and current < self.total and now - self.last_report < 1.0:
            return
        self.last_report = now
        self.log(
            f"[PROGRESS] {self.label}: {current}/{self.total} "
            f"elapsed={now - self.started:.3f} initial={self.initial} buffer={buffer_size}"
        )
        if current >= self.total:
            self.complete = True
            self.log(f"[TRAIN] Completed {self.label}: {current}/{self.total}.")


TIMING_COLUMNS = (
    "timing_mode",
    "clock",
    "phase",
    "stage_id",
    "input_ms",
    "inference_ms",
    "action_send_ms",
    "data_submit_ms",
    "execute_ms",
    "publish_ms",
    "preprocess_ms",
    "prefix_ms",
    "z_encode_ms",
    "reference_ms",
    "decision_ms",
    "action_prepare_ms",
    "action_execution_ms",
    "chunk_result_ms",
    "request_ms",
    "transition_ms",
    "send_ms",
    "timestamp",
    "source",
    "rank",
    "sample",
    "update",
    "step",
    "actor_updated",
    "critic_updates",
    "actor_updates",
    "policy_version",
    "episode",
    "chunk",
    "action",
    "intervals",
    "active_interval_ms",
    "samples",
    "work_ms",
    "max_ms",
    "work_hz",
    "control_ms",
    "command_ms",
    "pace_wait_ms",
    "observation_ms",
    "target_interval_ms",
    "action_interval_ms",
    "action_interval_hz",
    "target_hz",
    "measured_hz",
    "max_send_interval_ms",
    "work_overruns",
    "held_target_samples",
    "force_mode_ms",
    "sample_ms",
    "prepare_ms",
    "critic_ms",
    "actor_ms",
    "alpha_ms",
    "target_ms",
    "other_ms",
)


def format_timing_table(record: dict) -> str:
    """Render one operation or PD window without unrelated empty columns."""
    groups = (
        (
            "Position",
            (
                "rank",
                "sample",
                "update",
                "step",
                "actor_updated",
                "critic_updates",
                "actor_updates",
                "policy_version",
                "episode",
                "chunk",
                "action",
                "samples",
            ),
        ),
        (
            "Duration (ms)",
            (
                "work_ms",
                "max_ms",
                "control_ms",
                "command_ms",
                "pace_wait_ms",
                "observation_ms",
                "force_mode_ms",
                "sample_ms",
                "prepare_ms",
                "critic_ms",
                "actor_ms",
                "alpha_ms",
                "target_ms",
                "other_ms",
            ),
        ),
        (
            "Worker modules / details",
            (
                "timing_mode",
                "clock",
                "phase",
                "stage_id",
                "input_ms",
                "inference_ms",
                "action_send_ms",
                "data_submit_ms",
                "execute_ms",
                "publish_ms",
                "preprocess_ms",
                "prefix_ms",
                "z_encode_ms",
                "reference_ms",
                "decision_ms",
                "action_prepare_ms",
                "action_execution_ms",
                "chunk_result_ms",
                "request_ms",
                "transition_ms",
                "send_ms",
            ),
        ),
        (
            "Cadence / PD",
            (
                "work_hz",
                "target_interval_ms",
                "action_interval_ms",
                "action_interval_hz",
                "target_hz",
                "measured_hz",
                "intervals",
                "active_interval_ms",
                "max_send_interval_ms",
                "work_overruns",
                "held_target_samples",
            ),
        ),
    )
    width = 118
    lines = ["╭" + "─" * width + "╮"]

    def row(text: str) -> None:
        lines.append("│ " + text.ljust(width - 2) + " │")

    row(f"Timing | {record['source']} | {record['timestamp']}")
    for title, keys in groups:
        cells = []
        for key in keys:
            if key not in record:
                continue
            value = record[key]
            if value is None:
                value = "n/a"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(f"{key}={value}")
        if not cells:
            continue
        lines.append("├" + f" {title} ".center(width, "─") + "┤")
        for start in range(0, len(cells), 3):
            row(" | ".join(cell.ljust(36) for cell in cells[start : start + 3]))
    lines.append("╰" + "─" * width + "╯")
    return "\n".join(lines) + "\n\n"


def emit_timing_record(log: Callable[[str], None], source: str, **values) -> None:
    """Transport one timing row to the driver's single file writer."""
    record = {"timestamp": time.time(), "source": source, **values}
    log("[TIMING_RECORD] " + json.dumps(record, allow_nan=False))


class StageMeasurement:
    """Collect local wall-clock sections without accelerator synchronization."""

    def __init__(self, enabled: bool, synchronize: Callable[[], None] | None = None):
        self.enabled = enabled
        self.synchronize = synchronize
        self.components: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Measure a section without emitting a separate log record."""
        if not self.enabled:
            yield
            return
        started = time.monotonic()
        yield
        self.components[name] = self.components.get(name, 0.0) + (
            time.monotonic() - started
        )


class StageTiming:
    """Summarize local elapsed work without changing device execution."""

    def __init__(
        self,
        label: str,
        enabled: bool,
        log: Callable[[str], None],
        synchronize: Callable[[], None] | None = None,
        *,
        timing_mode: str | None = None,
    ):
        self.mode = timing_mode
        self.label, self.enabled, self.log = label, enabled, log
        self.synchronize = synchronize
        self.samples = 0
        self.totals: dict[str, float] = {}
        self.maximum = 0.0
        self.last_report = 0.0
        self.sequence = 0
        self.interval_samples = 0

    @contextmanager
    def measure(self, **context):
        """Measure local elapsed work only when enabled."""
        measurement = StageMeasurement(self.enabled, self.synchronize)
        if not self.enabled:
            yield measurement
            return
        started = time.monotonic()
        yield measurement
        seconds = time.monotonic() - started
        components = measurement.components
        if components:
            components["other"] = max(0.0, seconds - sum(components.values()))
        self.record(seconds, context=context, **components)

    def record(
        self, seconds: float, *, context: dict | None = None, **components: float
    ) -> None:
        """Record every operation; emit terminal window summaries every 5 seconds."""
        if not self.enabled:
            return
        self.sequence += 1
        interval = components.get("action_interval", 0.0)
        emit_timing_record(
            self.log,
            self.label,
            timing_mode=self.mode,
            clock="host_elapsed",
            sample=self.sequence,
            samples=1,
            work_ms=seconds * 1000,
            max_ms=seconds * 1000,
            work_hz=1 / seconds if seconds > 0 else None,
            **{f"{key}_ms": value * 1000 for key, value in components.items()},
            action_interval_hz=1 / interval if interval > 0 else None,
            **(context or {}),
        )
        self.samples += 1
        if interval > 0:
            self.interval_samples += 1
        self.maximum = max(self.maximum, seconds)
        for name, value in {"work": seconds, **components}.items():
            self.totals[name] = self.totals.get(name, 0.0) + value
        now = time.monotonic()
        if now - self.last_report < 5.0:
            return
        mean = self.totals["work"] / self.samples
        details = " ".join(
            f"{key}_ms={1000 * value / self.samples:.2f}"
            for key, value in self.totals.items()
        )
        interval = self.totals.get("action_interval", 0.0)
        if interval > 0:
            details += f" action_interval_hz={self.interval_samples / interval:.2f}"
        self.log(
            f"[TIMING] {self.label}: work_hz={1 / mean if mean else 0:.2f} "
            f"{details} max_ms={self.maximum * 1000:.2f} samples={self.samples}"
        )
        self.samples, self.maximum, self.totals = 0, 0.0, {}
        self.interval_samples = 0
        self.last_report = now


def _stage_tone(tag: str | None, message: str) -> str | None:
    """Use the same category colors for live rows and permanent messages."""
    if "press enter" in message.lower():
        return "1;33"
    if tag in {"PROGRESS", "TRAIN"}:
        return "1;34" if "warmup" in message.lower() else "1;32"
    return {
        "LOAD": "1;36",
        "INFER": "1;36",
        "SAVE": "38;5;208",
        "COLLECT": "1;35",
        "BUFFER": "1;35",
        "RLT": "1;35",
        "EPISODE": "1;97",
        "RESET": "1;97",
        "TIMING": "90",
        "PD": "90",
        "EXEC": "38;5;214",
    }.get(tag)


def stream_run_log(
    source,
    terminal,
    log_path: str,
    *,
    color: bool = False,
    compact: bool = False,
    interactive: bool = False,
) -> None:
    """Keep full plain logs and optionally show a compact live stage summary."""
    import re
    import shutil
    import textwrap

    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    stage = re.compile(
        r"\[(LOAD|SAVE|INFER|COLLECT|TRAIN|EPISODE|RLT|RESET|PROGRESS|BUFFER|TIMING|PD|EXEC)\]"
    )
    active: dict[str, tuple[str, str | None]] = {}
    live_width = 0
    live_rows = 0
    traceback_active = False
    collect_started = None
    online_buffer_in_progress = False

    def clear_live():
        nonlocal live_width, live_rows
        if live_width:
            terminal.write("\r")
            if live_rows > 1:
                terminal.write(f"\033[{live_rows - 1}A")
            terminal.write("\033[J")
            live_width = 0
            live_rows = 0

    def display_live():
        nonlocal live_width, live_rows
        if not interactive or not active:
            return
        width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        lines = []
        visible_length = 0
        for value, tone in active.values():
            for row in textwrap.wrap(value, width=width):
                visible_length += len(row)
                lines.append(f"\033[{tone}m{row}\033[0m" if color and tone else row)
        live_rows = len(lines)
        terminal.write("\n".join(lines))
        live_width = visible_length

    with open(log_path, "a", encoding="utf-8") as log, ExitStack() as files:
        timing_file = None
        timing_writer = None
        timing_table_file = None
        for line in source:
            plain = ansi.sub("", line)
            log.write(plain)
            log.flush()
            if "[TIMING_RECORD] " in plain:
                try:
                    record = json.loads(plain.split("[TIMING_RECORD] ", 1)[1])
                    if not isinstance(record, dict) or set(record) - set(
                        TIMING_COLUMNS
                    ):
                        raise ValueError("Unexpected timing record fields")
                    record["timestamp"] = (
                        datetime.fromtimestamp(record["timestamp"])
                        .astimezone()
                        .isoformat(timespec="milliseconds")
                    )
                except (ValueError, TypeError, KeyError, OverflowError):
                    # The original malformed record remains in the full run log.
                    clear_live()
                    terminal.write(
                        "[WARNING] Invalid timing record; see run_embodiment.log.\n"
                    )
                    display_live()
                    terminal.flush()
                    continue
                if timing_writer is None:
                    timing_file = files.enter_context(
                        open(
                            Path(log_path).with_name("timing.tsv"),
                            "a",
                            encoding="utf-8",
                            newline="",
                        )
                    )
                    timing_writer = csv.DictWriter(
                        timing_file, fieldnames=TIMING_COLUMNS, delimiter="\t"
                    )
                    if timing_file.tell() == 0:
                        timing_writer.writeheader()
                if timing_table_file is None:
                    timing_table_file = files.enter_context(
                        open(
                            Path(log_path).with_name("timing.log"),
                            "a",
                            encoding="utf-8",
                        )
                    )
                    if timing_table_file.tell() == 0:
                        timing_table_file.write(
                            "Timing records: local host elapsed time; no timing-only device synchronization.\nWorker work_ms is the sum of its major modules, excluding upstream input waiting.\nNested details and background PD windows are not added to worker totals.\n"
                            "Each source has its own indices; records do not share a training round.\n"
                            "work_hz = 1000 / work_ms, not actual execution frequency.\n"
                            "Actual cadence: target_interval_ms (UR submission); action_interval_hz is step-entry cadence; measured_hz is PD.\n"
                            "Learner sections: sample/prepare/critic/actor/alpha/target_ms; actor_updated=false means actor was skipped.\n"
                            "Learner work_ms includes all sections; other_ms includes remaining work and measurement overhead.\n"
                            "Runner step work_ms includes learner calls, weight sync waits, and scheduled checkpoint/evaluation work.\n"
                            "UR durations: control_ms = command_ms + pace_wait_ms (per call, not a full action period); observation_ms = observation acquisition.\n"
                            "PD diagnostics: max_send_interval_ms, work_overruns, held_target_samples (unchanged target).\n"
                            "PD episode/chunk/action identify the window end; n/a = not measured.\n"
                            "Raw numeric records: timing.tsv.\n\n"
                        )
                timing_writer.writerow(
                    {
                        key: f"{value:.6f}" if isinstance(value, float) else value
                        for key, value in record.items()
                    }
                )
                timing_file.flush()
                timing_table_file.write(format_timing_table(record))
                timing_table_file.flush()
                continue
            match = stage.search(plain)
            message = (
                plain[match.start() :].rstrip()
                if compact and match
                else plain.rstrip("\n")
            )
            tone = _stage_tone(match.group(1) if match else None, message)
            diagnostic = re.search(
                r"\b(?:ERROR|WARNING|Error|Exception|KeyboardInterrupt|RuntimeWarning|UserWarning)\b|Error:",
                plain,
            )
            if diagnostic:
                warning = re.search(
                    r"\b(?:WARNING|RuntimeWarning|UserWarning)\b", plain
                )
                error = re.search(
                    r"\b(?:ERROR|Error|Exception|KeyboardInterrupt)\b|Error:", plain
                )
                tone = "38;5;174" if warning and not error else "1;31"
            if compact:
                if "Traceback (most recent call last)" in plain:
                    traceback_active = True
                elif re.search(r"\[(?:INFO|WARNING|ERROR)\b", plain):
                    traceback_active = False
                important = match or traceback_active or diagnostic
                if not important:
                    continue
                if (
                    match
                    and match.group(1) in {"TIMING", "PD"}
                    and not diagnostic
                    and not traceback_active
                ):
                    continue
                if message.startswith("[PROGRESS] Online RL runner rounds:"):
                    online_buffer_in_progress = True
                    for key in list(active):
                        if key.startswith("[BUFFER]"):
                            active.pop(key)
                if (
                    match
                    and match.group(1) == "BUFFER"
                    and online_buffer_in_progress
                    and not diagnostic
                ):
                    continue
                if (
                    not diagnostic
                    and not traceback_active
                    and message.startswith(
                        (
                            "[RESET] Opening gripper and moving UR to reset pose.",
                            "[RESET] Completed forceModeStop ->",
                            "[RLT] Episode reset;",
                            "[RLT] Success:",
                            "[RLT] Success confirmed;",
                            "[RLT] Enter confirmed;",
                        )
                    )
                ):
                    continue
                if match and match.group(1) in {
                    "PROGRESS",
                    "BUFFER",
                    "TIMING",
                    "PD",
                    "EXEC",
                }:
                    from tqdm import tqdm

                    key = message.split(":", 1)[0]
                    if match.group(1) == "PROGRESS":
                        progress = re.search(
                            r"\[PROGRESS\] (.+): (\d+)/(\d+) elapsed=([\d.]+) initial=(\d+) buffer=(\d+)",
                            message,
                        )
                        if progress:
                            label, current, total, elapsed, initial, buffer = (
                                progress.groups()
                            )
                            current, total = int(current), int(total)
                            rate = (current - int(initial)) / max(float(elapsed), 0.001)
                            message = (
                                tqdm.format_meter(
                                    current,
                                    total,
                                    float(elapsed),
                                    prefix=label,
                                    rate=rate or None,
                                    unit="it",
                                    ncols=78,
                                    bar_format="{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                                )
                                + f" buffer={buffer}"
                            )
                            if current >= total:
                                active.pop(key, None)
                            else:
                                first = key not in active
                                active[key] = (message, tone)
                                active.pop("collect", None)
                                if interactive:
                                    clear_live()
                                    display_live()
                                    terminal.flush()
                                    continue
                                if not first:
                                    continue
                    else:
                        if match.group(1) == "EXEC":
                            count = re.search(r"(\d+)/(\d+)", message)
                            if count:
                                message = tqdm.format_meter(
                                    int(count[1]),
                                    int(count[2]),
                                    0,
                                    prefix="Episode chunks",
                                    bar_format="{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt}",
                                )
                        first = key not in active
                        active[key] = (message, tone)
                        if interactive:
                            clear_live()
                            display_live()
                            terminal.flush()
                            continue
                        if not first:
                            continue
                elif match and match.group(1) in {"LOAD", "SAVE", "INFER"}:
                    description, _, timing = message.rpartition(": ")
                    key = description
                    short = re.sub(
                        r"(?<!\S)/[^ ]+",
                        lambda m: m.group(0).rstrip("/").split("/")[-1],
                        description,
                    )
                    message = f"{short}: {timing}" if timing else message
                    if timing.endswith("s elapsed"):
                        first = key not in active
                        active[key] = (message, tone)
                        if interactive:
                            clear_live()
                            display_live()
                            terminal.flush()
                            continue
                        if not first:
                            continue
                    else:
                        active.pop(key, None)
                elif match and match.group(1) == "COLLECT":
                    count = re.search(r"(\d+)/(\d+)", message)
                    if count:
                        from tqdm import tqdm

                        if collect_started is None:
                            collect_started = time.monotonic()
                        message = tqdm.format_meter(
                            int(count[1]),
                            int(count[2]),
                            time.monotonic() - collect_started,
                            prefix="[COLLECT] Critical chunks",
                            bar_format="{desc}: {percentage:3.0f}%|{bar:12}| {n_fmt}/{total_fmt}",
                        )
                        previous = active.get("collect")
                        active["collect"] = (message, tone)
                        if interactive and count[1] != count[2]:
                            clear_live()
                            display_live()
                            terminal.flush()
                            continue
                        if previous == (message, tone):
                            continue
                        if count[1] == count[2]:
                            active.pop("collect", None)
            clear_live()
            terminal.write(
                f"\033[{tone}m{message}\033[0m\n" if color and tone else message + "\n"
            )
            display_live()
            terminal.flush()
        clear_live()
        terminal.flush()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    stream_run_log(
        sys.stdin,
        sys.stdout,
        args.log_file,
        color=sys.stdout.isatty() and "NO_COLOR" not in os.environ,
        compact=args.compact,
        interactive=sys.stdout.isatty(),
    )
