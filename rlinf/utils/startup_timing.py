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

"""Scoped startup diagnostics, inactive during ordinary rollout and learning."""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from types import TracebackType
from typing import Any

_ACTIVE = ContextVar("startup_timing", default=None)
_NO_TIMING = nullcontext()


def _process_usage() -> dict[str, int | float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values = {
        "cpu_sec": usage.ru_utime + usage.ru_stime,
        "major_faults": usage.ru_majflt,
        "minor_faults": usage.ru_minflt,
    }
    try:
        fields = dict(
            line.split(": ") for line in Path("/proc/self/io").read_text().splitlines()
        )
        values["read_bytes"] = int(fields["read_bytes"])
    except (OSError, KeyError, ValueError):
        pass
    return values


def _compiler_metrics() -> list[Any]:
    # Do not import the compiler merely to observe an uncompiled invocation.
    module = sys.modules.get("torch._dynamo.utils")
    getter = getattr(module, "get_compilation_metrics", None)
    return getter() if getter is not None else []


def startup_section(name: str, **metadata: Any) -> AbstractContextManager:
    """Measure startup work only inside a StartupTiming scope."""
    active = _ACTIVE.get()
    return active.section(name, **metadata) if active is not None else _NO_TIMING


def startup_scope(
    phase: str, log: Callable[[str], None], **options: Any
) -> AbstractContextManager:
    """Reuse an enclosing scope for the same startup phase."""
    active = _ACTIVE.get()
    if active is not None and active.phase == phase:
        return _NO_TIMING
    return StartupTiming(phase, log, **options)


def synchronize_startup_device() -> None:
    """Drain an initialized accelerator without creating a device context."""
    from rlinf.scheduler import Worker

    platform = Worker.torch_platform
    if (
        Worker.torch_device_type != "cpu"
        and getattr(platform, "is_initialized", lambda: True)()
    ):
        platform.synchronize()


def record_worker_startup(worker: Any, event: str, **fields: Any) -> None:
    """Persist per-worker startup records beside the run's regular logs."""
    from rlinf.utils.ur_logging import append_record

    directory = worker.cfg.get("runner", {}).get("logger", {}).get("log_path")
    if directory is None:
        return
    path = (
        Path(directory)
        / "startup_timing"
        / f"{fields.get('component', 'worker')}_{fields.get('rank', 0)}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    append_record(path, {"event": event, **fields})


class StartupTiming:
    """Record startup wall time, process I/O and compiler diagnostics.

    Args:
        phase: Startup phase, such as load or warmup.
        log: Logger callback receiving readable and JSON records.
        record: Optional callback accepting an event name and keyword fields.
        synchronize: Optional accelerator barrier, used only at startup boundaries.
        enabled: Disable all measurement, process probes and barriers when false.
        context: Run-local identifiers included in each record.

    Stage durations include their children. Compiler times are explanatory and
    must not be added to stage durations. Process counters include other threads.
    """

    def __init__(
        self,
        phase: str,
        log: Callable[[str], None],
        *,
        record: Callable[..., None] | None = None,
        synchronize: Callable[[], None] | None = None,
        enabled: bool = True,
        **context: Any,
    ) -> None:
        self.phase, self.log, self.record = phase, log, record
        self.synchronize, self.enabled = synchronize, enabled
        self.context = context

    def __enter__(self) -> StartupTiming:
        if not self.enabled:
            self.token = _ACTIVE.set(None)
            return self
        self.stack, self.records = [], []
        self.compiler_before = {id(value) for value in _compiler_metrics()}
        self.started = time.perf_counter()
        self.usage = _process_usage()
        self.token = _ACTIVE.set(self)
        return self

    @contextmanager
    def section(self, name: str, **metadata: Any) -> Iterator[None]:
        path = "/".join((*self.stack, name))
        depth = len(self.stack)
        self.stack.append(name)
        usage = _process_usage()
        started = time.perf_counter()
        status = "complete"
        try:
            yield
            if self.synchronize is not None:
                self.synchronize()
        except BaseException:
            status = "failed"
            raise
        finally:
            seconds = time.perf_counter() - started
            after = _process_usage()
            self.stack.pop()
            self.records.append(
                {
                    "stage": path,
                    "depth": depth,
                    "duration_sec": seconds,
                    "timestamp": time.time(),
                    "status": status,
                    **{
                        key: after[key] - value
                        for key, value in usage.items()
                        if key in after
                    },
                    **metadata,
                }
            )

    def _emit(self, event: str, **fields: Any) -> None:
        fields = {
            "phase": self.phase,
            "timestamp": time.time(),
            **self.context,
            **fields,
        }
        if self.record is not None:
            self.record(event, **fields)
        else:
            self.log("[STARTUP_TIMING] " + json.dumps({"event": event, **fields}))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ACTIVE.reset(self.token)
        if not self.enabled:
            return
        seconds = time.perf_counter() - self.started
        usage = _process_usage()
        for row in self.records:
            self.log(
                f"[{'LOAD' if self.phase == 'load' else 'INFER'}] {row['stage']}: {row['status']} "
                f"after {row['duration_sec']:.3f}s "
                f"cpu={row['cpu_sec']:.3f}s "
                f"read_bytes={row.get('read_bytes', 'n/a')} "
                f"major_faults={row['major_faults']}"
            )
            self._emit("startup_stage", **row)
        covered = sum(row["duration_sec"] for row in self.records if row["depth"] == 0)
        self._emit(
            "startup_summary",
            duration_sec=seconds,
            unattributed_sec=max(0.0, seconds - covered),
            synchronized=self.synchronize is not None,
            status="complete" if exc_type is None else "failed",
            **{
                key: usage[key] - value
                for key, value in self.usage.items()
                if key in usage
            },
        )
        for metric in _compiler_metrics():
            if id(metric) in self.compiler_before:
                continue
            fields = {}
            for name in (
                "entire_frame_compile_time_s",
                "backend_compile_time_s",
                "inductor_compile_time_s",
                "code_gen_time_s",
                "runtime_cudagraphify_time_us",
                "runtime_triton_autotune_time_us",
                "triton_compile_time_us",
                "inductor_fx_local_cache_hit_count",
                "inductor_fx_local_cache_miss_count",
                "fail_type",
            ):
                value = getattr(metric, name, None)
                if value is not None:
                    fields[name] = value
            self._emit(
                "startup_compile",
                compile_id=str(getattr(metric, "compile_id", "unknown")),
                **fields,
            )
