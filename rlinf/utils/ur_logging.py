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

"""Record files and latest links for single-robot UR evaluation runs."""

import json
import os
import threading
from pathlib import Path
from typing import Any

import yaml


def publish_latest(target: Path, link: Path) -> None:
    """Atomically point a latest link at an absolute run-file path."""
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{os.getpid()}.{threading.get_ident()}")
    try:
        temporary.symlink_to(target.resolve())
        temporary.replace(link)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_run_logs(log_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Create a run's log directory and publish its YAML and JSONL links."""
    directory = log_dir / "all_logs"
    directory.mkdir(parents=True, exist_ok=True)
    yaml_path = directory / f"log_{run_id}.yaml"
    timing_path = directory / f"timing_{run_id}.jsonl"
    publish_latest(yaml_path, log_dir / "latest.yaml")
    publish_latest(timing_path, log_dir / "latest.jsonl")
    return yaml_path, timing_path


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one complete record using the VLA YAML or JSONL format."""
    if path.suffix == ".jsonl":
        content = json.dumps(record) + "\n"
    else:
        content = yaml.safe_dump(
            record, sort_keys=False, default_flow_style=None, allow_unicode=True
        )
        content += " " + "-" * 60 + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(content)
