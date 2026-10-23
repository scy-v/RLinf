#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Run the official NVIDIA OpenPI install with this workstation's paths."""

import argparse
import importlib.util
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
spec = importlib.util.spec_from_file_location(
    "rlinf_install_driver", REPO / ".agents/skills/test-install/driver.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dry-run", action="store_true")
args, extra = parser.parse_known_args()
job = next(
    j for j in driver.load_jobs() if j.name == "embodied-openpi-maniskill-libero-test"
)
# CI points to /workspace/dataset. Keep its installer command and place all
# mutable files in this checkout; --no-root uses the host's existing toolchain.
command = next(
    line
    for line in job.install_run.splitlines()
    if line.startswith("bash requirements/install.sh")
)
job.install_run = command + " --no-root --install-rlinf"
wheel = (
    REPO
    / "recap-setup/support/wheels/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)
use_local_wheel = (
    wheel.is_file()
    and not extra
    and '"torch==2.11.0"' in (REPO / "pyproject.toml").read_text()
)
if use_local_wheel:
    job.install_run += " --no-flash-attn"
if extra:
    job.install_run += " " + shlex.join(extra)
script = "set -e\n" + driver._install_script(job, str(REPO / ".venv"), mirror=False)
if use_local_wheel:
    python = REPO / ".venv/bin/python"
    check = "import torch; assert torch.__version__.startswith('2.11.') and torch.version.cuda.startswith('13.') and torch._C._GLIBCXX_USE_CXX11_ABI"
    script += "\n" + shlex.join([str(python), "-c", check])
    script += "\n" + shlex.join(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]
    )
(REPO / "recap-setup/support/generated-install.sh").write_text(script + "\n")
print("Official CI job:", job.name, flush=True)
sys.exit(driver._run_shell(script, args.dry_run))
