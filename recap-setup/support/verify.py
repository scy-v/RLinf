#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Inspect RECAP imports and prove which checkout the environment loads."""

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from openpi.models.tokenizer import PaligemmaTokenizer

import rlinf

repo = Path(__file__).resolve().parents[2]
report = {
    "python": sys.executable,
    "repo": str(repo),
    "head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip(),
    "packages": {},
}
for name in (
    "rlinf",
    "torch",
    "openpi",
    "transformers",
    "jax",
    "datasets",
    "pyarrow",
    "ray",
    "hydra",
    "flash_attn",
    "libero.libero",
    "mani_skill",
):
    module = importlib.import_module(name)
    report["packages"][name] = {
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }

assert Path(rlinf.__file__).resolve().is_relative_to(repo / "rlinf")
assert Path(sys.prefix).resolve() == repo / ".venv"
report["cuda_available"] = torch.cuda.is_available()
report["gpu_count"] = torch.cuda.device_count()
if report["cuda_available"]:
    x = torch.ones(8, device="cuda")
    assert (x @ x).item() == 8
    torch.cuda.synchronize()
    report["gpu"] = torch.cuda.get_device_name(0)
    from flash_attn import flash_attn_func

    q = torch.ones(
        (1, 8, 2, 64), device="cuda", dtype=torch.float16, requires_grad=True
    )
    k = q.detach().clone().requires_grad_()
    v = q.detach().clone().requires_grad_()
    output = flash_attn_func(q, k, v)
    assert torch.allclose(output, torch.ones_like(output), atol=1e-3)
    output.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    report["flash_attention_forward_backward"] = "passed"
tokenizer_file = (
    Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
)
assert tokenizer_file.is_file(), tokenizer_file

ids, mask = PaligemmaTokenizer().tokenize("verify installation")
assert ids.shape == mask.shape and mask.any()
report["tokenizer"] = str(tokenizer_file)
physx = Path(os.environ["PHYSX_DIR"]) / "libPhysXGpu_64.so"
assert physx.is_file() and physx.stat().st_size > 0
report["physx"] = str(physx)
report["entrypoints"] = {}
entrypoints = (
    (
        "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py",
        "recap_compute_returns",
    ),
    (
        "examples/offline_rl/advantage_labeling/recap/train_value.py",
        "recap_value_model_sft",
    ),
    (
        "examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py",
        "recap_compute_advantages",
    ),
    ("examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py", "cfg_rl_openpi"),
)
for script, config in entrypoints:
    result = subprocess.run(
        [
            sys.executable,
            str(repo / script),
            "--config-path",
            str(repo / "examples/offline_rl/config"),
            "--config-name",
            config,
            "--cfg",
            "job",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=180,
    )
    log = repo / "recap-setup/logs" / f"verify-{config}.log"
    log.write_text(result.stdout + result.stderr)
    report["entrypoints"][config] = {"exit_code": result.returncode, "log": str(log)}
report["distributions"] = {
    d.metadata["Name"]: d.version
    for d in importlib.metadata.distributions()
    if d.metadata["Name"]
}
(repo / "recap-setup/support/versions.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "distributions"}, indent=2))

assert all(x["exit_code"] == 0 for x in report["entrypoints"].values()), (
    "See verify-*.log for entrypoint failures"
)
