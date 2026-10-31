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

"""Offline VLA-Precision policy and conversion timings; no hardware objects."""

import ast
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import torch
import yaml

from evaluations.realworld.vla_precision_ur import image_tools
from evaluations.realworld.vla_precision_ur.policy import create_policy

ROOT = Path("/home/suchenyu/RLinf-rltoken")
OUT = Path("/tmp/vla_chunk_gap_no_motion.json")


def stats(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[int((len(values) - 1) * 0.95)],
        "max_ms": max(values),
        "samples_ms": values,
    }


# Compile only this pure method's AST, without importing the hardware module.
tree = ast.parse((ROOT / "evaluations/realworld/vla_precision_ur/ur.py").read_text())
method = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and node.name == "_transfer_obs_state"
)
namespace = {"np": np, "image_tools": image_tools, "Any": object}
exec(compile(ast.Module(body=[method], type_ignores=[]), "ur.py", "exec"), namespace)
transfer = namespace["_transfer_obs_state"]
receiver = SimpleNamespace(fix_gripper=False)
rng = np.random.default_rng(42)
raw = {
    "tcp_pose": np.zeros(6),
    "tcp_speed": np.zeros(6),
    "tcp_force": np.zeros(6),
    "gripper_position": 0.0,
    "exterior_image": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
    "wrist_image": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
    "prompt": "Insert the black rack into the blue slot",
}
config = yaml.safe_load(
    (ROOT / "evaluations/realworld/vla_precision_ur.yaml").read_text()
)
policy = create_policy(config)
results = {
    "metadata": {
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": config["model"]["checkpoint_dir"],
        "input": "synthetic 640x480 exterior/wrist uint8, zero 19-state",
        "warmup": 5,
        "samples": 40,
        "scope": "real policy infer and pure observation conversion only; no camera reads, RTDE, gripper, recorder, Ray or training",
        "simulated_actions": "3 sleeps of 20ms per chunk; timings are offline surrogate, not measured hardware cadence",
    }
}
for steps in [10, 4]:
    policy.model.num_steps = steps
    print(f"WARMUP {steps} steps", flush=True)
    for _ in range(5):
        torch.compiler.cudagraph_mark_step_begin()
        policy.infer(transfer(receiver, raw))
    torch.cuda.synchronize()
    values = {
        k: []
        for k in [
            "observation_conversion",
            "infer_outer",
            "infer_internal",
            "action_slice",
            "boundary_no_action",
            "last_start_to_next_start",
            "chunk_start_to_start",
        ]
    }
    previous_last_start = None
    previous_chunk_start = None
    for _ in range(41):
        started = time.perf_counter()
        obs = transfer(receiver, raw)
        transferred = time.perf_counter()
        torch.compiler.cudagraph_mark_step_begin()
        prediction = policy.infer(obs)
        inferred = time.perf_counter()
        actions = np.asarray(prediction["actions"])[..., :7]
        sliced = time.perf_counter()
        assert actions.shape[0] >= 3 and np.isfinite(actions).all()
        chunk_start = time.perf_counter()
        if previous_last_start is not None:
            values["observation_conversion"].append((transferred - started) * 1000)
            values["infer_outer"].append((inferred - transferred) * 1000)
            values["infer_internal"].append(prediction["policy_timing"]["infer_ms"])
            values["action_slice"].append((sliced - inferred) * 1000)
            values["boundary_no_action"].append((chunk_start - started) * 1000)
            values["last_start_to_next_start"].append(
                (chunk_start - previous_last_start) * 1000
            )
            values["chunk_start_to_start"].append(
                (chunk_start - previous_chunk_start) * 1000
            )
        previous_chunk_start = chunk_start
        for _ in range(3):
            previous_last_start = time.perf_counter()
            time.sleep(0.02)
    results[f"steps_{steps}"] = {k: stats(v) for k, v in values.items()}
    OUT.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "samples_ms"}
                for k, v in results[f"steps_{steps}"].items()
            }
        ),
        flush=True,
    )
print(f"DONE {OUT}; no hardware connected", flush=True)
