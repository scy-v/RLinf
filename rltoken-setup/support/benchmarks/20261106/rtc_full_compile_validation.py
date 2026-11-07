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

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

os.environ["EMBODIED_PATH"] = "/home/suchenyu/RLinf-rltoken/examples/embodiment"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from hydra import compose, initialize_config_dir

from rlinf.models import get_model
from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
    RTCGuidanceContext,
    sample_actions_with_rtc_guidance,
)

logger = get_logger()
ROOT = Path(__file__).resolve().parents[4]
REFERENCE = "5ba46cc0f133933fd21459eaa7773915a55c38ef"
source = subprocess.check_output(
    [
        "git",
        "show",
        f"{REFERENCE}:rlinf/models/embodiment/openpi/sampling/rtc_guidance.py",
    ],
    cwd=ROOT,
    text=True,
)
legacy = ModuleType("legacy_rtc")
sys.modules[legacy.__name__] = legacy
exec(compile(source, "legacy_rtc.py", "exec"), legacy.__dict__)
torch.manual_seed(20261106)
torch.set_num_threads(4)
with initialize_config_dir(
    config_dir="/home/suchenyu/RLinf-rltoken/evaluations/realworld", version_base="1.3"
):
    cfg = compose(config_name="realworld_ur5e_eval_pi05_sft")
cfg.rollout.model.model_path = cfg.rollout.model.model_path.replace(
    "global_step_30000", "global_step_5000"
)
cfg.rollout.model.num_steps = cfg.rollout.model.openpi.num_steps = 10
cfg.rollout.model.openpi.rtc_enabled = True
cfg.rollout.model.openpi.rtc_guidance_mode = "approx"
cfg.rollout.model.openpi.rtc_guidance_clip = 5.0
m = get_model(cfg.rollout.model).eval().requires_grad_(False).to("cuda")
obs = {
    "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "wrist_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "states": torch.zeros((1, 20)),
    "task_descriptions": ["Insert the black rack into the blue slot"],
}
r = {
    "metadata": {
        "reference_commit": REFERENCE,
        "sampler_compile": {"mode": "max-autotune", "fullgraph": False},
        "seed": 20261106,
        "num_steps": 10,
        "warmup": 5,
        "timed_calls": 30,
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model_config": OmegaConf.to_container(cfg.rollout.model, resolve=True),
        "scope": "model only; synthetic zero images; GPU synchronized; no robot or Ray; fixed-noise parity; fixed previous plan",
    },
    "cases": [],
    "benchmark": {},
}


def save():
    Path(__file__).with_suffix(".json").write_text(json.dumps(r, indent=2))


with torch.inference_mode():
    observation = m.env_obs_to_observation(obs)
    noise = torch.randn((1, 20, 32), device="cuda")
    prev = torch.randn((1, 20, 32))
    contexts = [(2, 1), (2, 2), (3, 3), (4, 4), (20, 2), (0, 0)]
    for e, d in contexts:
        c = RTCGuidanceContext(prev, e, d)
        g = torch.Generator(device="cuda").manual_seed(10)
        a = legacy.sample_actions_with_rtc_guidance(
            m, observation, c, num_steps=10, noise=noise, rng=g
        )
        rng_state = g.get_state()
        g.manual_seed(10)
        b = sample_actions_with_rtc_guidance(
            m, observation, c, num_steps=10, noise=noise, rng=g
        )
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.equal(rng_state, g.get_state())
    r["eager_legacy_exact_and_rng_parity"] = True
    logger.info("Eager legacy parity and RNG advancement PASS")
    m.enable_rtc_torch_compile(mode="max-autotune")
    graph_count = None
    for e, d in contexts:
        c = RTCGuidanceContext(prev, e, d)
        torch.compiler.cudagraph_mark_step_begin()
        a = legacy.sample_actions_with_rtc_guidance(
            m, observation, c, num_steps=10, noise=noise
        ).clone()
        started = time.perf_counter()
        b = sample_actions_with_rtc_guidance(
            m, observation, c, num_steps=10, noise=noise, sampler=m._rtc_sampler
        ).clone()
        torch.cuda.synchronize()
        aa = m.decode_actions(a, observation.state)
        bb = m.decode_actions(b, observation.state)
        row = {
            "executed": e,
            "delay": d,
            "call_sec": time.perf_counter() - started,
            "model_max_abs": float((a - b).abs().max()),
            "action_max_abs": float((aa - bb).abs().max()),
            "unique_graphs": dict(torch._dynamo.utils.counters["stats"]),
        }
        count = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        if graph_count is None:
            graph_count = count
        assert count == graph_count, "Changing RTC delays must reuse the compiled graph"
        r["cases"].append(row)
        save()
        logger.info("%s", row)
        torch.testing.assert_close(aa[..., :6], bb[..., :6], rtol=0.02, atol=0.002)
        threshold = cfg.env.eval.override_cfg.gripper_close_threshold
        torch.testing.assert_close(aa[..., 6] > threshold, bb[..., 6] > threshold)
        row["arm_max_abs"] = float((aa[..., :6] - bb[..., :6]).abs().max())
        row["gripper_decisions_equal"] = True
        save()
    c = RTCGuidanceContext(prev, 2, 2)
    for name, sampler in [("legacy_local", None), ("compiled_full", m._rtc_sampler)]:
        vals = []
        for i in range(35):
            torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            t = time.perf_counter()
            if sampler is None:
                model_actions = legacy.sample_actions_with_rtc_guidance(
                    m, m.env_obs_to_observation(obs), c, num_steps=10
                )
                out = m.decode_actions(model_actions, observation.state)
            else:
                out = m.predict_action_batch(env_obs=obs, mode="eval", rtc_context=c)[0]
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) * 1000
            if i >= 5:
                vals.append(dt)
        r["benchmark"][name] = {"mean_ms": sum(vals) / len(vals), "samples_ms": vals}
        save()
        logger.info("%s: %s", name, r["benchmark"][name]["mean_ms"])
    logger.info("PASS")
