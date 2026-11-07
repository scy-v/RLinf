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
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.openpi.sampling.rtc_guidance import RTCGuidanceContext
from rlinf.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_suffix(".json")
logger = get_logger()
os.environ["EMBODIED_PATH"] = str(ROOT / "examples/embodiment")
torch.set_num_threads(4)
torch.manual_seed(20261106)
with initialize_config_dir(
    config_dir=str(ROOT / "evaluations/realworld"), version_base="1.3"
):
    cfg = compose(config_name="realworld_ur5e_eval_pi05_sft")
cfg.rollout.model.model_path = cfg.rollout.model.model_path.replace(
    "global_step_30000", "global_step_5000"
)
cfg.rollout.model.num_steps = 10
cfg.rollout.model.openpi.num_steps = 10
cfg.rollout.model.openpi.rtc_enabled = True
model = get_model(cfg.rollout.model).eval().requires_grad_(False).to("cuda")
assert model.num_steps == 10
obs = {
    "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "wrist_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "states": torch.zeros((1, 20)),
    "task_descriptions": ["Insert the black rack into the blue slot"],
}
result = {
    "metadata": {
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seed": 20261106,
        "cpu_threads": 4,
        "warmup": 5,
        "samples": 30,
        "num_steps": 10,
        "batch_size": 1,
        "input": "zero uint8 main/wrist 224x224; zero states (20)",
        "scope": "model.predict_action_batch; inference_mode; GPU-synchronized wall time; no robot/Ray",
        "model_config": OmegaConf.to_container(cfg.rollout.model, resolve=True),
        "rtc": {
            "mode": "approx",
            "executed_horizon": 2,
            "delay_steps": 2,
            "previous_actions": "fixed ordinary bootstrap model actions",
        },
    }
}


def bench(name, context=None):
    logger.info("Benchmark %s", name)
    torch.manual_seed(20261106)
    values = []
    with torch.inference_mode():
        for i in range(35):
            torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            started = time.perf_counter()
            actions, _ = model.predict_action_batch(
                env_obs=obs, mode="eval", rtc_context=context
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            assert actions.shape == (1, 10, 7) and torch.isfinite(actions).all()
            if i >= 5:
                values.append(elapsed)
    result[name] = {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": max(values),
        "samples_ms": values,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("%s: %.3f ms", name, np.mean(values))


bench("ordinary_original")
model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune")
bench("ordinary_compiled")
with torch.inference_mode():
    _, bootstrap = model.predict_action_batch(env_obs=obs, mode="eval")
    previous = bootstrap["model_actions"].detach().cpu().clone()
context = RTCGuidanceContext(
    prev_model_actions=previous, executed_horizon=2, delay_steps=2
)
model.rtc_guidance_clip = 5.0
bench("rtc_approx_clip5", context)
model.rtc_guidance_clip = 0.0
bench("rtc_approx_clip0", context)
bench("ordinary_compiled_repeat")
