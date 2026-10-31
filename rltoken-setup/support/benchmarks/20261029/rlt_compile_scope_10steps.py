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
import statistics
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import RealworldRLTRoute
from rlinf.models import get_model

# Model-only benchmark: no robot, environment, RTDE, camera or gripper objects.
torch.set_num_threads(4)
cfg = OmegaConf.load(
    "logs/online_rltoken/20261028-190237-ur_rlt_stage2_ac_mlp/tensorboard/config.yaml"
)
print("LOAD model", flush=True)
f = get_model(cfg.rollout.rlt_feature_model).to("cuda").eval().requires_grad_(False)
p = get_model(OmegaConf.merge(cfg.actor.model, cfg.rollout.model)).to("cuda").eval()
p.enable_torch_compile(mode="max-autotune-no-cudagraphs")
obs = {
    "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "wrist_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "states": torch.zeros((1, 20)),
    "task_descriptions": ["Insert the black rack into the blue slot"],
}
results = {
    "metadata": {
        "num_steps": 10,
        "warmup": 5,
        "samples": 30,
        "batch_size": 1,
        "input": "zero synthetic main/wrist uint8 224x224; zero states",
        "context": "torch.inference_mode; synchronized GPU wall time",
        "checkpoint": str(cfg.rollout.rlt_feature_model.model_path),
        "scope": "model only; no hardware or Ray",
    }
}


def bench(name, fn, n=30, ctx=torch.inference_mode):
    print("START " + name, flush=True)
    with ctx():
        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            fn()
        vals = []
        for _ in range(n):
            torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            vals.append((time.perf_counter() - t) * 1000)
    vals.sort()
    res = {
        "n": n,
        "mean_ms": statistics.mean(vals),
        "p50_ms": statistics.median(vals),
        "p95_ms": vals[int(0.95 * (n - 1))],
        "max_ms": max(vals),
        "samples_ms": vals,
    }
    results[name] = res
    print(name, json.dumps(res), flush=True)
    with open("/tmp/rlt_compile_scope_10steps.json", "w") as out:
        json.dump(results, out, indent=2)


route = RealworldRLTRoute(use_schedule=True, warmup_updates=150)


def online():
    return predict_rlt_actions(
        policy_model=p,
        feature_model=f,
        rlt_route=route,
        env_obs=obs,
        final_obs=None,
        mode="train",
        version=200,
        rlt_switch_flags=torch.ones((1, 1), dtype=torch.bool),
    )


original_prefix = f.build_prefix_cache
original_z = f._encode_rlt_flat
original_suffix = f._sample_actions_from_prefix_cache
original_sample = f.sample_actions
f.num_steps = 10
bench("rlt10_existing", online)
bench("eval10_existing", lambda: f.predict_action_batch(env_obs=obs, mode="eval"))
f.sample_actions = torch.compile(original_sample, mode="max-autotune")
bench("eval10_whole_sampler", lambda: f.predict_action_batch(env_obs=obs, mode="eval"))
f.sample_actions = original_sample
f._sample_actions_from_prefix_cache = torch.compile(
    original_suffix, mode="max-autotune"
)
bench("rlt10_suffix_only", online)
f.build_prefix_cache = torch.compile(original_prefix, mode="max-autotune")
bench("rlt10_suffix_plus_prefix", online)
f._encode_rlt_flat = torch.compile(original_z, mode="max-autotune")
bench("rlt10_suffix_prefix_z", online)
f.build_prefix_cache = original_prefix
f._encode_rlt_flat = original_z
bench("rlt10_suffix_only_repeat", online)
f._sample_actions_from_prefix_cache = original_suffix
bench("rlt10_existing_repeat", online)
print("DONE; no hardware connections", flush=True)
