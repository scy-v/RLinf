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

import argparse
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

from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import RealworldRLTRoute
from rlinf.models import get_model
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.sampling.rtc_guidance import RTCGuidanceContext
from rlinf.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[4]
logger = get_logger()
parser = argparse.ArgumentParser()
parser.add_argument("group", choices=["pi05_rtc", "rlt"])
parser.add_argument("--none-only", action="store_true")
args = parser.parse_args()
suffix = "_none_recheck" if args.none_only else ""
out = Path(__file__).with_name(f"inference_compile_{args.group}{suffix}.json")
os.environ["EMBODIED_PATH"] = str(ROOT / "examples/embodiment")
torch.set_num_threads(4)
torch.manual_seed(20261107)
with initialize_config_dir(
    config_dir=str(ROOT / "evaluations/realworld"), version_base="1.3"
):
    cfg = compose(config_name="realworld_ur5e_eval_pi05_sft")
if args.group == "rlt":
    training = OmegaConf.load(cfg.stage2.training_config)
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(training.rollout.rlt_feature_model, resolve=True)
    )
else:
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.rollout.model, resolve=True)
    )
    model_cfg.model_path = model_cfg.model_path.replace(
        "global_step_30000", "global_step_5000"
    )
model_cfg.num_steps = model_cfg.openpi.num_steps = 10
OmegaConf.update(model_cfg, "openpi.torch_compile", False, force_add=True)
OmegaConf.update(
    model_cfg, "openpi.rtc_enabled", args.group == "pi05_rtc", force_add=True
)
OmegaConf.update(model_cfg, "openpi.rtc_guidance_clip", 5.0, force_add=True)
OmegaConf.update(model_cfg, "openpi.rtc_guidance_mode", "approx", force_add=True)
model = get_model(model_cfg).eval().requires_grad_(False).to("cuda")
assert model.num_steps == 10 and "global_step_5000" in str(model_cfg.model_path)
obs = {
    "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "wrist_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
    "states": torch.zeros((1, 20)),
    "task_descriptions": ["Insert the black rack into the blue slot"],
}
report = {
    "metadata": {
        "group": args.group,
        "none_only": args.none_only,
        "rounds": 3 if args.none_only else 1,
        "seed": 20261107,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "num_steps": 10,
        "warmup": 5,
        "samples": 30,
        "batch_size": 1,
        "cpu_threads": 4,
        "scope": "model inference; synchronized GPU wall time; no robot/Ray/training update",
        "input": "zero uint8 main/wrist images 224x224; zero states (20)",
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "whole_compile_mode": "max-autotune",
        "fullgraph_argument": "omitted (False)",
        "no_compile_check": "zero new Dynamo graphs, local kernels disabled, sampler and MLP unwrapped",
    },
    "results": {},
}


def bench(name, fn, no_compile=False):
    logger.info("Starting %s", name)
    torch.manual_seed(20261107)
    before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    values = []
    with torch.inference_mode():
        for index in range(35):
            torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            started = time.perf_counter()
            actions, _ = fn()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            assert torch.isfinite(actions).all()
            assert actions.shape == (1, 10, 7), actions.shape
            if index >= 5:
                values.append(elapsed)
    after = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    if no_compile:
        assert before == after, "Uncompiled inference unexpectedly compiled a graph"
    report["results"][name] = {
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": max(values),
        "samples_ms": values,
        "new_graphs": after - before,
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("%s: %.3f ms", name, np.mean(values))


set_torch_compile(False)
if args.group == "pi05_rtc":
    with torch.inference_mode():
        _, initial = model.predict_action_batch(obs, mode="eval")
        previous = initial["model_actions"].detach().cpu().clone()
    context = RTCGuidanceContext(previous, executed_horizon=2, delay_steps=2)
    report["metadata"]["rtc_context"] = {
        "previous_plan": "fixed ordinary bootstrap",
        "executed_horizon": 2,
        "delay_steps": 2,
        "clip": 5.0,
    }

    def pi05():
        return model.predict_action_batch(obs, mode="eval")

    def rtc():
        return model.predict_action_batch(obs, mode="eval", rtc_context=context)

    if args.none_only:
        for round_index in range(3):
            paths = [("pi05", pi05), ("rtc", rtc)]
            if round_index % 2:
                paths.reverse()
            for name, fn in paths:
                bench(f"{name}_none_round{round_index + 1}", fn, no_compile=True)
        logger.info("Completed uncompiled recheck %s", args.group)
        raise SystemExit(0)

    bench("pi05_none", pi05, no_compile=True)
    bench("rtc_none", rtc, no_compile=True)
    set_torch_compile(True)
    bench("pi05_local", pi05)
    bench("rtc_local", rtc)
    model.sample_actions = torch.compile(model.sample_actions, mode="max-autotune")
    bench("pi05_whole", pi05)
    model.enable_rtc_torch_compile(mode="max-autotune")
    bench("rtc_whole", rtc)
else:
    head_cfg = OmegaConf.merge(training.actor.model, training.rollout.model)
    head = get_model(head_cfg).eval().requires_grad_(False).to("cuda")
    head.load_state_dict(
        torch.load(cfg.stage2.checkpoint_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    assert not head.torch_compile_enabled
    report["metadata"]["mlp_config"] = OmegaConf.to_container(head_cfg, resolve=True)
    report["metadata"]["mlp_checkpoint"] = str(cfg.stage2.checkpoint_path)
    report["metadata"]["training_config"] = str(cfg.stage2.training_config)
    report["metadata"]["rlt_call_mode"] = "train (rollout inference only)"
    route = RealworldRLTRoute(use_schedule=True, warmup_updates=150)

    def rlt():
        return predict_rlt_actions(
            policy_model=head,
            feature_model=model,
            rlt_route=route,
            env_obs=obs,
            final_obs=None,
            mode="train",
            version=200,
            rlt_switch_flags=torch.ones((1, 1), dtype=torch.bool),
        )

    if args.none_only:
        for round_index in range(3):
            bench(f"rlt_none_round{round_index + 1}", rlt, no_compile=True)
        logger.info("Completed uncompiled recheck %s", args.group)
        raise SystemExit(0)

    bench("rlt_none", rlt, no_compile=True)
    set_torch_compile(True)
    head.enable_torch_compile(mode="max-autotune-no-cudagraphs")
    with torch.inference_mode():
        features = model.extract_rlt_obs(obs)
        graphs_before_head = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        head.predict_action_batch(features, mode="train")
        graphs_after_head = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    report["metadata"]["mlp_graphs_after_enabling_compile"] = (
        graphs_after_head - graphs_before_head
    )
    report["metadata"]["mlp_actual_path"] = (
        "RLTMLPPolicy.predict_action_batch -> sac_forward; inherited compile hook wraps unused _sample_actions"
    )
    bench("rlt_local", rlt)
    model.enable_rlt_torch_compile(mode="max-autotune")
    bench("rlt_suffix", rlt)
    model.build_prefix_cache = torch.compile(
        model.build_prefix_cache, mode="max-autotune"
    )
    bench("rlt_prefix_suffix", rlt)
    model._encode_rlt_flat = torch.compile(model._encode_rlt_flat, mode="max-autotune")
    bench("rlt_prefix_z_suffix", rlt)
logger.info("Completed %s", args.group)
