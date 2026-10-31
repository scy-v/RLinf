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

"""Real Ray/learner/rollout startup probe; no physical environment is constructed."""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import psutil
import ray
import torch
from omegaconf import OmegaConf

from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.robot_lifecycle import termination_signals
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import AsyncRLTACFSDPPolicy
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


class SyntheticEnvWorker(Worker):
    """Replace only the physical hardware boundary during startup."""

    def init_worker(self):
        self.ready = True
        return {"synthetic_environment_ready": True, "physical_hardware": False}


class MemoryProbeRollout(AsyncMultiStepRolloutWorker):
    def __init__(self, cfg, output, loader):
        super().__init__(cfg)
        self.probe_output = output
        self.probe_loader = loader
        self.probe_stage = "before_model_load"

    def init_worker(self):
        stop = threading.Event()
        process = psutil.Process()
        original_load = torch.load

        def load(*args, **kwargs):
            self.probe_stage = "checkpoint_load"
            if self.probe_loader == "copy":
                kwargs["mmap"] = False
            result = original_load(*args, **kwargs)
            self.probe_stage = "checkpoint_copy_and_dtype_conversion"
            return result

        def sample():
            with open(self.probe_output + ".memory.jsonl", "w") as stream:
                while not stop.is_set():
                    info = process.memory_full_info()
                    vm = psutil.virtual_memory()
                    record = {
                        "time": time.time(),
                        "stage": self.probe_stage,
                        "rss_gib": info.rss / 2**30,
                        "uss_gib": info.uss / 2**30,
                        "swap_gib": info.swap / 2**30,
                        "host_used_gib": vm.used / 2**30,
                        "host_unavailable_gib": (vm.total - vm.available) / 2**30,
                    }
                    for line in Path("/proc/self/status").read_text().splitlines():
                        if line.startswith("RssAnon:"):
                            record["rss_anon_gib"] = int(line.split()[1]) / 2**20
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    stop.wait(0.25)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        torch.load = load
        try:
            super().init_worker()
            self.probe_stage = "model_init_complete"
            time.sleep(0.5)
        finally:
            torch.load = original_load
            stop.set()
            thread.join()
        rows = [json.loads(line) for line in open(self.probe_output + ".memory.jsonl")]
        return {
            "peak_rss_gib": max(row["rss_gib"] for row in rows),
            "peak_uss_gib": max(row["uss_gib"] for row in rows),
            "peak_host_unavailable_gib": max(
                row["host_unavailable_gib"] for row in rows
            ),
            "last": rows[-1],
        }

    def warmup_rlt_inference(self):
        if self.cfg.rollout.get("rlt_inference_warmup", {}).get("enabled", False):
            self.probe_stage = "inference_warmup"
        super().warmup_rlt_inference()

    def delayed_warmup(self):
        self.cfg.rollout.rlt_inference_warmup.enabled = True
        started = time.perf_counter()
        self.warmup_rlt_inference()
        return 1000 * (time.perf_counter() - started)

    def synthetic_inference(self):
        obs = {
            "main_images": torch.randint(256, (1, 224, 224, 3), dtype=torch.uint8),
            "wrist_images": torch.randint(256, (1, 224, 224, 3), dtype=torch.uint8),
            "states": torch.zeros(1, 20),
            "task_descriptions": [self.cfg.env.train.override_cfg.task_description],
        }
        durations = []
        for _ in range(5):
            started = time.perf_counter()
            actions, _ = self._predict_rollout_actions(obs)
            self.torch_platform.synchronize()
            assert torch.isfinite(actions).all()
            durations.append(1000 * (time.perf_counter() - started))
        return durations

    def finish_probe(self):
        from torch._inductor.async_compile import shutdown_compile_workers

        shutdown_compile_workers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--order", choices=["before-reset", "after-reset"], required=True
    )
    parser.add_argument("--loader", choices=["copy", "default"], default="default")
    args = parser.parse_args()
    output = str(Path(args.output).resolve())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    # Remove hardware registration entirely, including UR connection parameters.
    cfg.cluster.node_groups = []
    cfg.cluster.component_placement.env = 0
    cfg.runner.logger.log_path = output + ".run"
    cfg.runner.logger.logger_backends = []
    cfg.runner.per_worker_log_path = None
    cfg.rollout.rlt_inference_warmup = {
        "enabled": args.order == "before-reset",
        "iterations": 3,
        "image_size": 224,
        "state_dim": 20,
    }
    # Never read or write the real run's replay cache.
    cfg.algorithm.rlt_schedule.warmup_cache_path = None
    report = {"order": args.order, "loader": args.loader, "physical_hardware": False}
    try:
        with termination_signals():
            cluster = Cluster(cluster_cfg=cfg.cluster)
            placement = HybridComponentPlacement(cfg, cluster)
            actor = AsyncRLTACFSDPPolicy.create_group(cfg).launch(
                cluster,
                name=cfg.actor.group_name,
                placement_strategy=placement.get_strategy("actor"),
            )
            rollout = MemoryProbeRollout.create_group(cfg, output, args.loader).launch(
                cluster,
                name=cfg.rollout.group_name,
                placement_strategy=placement.get_strategy("rollout"),
            )
            env = SyntheticEnvWorker.create_group().launch(
                cluster,
                name=cfg.env.group_name,
                placement_strategy=NodePlacementStrategy([0]),
            )
            runner = AsyncEmbodiedRunner(cfg, actor, rollout, env, None)
            runner.init_workers()
            report["initial_weight_sync"] = "pending"
            runner.update_rollout_weights()
            report["initial_weight_sync"] = "passed"
            if args.order == "after-reset":
                report["delayed_warmup_ms"] = rollout.delayed_warmup().wait()
            report["inference_ms"] = rollout.synthetic_inference().wait()
            rollout.finish_probe().wait()
            report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        Path(output + ".json").write_text(json.dumps(report, indent=2))
        ray.shutdown()
        logging.getLogger(__name__).info("Startup probe: %s", report)


if __name__ == "__main__":
    main()
