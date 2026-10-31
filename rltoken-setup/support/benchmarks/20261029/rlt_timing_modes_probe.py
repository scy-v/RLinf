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

"""Exercise timing modes with real Ray/FSDP/OpenPI and a hardware-free UR env."""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import ray
from omegaconf import OmegaConf

from rlinf.config import resolve_chunk_limits
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.robot_lifecycle import termination_signals
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import AsyncRLTACFSDPPolicy
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


class ProbeActor(AsyncRLTACFSDPPolicy):
    async def ingest_episode(self):
        initial = self.total_episodes_added
        while self.total_episodes_added == initial:
            self._drain_received_trajectories()
            await asyncio.sleep(0.01)
        return self.total_episodes_added


class HardwareFreeEnv(AsyncEnvWorker):
    def init_worker(self):
        import torch

        from rlinf.envs.real.env import RealWorldEnv
        from rlinf.robotics import RobotInfo, UR5eConfig

        class CriticalDummyEnv(RealWorldEnv):
            # Dummy env skips keyboard devices; supply the operator phase signal here.
            def reset(self, **kwargs):
                obs, infos = super().reset(**kwargs)
                infos["rlt_switch_flags"] = torch.ones(self.num_envs, dtype=torch.bool)
                return obs, infos

            def step(self, actions, **kwargs):
                obs, rewards, terminations, truncations, infos = super().step(
                    actions, **kwargs
                )
                infos["rlt_switch_flags"] = torch.ones(self.num_envs, dtype=torch.bool)
                return obs, rewards, terminations, truncations, infos

        assert self.cfg.env.train.override_cfg.is_dummy
        assert not self.cfg.cluster.node_groups
        # Construct the real env/wrappers without node-level realworld/ROS setup.
        self.worker_info.hardware_infos = [
            RobotInfo(
                type="UR5e",
                model="UR5e",
                config=UR5eConfig(
                    node_rank=0, camera_serials=["dummy-front", "dummy-wrist"]
                ),
            )
        ]
        self.update_env_cfg()
        self.env_list = self._setup_env_and_wrappers(
            CriticalDummyEnv, self.cfg.env.train, self.train_num_envs_per_stage
        )
        self._init_env()

    async def probe_episode(self, input_channel, rollout_channel, replay_ack_channel):
        self._replay_ack_channel = replay_ack_channel
        await self._run_interact_once(
            input_channel, rollout_channel, None, cooperative_yield=True
        )
        return self._trajectory_step


class ProbeRollout(AsyncMultiStepRolloutWorker):
    def finish_probe(self):
        from torch._inductor.async_compile import shutdown_compile_workers

        shutdown_compile_workers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=["disabled", "basic", "detailed"], required=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.runner.timing_mode = args.mode
    cfg.runner.logger.log_path = args.output + ".run"
    cfg.runner.logger.logger_backends = []
    cfg.runner.per_worker_log_path = None
    # Remove all physical device registration, and additionally use dummy-only UR execution.
    cfg.cluster.node_groups = []
    cfg.cluster.component_placement.env = 0
    cfg.env.train.override_cfg.is_dummy = True
    cfg.env.train.override_cfg.enable_camera_player = False
    cfg.env.train.override_cfg.camera_names = {
        "dummy-front": "exterior_image",
        "dummy-wrist": "wrist_image",
    }
    cfg.env.train.override_cfg.payload_settle_seconds = 0
    cfg.env.train.override_cfg.ft_zero_settle_seconds = 0
    cfg.env.train.teleop = "none"
    cfg.env.train.video_cfg.save_video = False
    cfg.env.train.max_episode_chunks = 3
    cfg.env.train.max_chunks_per_rollout_epoch = 3
    cfg.env.train.keyboard_wrapper_kwargs = {
        "start_with_actor": True,
        "success_key": None,
        "wait_for_start": False,
        "wait_for_reset": False,
        "wait_for_first_reset": False,
    }
    cfg.algorithm.rlt_schedule.warmup_cache_path = None
    cfg.algorithm.rlt_schedule.warmup_min_size = 1
    cfg.algorithm.rlt_schedule.warmup_post_collect_updates = 0
    cfg.algorithm.replay_buffer.min_buffer_size = 1
    resolve_chunk_limits(cfg, cfg.actor.model.num_action_chunks)
    report = {"mode": args.mode, "physical_hardware": False, "episodes": []}
    actor = env = rollout = None
    try:
        with termination_signals():
            cluster = Cluster(cluster_cfg=cfg.cluster)
            placement = HybridComponentPlacement(cfg, cluster)
            actor = ProbeActor.create_group(cfg).launch(
                cluster,
                name=cfg.actor.group_name,
                placement_strategy=placement.get_strategy("actor"),
            )
            rollout = ProbeRollout.create_group(cfg).launch(
                cluster,
                name=cfg.rollout.group_name,
                placement_strategy=placement.get_strategy("rollout"),
            )
            env = HardwareFreeEnv.create_group(cfg).launch(
                cluster,
                name=cfg.env.group_name,
                placement_strategy=placement.get_strategy("env"),
            )
            runner = AsyncEmbodiedRunner(cfg, actor, rollout, env, None)
            runner.init_workers()
            runner.update_rollout_weights()
            actor.recv_rollout_trajectories(
                runner.actor_channel, replay_ack_channel=runner.replay_ack_channel
            )
            for _ in range(2):
                ingestion = actor.ingest_episode()
                rollout_work = rollout.generate_one_epoch(
                    runner.rollout_channel, runner.env_channel, runner.actor_channel
                )
                env_work = env.probe_episode(
                    runner.env_channel,
                    runner.rollout_channel,
                    runner.replay_ack_channel,
                )
                report["episodes"].append(env_work.wait())
                rollout_work.wait()
                ingestion.wait()
                training = actor.run_training().wait()[0]
                assert training["rlt/critic_updates_run"] == 8
                assert training["rlt/actor_updates_run"] == 2
                report["updates"] = int(training["rlt/update_step"])
                runner.update_rollout_weights()
            records = []
            session = Path(ray._private.worker._global_node.get_session_dir_path())
            for log_file in (session / "logs").glob("worker-*.out"):
                for line in log_file.read_text(errors="replace").splitlines():
                    if "[TIMING_RECORD] " in line:
                        records.append(json.loads(line.split("[TIMING_RECORD] ", 1)[1]))
            counts = Counter(row["source"] for row in records)
            report["records"] = dict(counts)
            if args.mode == "disabled":
                assert not records, counts
            else:
                assert counts["Env step"] == counts["Rollout step"] == 6, counts
                assert counts["Learner update"] == 16, counts
                for row in records:
                    if row["source"] in ("Rollout step", "Env step"):
                        majors = (
                            ["input", "execute", "publish"]
                            if row["source"] == "Env step"
                            else ["input", "inference", "action_send", "data_submit"]
                        )
                        assert (
                            abs(row["work_ms"] - sum(row[k + "_ms"] for k in majors))
                            < 1e-6
                        )
                        detail = (
                            "prefix_ms"
                            if row["source"] == "Rollout step"
                            else "action_prepare_ms"
                        )
                        assert (detail in row) == (args.mode == "detailed")
                assert counts["UR action"] == (18 if args.mode == "detailed" else 0), (
                    counts
                )
                assert counts["Rollout terminal"] == (
                    2 if args.mode == "detailed" else 0
                ), counts
            Path(args.output + ".records.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            report["status"] = "passed"
    finally:
        try:
            if env is not None:
                env.stop().wait()
            if actor is not None:
                actor.stop().wait()
            if rollout is not None:
                rollout.finish_probe().wait()
        finally:
            Path(args.output + ".json").write_text(json.dumps(report, indent=2) + "\n")
            ray.shutdown()


if __name__ == "__main__":
    main()
