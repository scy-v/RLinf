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

"""UR-only evaluation entry retaining RLinf model sampling and RTC scheduling."""

from contextlib import ExitStack

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.inference import is_stage2, stage2_configs
from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.keyboard_focus import configure_terminal_keyboard
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.robot_lifecycle import termination_signals
from rlinf.workers.env.ur5e_eval_worker import (
    URRTCEnvWorker,
    URRTCRolloutWorker,
    URSyncEnvWorker,
    URSyncRolloutWorker,
)


def prepare_inference_config(cfg):
    """Select a Stage2 profile without changing the Stage1 configuration."""
    if is_stage2(cfg):
        if cfg.runner.rtc.enabled or cfg.runner.get("ckpt_path"):
            raise ValueError(
                "Stage2 uses synchronous inference and stage2.checkpoint_path; disable RTC and runner.ckpt_path."
            )
        feature, head = stage2_configs(cfg.stage2)
        cfg = OmegaConf.merge(
            cfg,
            {
                "env": {
                    "eval": {
                        "ur_evaluation": {"stage2_enabled": True},
                        "override_cfg": {
                            "gripper_close_threshold": cfg.stage2.gripper_close_threshold
                        },
                    }
                },
            },
        )
        # The worker exposes the MLP execution length while the feature model retains its horizon.
        feature.num_action_chunks = head.num_action_chunks
        cfg.rollout.model = feature
    return cfg


@hydra.main(
    version_base="1.1", config_path=".", config_name="realworld_ur5e_eval_pi05_sft"
)
def main(cfg) -> None:
    """Launch a single UR environment and its sync or RTC model worker."""
    mp.set_start_method("spawn", force=True)
    cfg = prepare_inference_config(cfg)
    cfg = validate_cfg(cfg)
    configure_terminal_keyboard(cfg.get("keyboard_terminal_only", False))
    if cfg.rollout.pipeline_stage_num != 1 or cfg.env.eval.total_num_envs != 1:
        raise ValueError(
            "UR evaluation requires one environment and one pipeline stage."
        )
    if not cfg.env.eval.ur_evaluation.enabled or cfg.env.eval.rollout_epoch != 1:
        raise ValueError(
            "UR evaluation requires its episode controller and rollout_epoch=1."
        )
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rtc = cfg.runner.rtc.enabled
    env_cls = URRTCEnvWorker if rtc else URSyncEnvWorker
    rollout_cls = URRTCRolloutWorker if rtc else URSyncRolloutWorker
    with termination_signals(), ExitStack() as cleanup:
        rollout = rollout_cls.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env = env_cls.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        cleanup.callback(lambda: env.close().wait())
        runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
        runner.init_workers()
        runner.run()


if __name__ == "__main__":
    main()
