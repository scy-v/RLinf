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

import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.ur5e_eval_worker import (
    URRTCEnvWorker,
    URSyncEnvWorker,
    URSyncRolloutWorker,
)
from rlinf.workers.rollout.hf.rtc_huggingface_worker import RTCMultiStepRolloutWorker


@hydra.main(
    version_base="1.1", config_path=".", config_name="realworld_ur5e_eval_pi05_sft"
)
def main(cfg) -> None:
    """Launch a single UR environment and its sync or RTC model worker."""
    mp.set_start_method("spawn", force=True)
    cfg = validate_cfg(cfg)
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
    rollout_cls = RTCMultiStepRolloutWorker if rtc else URSyncRolloutWorker
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
    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
