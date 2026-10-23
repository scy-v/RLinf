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

"""UR evaluation termination without changing generic sync or RTC workers."""

from collections import defaultdict
from typing import Any

import torch

from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.env.rtc_env_worker import RTCEnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def evaluation_complete(env: Any) -> bool:
    """Read the UR episode counter from the single real-world environment."""
    return bool(env.unwrapped.env.envs[0].get_wrapper_attr("evaluation_complete"))


class URSyncEnvWorker(EnvWorker):
    """Run the normal synchronous policy exchange until the UR episode budget ends."""

    def evaluate(self, input_channel: Channel, rollout_channel: Channel) -> dict:
        if (
            self.stage_num != 1
            or self.cfg.env.eval.total_num_envs != 1
            or self.env_decoupled_mode
        ):
            raise ValueError(
                "UR evaluation requires one environment and one coupled pipeline stage."
            )
        metrics = defaultdict(list)
        env = self.eval_env_list[0]
        try:
            env.is_start = True
            self.eval_prev_done[0] = torch.zeros(
                self.eval_num_envs_per_stage, dtype=torch.bool
            )
            obs, infos = env.reset()
            output = EnvOutput(obs=obs, env_infos=infos)
            for _ in range(self.n_eval_chunk_steps):
                self._send_ur_observation(rollout_channel, output)
                policy_output = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="eval_rollout_results",
                    route_key=0,
                    batch_size=self.eval_batch_size,
                )
                actions = (
                    policy_output.actions
                    if hasattr(policy_output, "actions")
                    else policy_output
                )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                output, info = self.env_evaluate_step(actions, 0)
                for key, value in info.items():
                    metrics[key].append(value)
                if evaluation_complete(env):
                    break
            self._send_ur_observation(rollout_channel, output, stop=True)
            self.finish_rollout(mode="eval")
        finally:
            env.close()
        return {
            key: torch.cat(value, dim=0).contiguous().cpu()
            for key, value in metrics.items()
        }

    def _send_ur_observation(
        self, channel: Channel, output: EnvOutput, stop: bool = False
    ) -> None:
        data = self._build_rollout_input_data(output.to_dict())
        data["ur_eval_stop"] = torch.tensor([stop])
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=channel,
            data=data,
            split_fn=self._obs_split_fn,
            mode="eval",
            tag="rollout_results",
            route_key=0,
        )


class URSyncRolloutWorker(MultiStepRolloutWorker):
    """Use the existing model predictor with an explicit UR stop message."""

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        merged = super()._merge_obs_batches(obs_batches)
        if obs_batches:
            merged["ur_eval_stop"] = torch.cat(
                [batch["ur_eval_stop"] for batch in obs_batches], dim=0
            )
        return merged

    async def evaluate(self, input_channel: Channel, output_channel: Channel) -> None:
        if self.enable_offload:
            self.reload_model()
        try:
            while True:
                output = await self.recv_from(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="eval_rollout_results",
                    route_key=0,
                    async_op=True,
                    batch_size=self.eval_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                ).async_wait()
                if bool(output["ur_eval_stop"].any()):
                    break
                actions, _ = self._predict_rollout_actions(
                    output["obs"],
                    mode="eval",
                    final_obs=output.get("final_obs"),
                )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                self.send_to(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=actions,
                    tag="eval_rollout_results",
                    route_key=0,
                    async_op=True,
                    batch_size=self.eval_batch_size,
                )
        finally:
            if self.enable_offload:
                self.offload_model()


class URRTCEnvWorker(RTCEnvWorker):
    """Retain RTC replanning and stop after the final UR outcome."""

    def init_worker(self) -> None:
        super().init_worker()
        for env in self.eval_env_list:
            env.manual_episode_control_only = env.unwrapped.manual_episode_control_only

    def _evaluate_rtc_action(self, env_action: Any, stage_id: int) -> tuple:
        output, info = super()._evaluate_rtc_action(env_action, stage_id)
        env = self.eval_env_list[stage_id]
        if evaluation_complete(env):
            # The inherited RTC loop drains pending requests and sends its normal stop.
            env.manual_episode_control_only = False
        return output, info

    def evaluate(self, input_channel: Channel, rollout_channel: Channel) -> dict:
        try:
            return super().evaluate(input_channel, rollout_channel)
        finally:
            for env in self.eval_env_list:
                env.close()
