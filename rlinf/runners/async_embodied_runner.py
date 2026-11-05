# Copyright 2025 The RLinf Authors.
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

import asyncio
import time
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.runners.async_weight_sync_mixin import AsyncWeightSyncMixin
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress, crossed_interval

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
        AsyncEmbodiedDAGGERFSDPPolicy,
    )
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


from rlinf.utils.progress import (
    ElapsedTime,
    StageTiming,
    TrainingProgress,
    timing_active,
    timing_detailed,
)


class AsyncEmbodiedRunner(AsyncWeightSyncMixin, EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union["AsyncEmbodiedSACFSDPPolicy", "AsyncEmbodiedDAGGERFSDPPolicy"],
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        reward: "EmbodiedRewardWorker",
        critic=None,
    ):
        super().__init__(cfg, actor, rollout, env, reward, critic)
        self._step_timing = StageTiming(
            "Runner step",
            timing_detailed(cfg),
            self.logger.info,
        )

        # Data channels
        self.env_metric_channel = Channel.create("EnvMetric")
        self.rollout_metric_channel = Channel.create("RolloutMetric")
        self.replay_ack_channel = (
            Channel.create("EpisodeReplayAck")
            if cfg.env.train.get("flush_replay_on_episode_end", False)
            else None
        )

        self.init_weight_sync_state()

    def get_env_metrics(self) -> tuple[dict, list[dict], list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.env_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, [], []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results, metric_field="time"
        )
        env_metrics, ranked_env_metrics_list = self._process_ranked_eval_results(
            results, metric_field="env"
        )
        if not env_metrics:
            return {**time_metrics}, ranked_time_metrics_list, ranked_env_metrics_list

        return (
            {**env_metrics, **time_metrics},
            ranked_time_metrics_list,
            ranked_env_metrics_list,
        )

    def get_rollout_metrics(self) -> tuple[dict, list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.rollout_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results, metric_field="time"
        )
        return time_metrics, ranked_time_metrics_list

    def evaluate(self):
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if env_decoupled_mode:
            # The rollout-side loop is unbounded by design. Ensure one service
            # exists and reuse it across every validation cycle.
            self.rollout.ensure_evaluate_service(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            ).wait()
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        if not env_decoupled_mode:
            rollout_handle: Handle = self.rollout.evaluate(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
        env_results = env_handle.wait()
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _advance_training_step(self, metrics: dict[str, float]) -> bool:
        """Count RLT rounds from cumulative critic updates, including partial rounds."""
        if not metrics:
            return False
        if self.cfg.algorithm.get("rlt_schedule", {}).get("enable", False):
            if metrics["rlt/critic_updates_run"] <= 0:
                return False
            self.global_step = int(metrics["rlt/update_step"]) // int(
                self.cfg.algorithm.update_epoch
            )
        else:
            self.global_step += 1
        return True

    def run(self):
        use_rlt_schedule = self.cfg.algorithm.get("rlt_schedule", {}).get(
            "enable", False
        )
        if use_rlt_schedule:
            update_epoch = int(self.cfg.algorithm.update_epoch)
            if update_epoch <= 0:
                raise ValueError(f"Expected positive update_epoch, got {update_epoch}.")
            updates = self.actor.get_rollout_sync_version().wait()
            if len(set(updates)) != 1:
                raise ValueError(f"Expected matching RLT update counts, got {updates}.")
            # Learner state preserves partial rounds and supersedes legacy folder counts.
            self.global_step = int(updates[0]) // update_epoch
            self.logger.info(
                f"RLT progress: {updates[0]} critic updates / {update_epoch} "
                f"updates per round = {self.global_step} completed rounds."
            )
        start_step = self.global_step
        start_time = time.time()
        self.env.set_global_step(self.global_step).wait()
        # Rollout must start from the actor's exact initial/resumed weights, so
        # this first sync always blocks even when overlap is enabled.
        with ElapsedTime(
            "LOAD",
            "Initial actor-to-rollout weight synchronization",
            log=self.logger.info,
            enabled=timing_detailed(self.cfg),
        ):
            self.update_rollout_weights()

        replay_options = {}
        if getattr(self, "replay_ack_channel", None) is not None:
            replay_options["replay_ack_channel"] = self.replay_ack_channel
        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            metric_channel=self.env_metric_channel,
            **replay_options,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            actor_channel=self.actor_channel,
            metric_channel=self.rollout_metric_channel,
        )
        if self.reward is not None:
            reward_handle: Handle = self.reward.compute_rewards_async(
                input_channel=self.reward_channel,
                output_channel=self.env_channel,
            )
        actor_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel, **replay_options
        )

        online_progress = None
        while self.global_step < self.max_steps:
            # Use the step we're ABOUT to run as the profiling key, mirroring
            # ``EmbodiedRunner.run`` which gates before ``self.global_step += 1``.
            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)
            skip_step = False
            with self.timer("step"):
                actor_training_handle: Handle = self.actor.run_training()
                actor_result = actor_training_handle.wait()
                metrics = self._aggregate_numeric_metrics(actor_result)
                previous_step = self.global_step
                skip_step = not self._advance_training_step(metrics)

                if not skip_step:
                    warmup = self.cfg.algorithm.get("rlt_schedule", {}).get(
                        "warmup_post_collect_updates", 0
                    )
                    update_step = metrics.get("rlt/update_step", 0)
                    warmup_completed = use_rlt_schedule and (
                        update_step - metrics["rlt/critic_updates_run"]
                        < warmup
                        <= update_step
                    )
                    if warmup_completed or crossed_interval(
                        previous_step, self.global_step, self.weight_sync_interval
                    ):
                        self.update_rollout_weights(no_wait=self.sync_weight_no_wait)

                    training_metrics = {f"train/{k}": v for k, v in metrics.items()}

                    if (
                        self.cfg.algorithm.get("rlt_schedule", {}).get("enable", False)
                        and update_step >= warmup
                    ):
                        buffer_size = int(
                            training_metrics.get("train/rlt/replay_size", 0)
                        )
                        if online_progress is None:
                            online_progress = TrainingProgress(
                                "Online RL runner rounds",
                                self.max_steps,
                                self.logger.info,
                                initial=self.global_step,
                                enabled=timing_active(self.cfg),
                            )
                        online_progress.update(self.global_step, buffer_size)

                    run_val, save_model, _ = check_progress(
                        self.global_step,
                        self.max_steps,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=False,
                        previous_step=previous_step,
                    )
                    if save_model:
                        self._save_checkpoint()
                    eval_metrics = {}
                    if run_val:
                        with self.timer("eval"):
                            eval_metrics = self.evaluate()
                            eval_metrics = {
                                f"eval/{k}": v for k, v in eval_metrics.items()
                            }

            if skip_step:
                if metrics:
                    self.metric_logger.log(
                        {f"train/{k}": v for k, v in metrics.items()}, self.global_step
                    )
                self.timer.consume_durations()
                if profiled_step is not None:
                    self._close_profiling_window(profiled_step)
                time.sleep(1.0)
                continue

            time_metrics = self.timer.consume_durations()
            self._step_timing.record(
                time_metrics["step"],
                context={
                    "step": self.global_step,
                    "critic_updates": training_metrics.get(
                        "train/rlt/critic_updates_run"
                    ),
                    "actor_updates": training_metrics.get(
                        "train/rlt/actor_updates_run"
                    ),
                },
            )
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            if self.actor_channel is not None:
                # Total outstanding across consumers, matching the pre-dispatcher
                # meaning of this series.
                training_metrics["train/replay_channel_qsize"] = (
                    self.actor_channel.qsize()
                )
            actor_training_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            actor_training_time_metrics = {
                f"time/actor/{k}": v for k, v in actor_training_time_metrics.items()
            }
            time_metrics.update(actor_training_time_metrics)
            env_metrics, env_time_metrics_per_rank, env_metrics_per_rank = (
                self.get_env_metrics()
            )
            rollout_metrics, rollout_time_metrics_per_rank = self.get_rollout_metrics()

            self.metric_logger.log(time_metrics, self.global_step)
            self.metric_logger.log(env_metrics, self.global_step)
            self.metric_logger.log(rollout_metrics, self.global_step)
            self.metric_logger.log(training_metrics, self.global_step)
            self.metric_logger.log(eval_metrics, self.global_step)
            self._log_ranked_metrics(
                metrics_list=actor_result,
                step=self.global_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
                add_prefix=False,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=self.global_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
                add_prefix=False,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
                add_prefix=False,
            )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                self.global_step - 1,
                self.max_steps,
                start_time,
                logging_metrics,
                start_step,
            )

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

        # Let any in-flight non-blocking sync land before the workers go away.
        self.drain_pending_rollout_weight_sync()

        self.env.stop().wait()
        self.rollout.stop().wait()
        self.actor.stop().wait()
        if self.reward is not None:
            self.reward.stop().wait()
            reward_handle.wait()
        env_handle.wait()
        rollout_handle.wait()
        actor_handle.wait()
