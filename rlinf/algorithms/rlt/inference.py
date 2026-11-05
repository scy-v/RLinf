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

"""Shared Stage2 inference for UR evaluation without learner state or replay."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.rlt.route import RealworldRLTRoute, RLTRouteContext
from rlinf.utils.logging import get_logger


def is_stage2(config: Any) -> bool:
    """Validate and read the explicit inference mode."""
    stage = config.get("inference_stage", "stage1")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Expected inference_stage stage1 or stage2, got {stage!r}.")
    return stage == "stage2"


def stage2_configs(options: Any) -> tuple[DictConfig, DictConfig]:
    """Resolve the paired frozen feature model and MLP from their training config."""
    training = OmegaConf.load(Path(options["training_config"]).expanduser())
    feature = OmegaConf.create(
        OmegaConf.to_container(training.rollout.rlt_feature_model, resolve=True)
    )
    head = OmegaConf.create(OmegaConf.to_container(training.actor.model, resolve=True))
    if (
        head.model_type != "rlt_mlp_policy"
        or head.proprio_dim != 19
        or head.action_dim != 7
    ):
        raise ValueError("Expected a UR RLT MLP with 19 state and 7 action dimensions.")
    steps = int(options.get("num_steps", feature.openpi.num_steps))
    if steps < 1:
        raise ValueError(f"Expected positive Stage2 num_steps, got {steps}.")
    feature.num_steps = feature.openpi.num_steps = steps
    feature.openpi.rtc_enabled = False
    checkpoint = Path(options["checkpoint_path"]).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Expected Stage2 full_weights.pt file at {checkpoint}."
        )
    from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights

    if resolve_full_weights(feature.model_path) is None:
        raise FileNotFoundError(f"Missing paired Stage1 weights: {feature.model_path}")
    if not Path(feature.openpi_data.norm_stats_path).is_file():
        raise FileNotFoundError(feature.openpi_data.norm_stats_path)
    if head.num_action_chunks > feature.openpi.action_horizon:
        raise ValueError(
            "Stage2 action chunk exceeds the frozen VLA prediction horizon."
        )
    return feature, head


class Stage2Inference:
    """Evaluate a frozen feature model and deterministic MLP with explicit routing."""

    def __init__(self, feature_model: Any, head: torch.nn.Module):
        self.feature_model = feature_model
        self.head = head.eval().requires_grad_(False)
        self.route = RealworldRLTRoute(use_schedule=False)

    @classmethod
    def load(cls, feature_model: Any, options: Any) -> "Stage2Inference":
        """Load only the exported MLP weights, never optimizer or replay state."""
        from rlinf.models.embodiment.mlp_policy import get_model

        feature_cfg, head_cfg = stage2_configs(options)
        head = get_model(head_cfg, torch_dtype=torch.float32)
        weights = torch.load(
            Path(options["checkpoint_path"]).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        head.load_state_dict(weights, strict=True)
        head.to(device=feature_model.device, dtype=torch.float32)
        head.eval().requires_grad_(False)
        feature_model.eval().requires_grad_(False)
        mlp_compile_mode = options.get("mlp_compile_mode", "max-autotune-no-cudagraphs")
        if mlp_compile_mode is not None:
            head.enable_torch_compile(mode=mlp_compile_mode)
        compile_mode = options.get("compile_mode", "max-autotune")
        if compile_mode is not None:
            feature_model.enable_rlt_torch_compile(mode=compile_mode)
        get_logger().info(
            "Stage2 inference: frozen=%s; MLP=%s; denoise=%s; compile=%s; "
            "actor=deterministic; reset selects VLA, double '.' selects MLP next chunk.",
            feature_cfg.model_path,
            options["checkpoint_path"],
            feature_cfg.num_steps,
            compile_mode,
        )
        return cls(feature_model, head)

    @torch.inference_mode()
    def predict(self, env_obs: dict, *, noise: torch.Tensor | None = None) -> tuple:
        """Use one observation for features, physical state and reference actions."""
        features = self.feature_model.extract_rlt_obs(env_obs, noise=noise)
        actions, result = self.head.predict_action_batch(features, mode="eval")
        routed = self.route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=features,
                student_actions=actions,
                result=result,
                mode="eval",
                rlt_switch_flags=env_obs.get("rlt_switch_flags"),
            )
        )
        if not torch.isfinite(routed.actions).all():
            raise ValueError("Stage2 inference produced non-finite actions.")
        return routed.actions, routed.result


class CriticalPhaseSwitch:
    """Latch actor control after two dot presses; reset clears both presses and mode."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self._last_press: float | None = None

    def clear_pending_press(self) -> None:
        """Forget an incomplete gesture while retaining the selected policy."""
        self._last_press = None

    def press(self, timestamp: float) -> bool:
        """Return True only when a double press first enables actor control."""
        previous = self._last_press
        self._last_press = timestamp
        if (
            not self.active
            and previous is not None
            and 0 <= timestamp - previous <= 0.5
        ):
            self.active = True
            return True
        return False
