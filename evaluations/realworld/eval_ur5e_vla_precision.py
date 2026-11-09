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

"""Run the copied VLA-Precision UR backend with an RLinf checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> None:
    """Validate or run the standalone single-process comparison deployment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("vla_precision_ur.yaml")
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--check",
        action="store_true",
        help="Check config, files and dependencies without connecting hardware",
    )
    checks.add_argument(
        "--check-model",
        action="store_true",
        help="Check actual checkpoint inference on GPU without connecting hardware",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["policy"].get("cuda_visible_devices", "0")
    )
    from evaluations.realworld.vla_precision_ur.policy import (
        check_model,
        load_norm_stats,
        model_config_from_yaml,
    )
    from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
    from rlinf.utils.logging import get_logger

    model_cfg = model_config_from_yaml(config)
    if resolve_full_weights(model_cfg.model_path) is None:
        raise FileNotFoundError(
            f"No full_weights.pt found under {model_cfg.model_path}"
        )
    stats = load_norm_stats(config["model"]["norm_stats_path"])
    from evaluations.realworld.vla_precision_ur.utils import (
        validate_single_arm_norm_stats,
    )

    validate_single_arm_norm_stats(
        stats,
        fixed_gripper=False,
        learned_state_dim=19,
        path=config["model"]["norm_stats_path"],
        padded_action_dim=int(model_cfg.openpi.model_action_dim),
    )
    if args.check_model:
        check_model(config)
        return
    from evaluations.realworld.vla_precision_ur.ur import Inference

    if args.check:
        get_logger().info(
            "VLA-Precision UR config, checkpoint paths, norm stats and hardware imports passed; no hardware connected"
        )
        return
    engine = Inference(args.config)
    engine.run()


if __name__ == "__main__":
    main()
