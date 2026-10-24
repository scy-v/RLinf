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
"""Compose observation selection with an existing robot data configuration."""

import dataclasses
import json
import pathlib

import openpi.models.model as model
from openpi.training.config import DataConfig, DataConfigFactory

from rlinf.models.embodiment.openpi.transforms.state import SelectState


@dataclasses.dataclass(frozen=True, kw_only=True)
class StateSelectionDataConfig(DataConfigFactory):
    """Apply shared state selection before a robot's input transforms."""

    data_factory: DataConfigFactory
    obs_indices: tuple[int, ...]

    def create(
        self, assets_dirs: pathlib.Path, model_config: model.BaseModelConfig
    ) -> DataConfig:
        selector = SelectState(self.obs_indices)
        if len(selector.indices) > model_config.action_dim:
            raise ValueError(
                f"Expected at most {model_config.action_dim} obs_indices, got {len(selector.indices)}"
            )
        factory = dataclasses.replace(
            self.data_factory,
            repo_id=self.repo_id,
            assets=self.assets,
            base_config=self.base_config,
        )
        data = factory.create(assets_dirs, model_config)
        metadata_path = (
            pathlib.Path(self.assets.assets_dir or assets_dirs)
            / (data.asset_id or "")
            / "norm_metadata.json"
        )
        if metadata_path.is_file():
            saved_indices = json.loads(metadata_path.read_text())["obs_indices"]
            if saved_indices != list(selector.indices):
                raise ValueError(
                    f"Expected norm stats obs_indices={list(selector.indices)}, got {saved_indices}; recompute stats"
                )
        if data.norm_stats is not None:
            state_dim = len(data.norm_stats["state"].mean)
            if state_dim not in (len(selector.indices), model_config.action_dim):
                raise ValueError(
                    f"Expected norm stats with {len(selector.indices)} selected or "
                    f"{model_config.action_dim} padded state dimensions, got {state_dim}; recompute stats"
                )
        return dataclasses.replace(
            data,
            data_transforms=dataclasses.replace(
                data.data_transforms, inputs=(selector, *data.data_transforms.inputs)
            ),
        )
