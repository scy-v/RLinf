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

# Ported from VLA-Precision; see provenance.json for source and changes.

"""Image conversion shared by offline preprocessing and robot observations.

The resize behavior matches OpenPI's fixed client implementation: preserve the
aspect ratio, resize with PIL bilinear interpolation, and zero-pad centrally.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert floating-point images in [0, 1] to compact uint8 images."""
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    method: int = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Resize images without distortion and zero-pad them to the target size."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flattened = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(image), height, width, method)
            for image in flattened
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image,
    height: int,
    width: int,
    method: int,
) -> Image.Image:
    current_width, current_height = image.size
    if current_width == width and current_height == height:
        return image

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = image.resize((resized_width, resized_height), resample=method)

    padded = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    padded.paste(resized, (pad_width, pad_height))
    return padded
