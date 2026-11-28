# SPDX-License-Identifier: Apache-2.0
"""Explicit RGB uint8 grading: linear exposure, S curve, channel balance and saturation."""

import numpy as np


def grade(rgb, params=None):
    p = params or {}
    if not isinstance(p, dict):
        raise ValueError('grade must be a parameter object, not a scene name')
    x = np.arange(256) / 255.0
    y = np.clip((np.clip(x**2.2 * p.get('exposure', 1.0), 0, None)) ** (1 / 2.2), 0, 1)
    lut = np.clip(y + p.get('curve', 0.10) * np.sin(2 * np.pi * (y - 0.5)) * y * (1 - y) * 2, 0, 1)
    y = lut[rgb].astype(np.float32)
    y *= np.asarray(p.get('cool', [0.985, 1.0, 1.03]))
    grey = y.mean(-1, keepdims=True)
    y = grey + (y - grey) * p.get('saturation', 1.06)
    return (np.clip(y, 0, 1) * 255).round().astype(np.uint8)
