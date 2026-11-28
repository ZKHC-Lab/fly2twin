# SPDX-License-Identifier: Apache-2.0
"""Per-training-camera right-composed translation and continuous 6D rotation."""

import torch
import torch.nn.functional as F


class CameraResidual(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(count, 9))

    def forward(self, base, indices):
        values = self.values[indices]
        v = values[..., 3:] + values.new_tensor([1, 0, 0, 0, 1, 0])
        u = F.normalize(v[..., :3], dim=-1)
        w = F.normalize(v[..., 3:] - (u * v[..., 3:]).sum(-1, keepdim=True) * u, dim=-1)
        rotation = torch.stack((u, w, torch.cross(u, w, dim=-1)), dim=-2)
        delta = torch.eye(4, device=values.device, dtype=values.dtype).expand(*values.shape[:-1], 4, 4).clone()
        delta[..., :3, :3] = rotation
        delta[..., :3, 3] = values[..., :3]
        return base @ delta
