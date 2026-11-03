# SPDX-License-Identifier: Apache-2.0
"""Local camera delta: translation plus continuous 6D rotation (Zhou et al., 2019).
Matches the gsplat example's right-composed camera parameterization.
"""
import torch
import torch.nn.functional as F


class CameraDelta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding construction preserves the reference RNG draw, then zero initialization.
        self.embedding = torch.nn.Embedding(1, 9)
        torch.nn.init.zeros_(self.embedding.weight)

    def forward(self, base):
        values = self.embedding.weight
        canonical = values.new_tensor([1,0,0,0,1,0])
        v = values[:,3:] + canonical
        u = F.normalize(v[:,:3], dim=-1)
        w = F.normalize(v[:,3:]-(u*v[:,3:]).sum(-1,keepdim=True)*u, dim=-1)
        rotation = torch.stack((u,w,torch.cross(u,w,dim=-1)),dim=-2)
        delta = torch.eye(4,device=values.device).repeat(1,1,1)
        delta[:,:3,:3] = rotation
        delta[:,:3,3] = values[:,:3]
        return base @ delta
