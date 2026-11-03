# SPDX-License-Identifier: Apache-2.0
# Callback ordering adapted from gsplat DefaultStrategy (Apache-2.0).
# Copyright 2023-2026 the Regents of the University of California,
# Nerfstudio Team and contributors. See licenses/gsplat-APACHE-2.0.txt.
"""Explicit reference scheduling around gsplat's numerical primitives.

Public 1.5.2 resets at zero; public 1.5.3 has a chained bitwise comparison that
never resets. The reference uses positive multiples of reset_every, before
refine_stop_iter. Keep the entire callback order explicit rather than delegate
to either version's conditional.
"""
from dataclasses import dataclass
import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa


@dataclass
class BaselineStrategy(DefaultStrategy):
    reset_at_zero: bool = False

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step >= self.refine_stop_iter:
            return
        self._update_state(params, state, info, packed=packed)
        refine = (step > self.refine_start_iter and step % self.refine_every == 0
                  and step % self.reset_every >= self.pause_refine_after_reset)
        if refine:
            duplicated, split = self._grow_gs(params, optimizers, state, step)
            pruned = self._prune_gs(params, optimizers, state, step)
            for key in ('grad2d', 'count'):
                state[key].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state['radii'].zero_()
            torch.cuda.empty_cache()
            if self.verbose:
                print(f'Step {step}: duplicate={duplicated}, split={split}, prune={pruned}', flush=True)
        if (step > 0 or self.reset_at_zero) and step % self.reset_every == 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=2*self.prune_opa)
