"""NUCLEUS-2 MoE on the staggered grid, data-only, with a solenoidal velocity input.

Identical to :mod:`nucleus2_moe_staggered` (same backbone, head, training module and
loss) except for one change at the input: the face velocity of every input frame is
replaced by its divergence-free part (Leray projection, Neumann walls, Dirichlet top
outflow) before the network sees it. The output is unchanged: the full face velocity,
including its expansion part, is still predicted and still in the loss.

Why. In the solver the velocity splits as u = u_sol + grad(chi) with laplacian(chi) =
div(u), and div(u) is set by the interfacial mass flux. grad(chi) is rebuilt from mdot at
every step and carries no memory: given temperature and the interface, the expansion part
of the input velocity holds no information about the future. A model that sees it can
copy the past volume source instead of computing it from the temperature field. With
the solenoidal input the volume source has to come from the other inputs.

The projection is done in physical units, because the standard normalizer scales the two
velocity components by different factors and the projection does not commute with that.
The velocity normalization constants are therefore part of the model config (set them
from the normalizer config, see ``config/model_cfg/nucleus2/nucleus2_staggered_solin.yaml``)
and are saved with the checkpoint.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch
from nucleus.physics.leray import leray_projection
from nucleus.physics.poisson import GRID_SPACING
from nucleus.models.nucleus2_moe_staggered import (
    Nucleus2MoEStaggeredBase,
    Nucleus2MoEStaggeredConfig,
    Nucleus2MoEStaggeredOutput,
)
from nucleus.models.nucleus2_moe_staggered_backbone import StaggeredFields, config_from_dict

from ._api import register_model

__all__ = ["Nucleus2MoEStaggeredSolin", "Nucleus2MoEStaggeredSolinBase", "Nucleus2MoEStaggeredSolinConfig"]


@dataclass
class Nucleus2MoEStaggeredSolinConfig(Nucleus2MoEStaggeredConfig):
    # Velocity normalization (physical = normalized * std + mean), from the normalizer config.
    velx_mean: float = 0.0
    velx_std: float = 1.0
    vely_mean: float = 0.0
    vely_std: float = 1.0
    # Physical grid spacing.
    dx: float = GRID_SPACING
    dy: float = GRID_SPACING


def _config_from_dict(d: dict) -> Nucleus2MoEStaggeredSolinConfig:
    return config_from_dict(Nucleus2MoEStaggeredSolinConfig, d)


class Nucleus2MoEStaggeredSolinBase(Nucleus2MoEStaggeredBase):
    """Uncompiled model (use for hooks); ``nucleus2_moe_staggered_solin`` is this class
    compiled."""

    config_class = Nucleus2MoEStaggeredSolinConfig

    def project_input(self, input) -> StaggeredFields:
        """Input fields with the face velocity replaced by its divergence-free part.
        Velocities are normalized on the way in and on the way out."""
        c = self.config
        vx = input.velx.float() * c.velx_std + c.velx_mean
        vy = input.vely.float() * c.vely_std + c.vely_mean
        sx, sy = leray_projection(vx, vy, c.dx, c.dy)
        return StaggeredFields(
            sdf=input.sdf,
            temperature=input.temperature,
            velx=((sx - c.velx_mean) / c.velx_std).to(input.velx.dtype),
            vely=((sy - c.vely_mean) / c.vely_std).to(input.vely.dtype),
        )

    def step(self, input, sim_params: Optional[torch.Tensor] = None, normalizer=None,
             **unused) -> Nucleus2MoEStaggeredOutput:
        if isinstance(input, DivFreeBatch):
            sim_params = input.sim_params_tensor
            input = input.input
        return super().step(self.project_input(input), sim_params)


@register_model("nucleus2_moe_staggered_solin")
@torch.compile(fullgraph=True)
class Nucleus2MoEStaggeredSolin(Nucleus2MoEStaggeredSolinBase):
    pass
