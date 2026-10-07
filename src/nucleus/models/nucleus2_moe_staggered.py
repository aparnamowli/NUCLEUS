"""NUCLEUS-2 MoE on the staggered (MAC) grid, data-only.

Baseline for the physics-constrained divergence-free model (``nucleus2_moe_divfree``):
the same staggered input, backbone, and training objective on the data fields, with
no physics in the model or the loss -- no temperature ansatz, no Leray/Poisson split,
no divergence-source head, no Helmholtz targets.

Velocity is predicted on the faces. The head outputs, per cell, the velocity on its
four faces (left, right, bottom, top), the same per-cell split the input uses
(:func:`divfree_input_to_cells`). Every interior face is predicted twice, once by each
of its two cells; the face value is the average of the two. Wall faces have a single
owner. The divergence of the output is then the exact discrete MAC divergence of the
face field the network produced.
"""
import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding

from nucleus.layers import LinearEmbed, LinearDebed, AdaptiveEmbed, AdaptiveDebed
from nucleus.layers.moe.topk_moe import TopkMoEOutput
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.trajectory import Trajectory
from nucleus.utils.inf_stabilizer import clip_temp_by_phase
from nucleus.models.nucleus2_moe_divfree import (
    GRADIENT_SPACING,
    NUCLEUS_DTYPE,
    DivFreeForecastModule,
    Nucleus2MoEDivFreeInput,
    TransformerMoEBlock,
    divfree_input_to_cells,
    get_dtype,
    _DTYPE_TO_STR,
    _STR_TO_DTYPE,
)

from ._api import register_model

__all__ = [
    "Nucleus2MoEStaggered",
    "Nucleus2MoEStaggeredBase",
    "Nucleus2MoEStaggeredConfig",
    "Nucleus2MoEStaggeredOutput",
    "StaggeredForecastModule",
    "faces_from_cell_split",
]

_DTYPE_FIELDS = ("embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype")


@dataclass
class Nucleus2MoEStaggeredConfig:
    patch_size: int
    embed_dim: int
    num_heads: int
    processor_blocks: int
    num_experts: int
    topk: int
    moe_intermediate_dim: int
    patching: str  # "Linear" or "Adaptive"
    embed_dtype: NUCLEUS_DTYPE = "float32"
    debed_dtype: NUCLEUS_DTYPE = "float32"
    activation_dtype: NUCLEUS_DTYPE = "float32"
    attention_dtype: NUCLEUS_DTYPE = "bfloat16"
    moe_dtype: NUCLEUS_DTYPE = "bfloat16"


def _config_to_dict(config: Nucleus2MoEStaggeredConfig) -> dict:
    d = dataclasses.asdict(config)
    for key in _DTYPE_FIELDS:
        if not isinstance(d[key], str):
            d[key] = _DTYPE_TO_STR[d[key]]
    return d


def _config_from_dict(d: dict) -> Nucleus2MoEStaggeredConfig:
    d = dict(d)
    for key in _DTYPE_FIELDS:
        if not isinstance(d[key], torch.dtype):
            d[key] = _STR_TO_DTYPE[d[key]]
    return Nucleus2MoEStaggeredConfig(**d)


def faces_from_cell_split(
    vel_left: torch.Tensor,
    vel_right: torch.Tensor,
    vel_bottom: torch.Tensor,
    vel_top: torch.Tensor,
):
    """Inverse of the per-cell face split in :func:`divfree_input_to_cells`.

    Each interior face appears twice, as the right/top face of one cell and the
    left/bottom face of its neighbor; its value is the average of the two. Wall faces
    have one owner and take its value.

    Args (leading dims written ``...``): four ``(..., H, W)`` tensors.

    Returns:
        ``velx`` ``(..., H, W+1)`` and ``vely`` ``(..., H+1, W)``.
    """
    velx = torch.cat(
        (
            vel_left[..., :, :1],
            0.5 * (vel_right[..., :, :-1] + vel_left[..., :, 1:]),
            vel_right[..., :, -1:],
        ),
        dim=-1,
    )
    vely = torch.cat(
        (
            vel_bottom[..., :1, :],
            0.5 * (vel_top[..., :-1, :] + vel_bottom[..., 1:, :]),
            vel_top[..., -1:, :],
        ),
        dim=-2,
    )
    return velx, vely


@dataclass
class Nucleus2MoEStaggeredOutput:
    """Model output, each field on its natural grid and in normalized units.

    Shapes: ``sdf``, ``temperature`` ``(B, T, H, W)``; ``velx`` ``(B, T, H, W+1)``;
    ``vely`` ``(B, T, H+1, W)``.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    moe_outputs: List[TopkMoEOutput]

    def to_cell_tensor(self) -> torch.Tensor:
        """The same six cell channels the model consumes (for logging/plotting)."""
        return divfree_input_to_cells(self.sdf, self.temperature, self.velx, self.vely)


class Nucleus2MoEStaggeredBase(nn.Module):
    """Uncompiled model. Use this class directly when forward hooks are needed
    (``torch.compile(fullgraph=True)`` does not run them); the registered
    ``nucleus2_moe_staggered`` is this class compiled, with identical weights."""

    config_class = Nucleus2MoEStaggeredConfig
    config_from_dict = staticmethod(_config_from_dict)
    expected_fluid_params = [
        "inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas",
        "stefan", "prandtl", "bulk_temp", "sat_temp",
    ]
    expected_heater_params = ["wallTemp", "xMin", "xMax"]
    expected_global_params = ["gravy"]
    # Input and output channels are the same per-cell split.
    expected_fields = [
        "dfun", "temperature",
        "vel_left", "vel_right", "vel_bottom", "vel_top",
    ]
    num_sim_params = len(expected_fluid_params) + len(expected_heater_params) + len(expected_global_params)
    layout = "t h w c"
    # Rollout code uses this to choose staggered (natural-grid) trajectories.
    staggered_io = True

    def __init__(self, config: Nucleus2MoEStaggeredConfig):
        super().__init__()
        self.config = config
        n_fields = len(self.expected_fields)

        self.embed_dtype = get_dtype(config.embed_dtype)
        self.debed_dtype = get_dtype(config.debed_dtype)

        self.rotary_emb = RotaryEmbedding(
            dim=(config.embed_dim // config.num_heads) // 3,
            freqs_for="pixel",
            max_freq=256,
            seq_before_head_dim=True,
        )

        drop_path_probs = torch.linspace(0.0, 0.1, config.processor_blocks)
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                config=config,
                num_sim_params=self.num_sim_params,
                drop_path_prob=drop_path_probs[idx].item(),
            )
            for idx in range(config.processor_blocks)
        ])

        self.out_norm = nn.RMSNorm(config.embed_dim, dtype=self.debed_dtype)

        assert config.patching in ("Linear", "Adaptive")
        if config.patching == "Linear":
            self.embed = LinearEmbed(
                patch_size=config.patch_size, in_channels=n_fields,
                embed_dim=config.embed_dim, dtype=self.embed_dtype,
            )
            self.debed = LinearDebed(
                patch_size=config.patch_size, embed_dim=config.embed_dim,
                out_channels=n_fields, dtype=self.debed_dtype,
            )
        else:
            self.embed = AdaptiveEmbed(
                in_channels=n_fields, out_channels=config.embed_dim,
                out_shape=(16, 16), dtype=config.embed_dtype,
            )
            self.debed = AdaptiveDebed(
                in_channels=config.embed_dim, out_channels=n_fields,
                patch_shape=(16, 16), dtype=config.debed_dtype,
            )

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": _config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = _config_from_dict(state["config"])

    def forward(self, batch: DivFreeBatch, normalizer=None) -> Nucleus2MoEStaggeredOutput:
        return self.step(batch)

    def model_step(self, input, sim_params: torch.Tensor):
        """Backbone: per-cell channels in, per-cell channels out ``(B, T, H, W, 6)``."""
        cells = divfree_input_to_cells(input.sdf, input.temperature, input.velx, input.vely)
        assert cells.dtype == torch.float32, f"expected float32, got {cells.dtype}"
        _, _, h, w, _ = cells.shape

        with record_function("encode"):
            x = embed = self.embed(cells.to(self.embed_dtype))
        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
            x = self.debed(x, target_shape=(h, w))
        return x, moe_outputs

    def step(self, input, sim_params: Optional[torch.Tensor] = None, normalizer=None, **unused):
        """One forecast window. ``input`` is a :class:`DivFreeBatch` (which carries the
        conditioning tensor) or any container with ``sdf``, ``temperature``, ``velx``,
        ``vely`` on their natural grids, with ``sim_params`` passed separately. Extra
        keyword arguments used by the physics model's ``step`` (``x_coords``,
        ``sim_params_dict``, ...) are accepted and ignored, so both models share the
        training loop."""
        if isinstance(input, DivFreeBatch):
            sim_params = input.sim_params_tensor
            input = input.input
        assert sim_params is not None and sim_params.dtype == torch.float32

        x, moe_outputs = self.model_step(input, sim_params)
        sdf, temperature, vel_left, vel_right, vel_bottom, vel_top = x.float().unbind(-1)
        velx, vely = faces_from_cell_split(vel_left, vel_right, vel_bottom, vel_top)
        return Nucleus2MoEStaggeredOutput(
            sdf=sdf, temperature=temperature, velx=velx, vely=vely, moe_outputs=moe_outputs,
        )

    def _normalized_sim_params(self, sim_params_dict: dict, normalizer, device, batch_size: int) -> torch.Tensor:
        normalized = normalizer.normalize_params([sim_params_dict])[0]
        values = (
            [normalized[param] for param in self.expected_fluid_params]
            + [normalized["heater"][param] for param in self.expected_heater_params]
            + [normalized[param] for param in self.expected_global_params]
        )
        sim_params = torch.tensor(values, device=device, dtype=torch.float32)
        return sim_params[None, :].expand(batch_size, -1).contiguous()

    def forward_trajectory(
        self,
        trajectory: Trajectory,
        normalizer,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        return_moe_outputs: bool = False,
        clip_temp: bool = False,
    ):
        """Autoregressive rollout on the natural grids. ``trajectory`` and the result
        are in physical units; ``dx`` is accepted for a common signature."""
        assert input_time_window_size <= trajectory.num_steps
        sim_params_dict = trajectory.sim_params[0]
        bulk_temp = sim_params_dict["bulk_temp"]
        sim_params = self._normalized_sim_params(
            sim_params_dict, normalizer, trajectory.sdf.device, trajectory.sdf.shape[0]
        )
        keep = slice(-output_time_window_size, None)
        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            window = trajectory.last(input_time_window_size)
            model_input = Nucleus2MoEDivFreeInput(
                sdf=normalizer.normalize_sdf(window.sdf),
                temperature=normalizer.normalize_temp(window.temp, bulk_temp),
                velx=normalizer.normalize_velx(window.velx),
                vely=normalizer.normalize_vely(window.vely),
            )
            output = self.step(model_input, sim_params)
            sdf = normalizer.unnormalize_sdf(output.sdf)[:, keep]
            temp = normalizer.unnormalize_temp(output.temperature, bulk_temp)[:, keep]
            velx = normalizer.unnormalize_velx(output.velx)[:, keep]
            vely = normalizer.unnormalize_vely(output.vely)[:, keep]
            if clip_temp:
                temp = clip_temp_by_phase(
                    temp, sdf, sim_params_dict["sat_temp"], sim_params_dict["heater"]["wallTemp"],
                )
            trajectory = trajectory.extend(sdf, temp, velx, vely)
            if return_moe_outputs:
                trajectory_moe_outputs.append(output.moe_outputs)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory


@register_model("nucleus2_moe_staggered")
@torch.compile(fullgraph=True)
class Nucleus2MoEStaggered(Nucleus2MoEStaggeredBase):
    pass


class StaggeredForecastModule(DivFreeForecastModule):
    """Training module for :class:`Nucleus2MoEStaggered`.

    Shares the divfree module's training loop, augmentation, SDF-sign loss, router
    losses, and output-head diagnostics. The data loss is the divfree module's
    ``_field_loss`` (mean absolute error over all elements of the listed fields)
    restricted to the four data fields, with velocity compared on the faces after
    averaging. No physics-derived targets.
    """

    def _loss_fields(self, output: Nucleus2MoEStaggeredOutput, target: DivFreeData):
        names = ["sdf", "temperature", "velx", "vely"]
        return [getattr(output, n) for n in names], [getattr(target, n) for n in names]

    def _per_field_mae(self, output: Nucleus2MoEStaggeredOutput, target: DivFreeData, prefix: str) -> dict:
        metrics = {
            f"{prefix}/mae_{name}": (getattr(output, name) - getattr(target, name)).abs().mean()
            for name in ("sdf", "temperature", "velx", "vely")
        }
        # Diagnostic only: discrete divergence of predicted vs. true faces, in units of
        # the normalized velocity per grid spacing. Not part of the loss.
        div_pred = divergence_centers_from_faces(output.velx, output.vely, GRADIENT_SPACING, GRADIENT_SPACING)
        div_true = divergence_centers_from_faces(target.velx, target.vely, GRADIENT_SPACING, GRADIENT_SPACING)
        metrics[f"{prefix}/mae_divergence"] = (div_pred - div_true).abs().mean()
        return metrics

    @staticmethod
    def _data_cell_tensor(data: DivFreeData) -> torch.Tensor:
        return divfree_input_to_cells(data.sdf, data.temperature, data.velx, data.vely)
