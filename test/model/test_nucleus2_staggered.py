import random

import pytest
import torch

from nucleus.models import get_model
from nucleus.models.modules import get_train_module
from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFreeInput, divfree_input_to_cells
from nucleus.models.nucleus2_moe_staggered import (
    Nucleus2MoEStaggeredBase,
    Nucleus2MoEStaggeredConfig,
    Nucleus2MoEStaggeredOutput,
    StaggeredForecastModule,
    faces_from_cell_split,
)
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch
from nucleus.data.normalize import DivFreeNormalizer, NormalizerConstants
from nucleus.trajectory import Trajectory


def small_config() -> Nucleus2MoEStaggeredConfig:
    return Nucleus2MoEStaggeredConfig(
        patch_size=16, embed_dim=32, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )


@pytest.fixture
def model() -> Nucleus2MoEStaggeredBase:
    return Nucleus2MoEStaggeredBase(small_config()).eval()


class _IdentityParamsNormalizer(DivFreeNormalizer):
    def normalize_params(self, sim_params_dicts):
        return sim_params_dicts

    def unnormalize_params(self, sim_params_dicts):
        return sim_params_dicts


def make_normalizer() -> _IdentityParamsNormalizer:
    constants = NormalizerConstants(
        max_domain_size=9.0, sdf_mean=0.0, sdf_std=1.0, absmax_temp=1.0,
        temp_mean=0.0, temp_std=1.0, velx_mean=0.0, velx_std=1.0, vely_mean=0.0,
        vely_std=1.0, psi_mean=0.0, psi_std=1.0, phi_mean=0.0, phi_std=1.0,
    )
    return _IdentityParamsNormalizer(constants)


def make_input(batch_size, time, height, width) -> Nucleus2MoEDivFreeInput:
    return Nucleus2MoEDivFreeInput(
        sdf=torch.randn(batch_size, time, height, width),
        temperature=torch.randn(batch_size, time, height, width),
        velx=torch.randn(batch_size, time, height, width + 1),
        vely=torch.randn(batch_size, time, height + 1, width),
    )


def test_faces_from_cell_split_inverts_input_split():
    # Splitting true faces onto cells and averaging back must return them exactly:
    # both copies of an interior face hold the same value.
    torch.manual_seed(0)
    height, width = 7, 5
    sdf, temp = torch.randn(2, 3, height, width), torch.randn(2, 3, height, width)
    velx, vely = torch.randn(2, 3, height, width + 1), torch.randn(2, 3, height + 1, width)
    cells = divfree_input_to_cells(sdf, temp, velx, vely)
    velx_back, vely_back = faces_from_cell_split(*cells[..., 2:].unbind(-1))
    assert torch.equal(velx_back, velx)
    assert torch.equal(vely_back, vely)


def test_faces_from_cell_split_averages_interior_and_keeps_walls():
    height, width = 3, 4
    left = torch.full((height, width), 1.0)
    right = torch.full((height, width), 3.0)
    bottom = torch.full((height, width), -2.0)
    top = torch.full((height, width), 4.0)
    velx, vely = faces_from_cell_split(left, right, bottom, top)
    assert velx.shape == (height, width + 1) and vely.shape == (height + 1, width)
    assert torch.all(velx[:, 0] == 1.0) and torch.all(velx[:, -1] == 3.0)
    assert torch.all(velx[:, 1:-1] == 2.0)
    assert torch.all(vely[0] == -2.0) and torch.all(vely[-1] == 4.0)
    assert torch.all(vely[1:-1] == 1.0)


def test_step_returns_faces(model):
    torch.manual_seed(0)
    batch_size, time, height, width = 2, 2, 64, 48
    sim_params = torch.randn(batch_size, model.num_sim_params)
    with torch.no_grad():
        output = model.step(make_input(batch_size, time, height, width), sim_params)
    assert isinstance(output, Nucleus2MoEStaggeredOutput)
    assert output.sdf.shape == (batch_size, time, height, width)
    assert output.temperature.shape == (batch_size, time, height, width)
    assert output.velx.shape == (batch_size, time, height, width + 1)
    assert output.vely.shape == (batch_size, time, height + 1, width)
    for field in (output.sdf, output.temperature, output.velx, output.vely):
        assert torch.isfinite(field).all()
    assert output.to_cell_tensor().shape == (batch_size, time, height, width, 6)


def test_step_accepts_batch_and_ignores_physics_kwargs(model):
    # The training loop passes x_coords / sim_params_dict for the physics model; the
    # data-only model accepts and ignores them, and a DivFreeBatch carries sim_params.
    torch.manual_seed(0)
    # patch 16 needs >= 3 tokens per side for the 3x3 neighborhood attention, so >= 48 cells
    data = make_input(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    with torch.no_grad():
        direct = model.step(data, sim_params, make_normalizer(),
                            x_coords=torch.zeros(64), sim_params_dict=[{}])
        batch = DivFreeBatch(input=data, target=None, sim_params=[{}],
                             dx=torch.ones(1), dy=torch.ones(1), sim_params_tensor=sim_params)
        via_batch = model.step(batch)
    assert torch.allclose(direct.velx, via_batch.velx)


def test_registered_model_and_train_module():
    # The registered model is the compiled class; check it by behavior, not isinstance.
    model = get_model("nucleus2_moe_staggered", **vars(small_config()))
    assert getattr(model, "staggered_io", False)
    assert len(model.expected_fields) == 6
    assert get_train_module("staggered_forecast") is StaggeredForecastModule


def test_forward_trajectory_extends_on_natural_grids(model):
    torch.manual_seed(0)
    height, width, window = 64, 64, 4
    params = {"bulk_temp": 50.0, "sat_temp": 58.0}
    params.update({p: random.random() for p in model.expected_fluid_params if p not in params})
    params["heater"] = {p: random.random() for p in model.expected_heater_params}
    params.update({p: random.random() for p in model.expected_global_params})
    trajectory = Trajectory(
        sdf=torch.randn(1, window, height, width),
        temp=torch.randn(1, window, height, width),
        velx=torch.randn(1, window, height, width + 1),
        vely=torch.randn(1, window, height + 1, width),
        sim_params=[params],
    )
    with torch.no_grad():
        out = model.forward_trajectory(
            trajectory, make_normalizer(), dx=1 / 32,
            input_time_window_size=window, output_time_window_size=window, trajectory_steps=12,
        )
    assert out.num_steps == 12
    assert out.velx.shape[-2:] == (height, width + 1)
    assert out.vely.shape[-2:] == (height + 1, width)
    assert torch.isfinite(out.velx).all() and torch.isfinite(out.vely).all()
