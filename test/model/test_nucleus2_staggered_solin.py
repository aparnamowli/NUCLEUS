"""Tests for the staggered data-only model with a solenoidal velocity input."""
import pathlib

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_staggered import Nucleus2MoEStaggeredBase, Nucleus2MoEStaggeredConfig
from nucleus.models.nucleus2_moe_staggered_backbone import StaggeredFields, config_to_dict
from nucleus.models.nucleus2_moe_staggered_solin import (
    Nucleus2MoEStaggeredSolinBase,
    Nucleus2MoEStaggeredSolinConfig,
)
from nucleus.physics.poisson import divergence_centers_from_faces, grad_faces_from_centers

DX = 1 / 32
# Unequal scales and nonzero means, as in the standard normalizer.
VEL = dict(velx_mean=-0.005, velx_std=0.45, vely_mean=0.055, vely_std=0.54)
SMALL = dict(patch_size=16, embed_dim=32, num_heads=2, processor_blocks=1, num_experts=2,
             topk=1, moe_intermediate_dim=32, patching="Linear")


def solin_model() -> Nucleus2MoEStaggeredSolinBase:
    return Nucleus2MoEStaggeredSolinBase(Nucleus2MoEStaggeredSolinConfig(**SMALL, **VEL)).eval()


def make_input(b=2, t=2, h=64, w=48) -> StaggeredFields:
    return StaggeredFields(
        sdf=torch.randn(b, t, h, w), temperature=torch.randn(b, t, h, w),
        velx=torch.randn(b, t, h, w + 1), vely=torch.randn(b, t, h + 1, w),
    )


def physical(model, velx, vely):
    c = model.config
    return velx * c.velx_std + c.velx_mean, vely * c.vely_std + c.vely_mean


def test_projected_input_is_divergence_free_in_physical_units():
    torch.manual_seed(0)
    model, x = solin_model(), make_input()
    p = model.project_input(x)
    vx, vy = physical(model, x.velx, x.vely)
    px, py = physical(model, p.velx, p.vely)
    div_in = divergence_centers_from_faces(vx, vy, DX, DX).abs().max()
    div_out = divergence_centers_from_faces(px, py, DX, DX).abs().max()
    assert div_out < 1e-4 * div_in
    assert torch.equal(p.sdf, x.sdf) and torch.equal(p.temperature, x.temperature)


def test_projection_is_idempotent():
    torch.manual_seed(0)
    model = solin_model()
    p1 = model.project_input(make_input())
    p2 = model.project_input(p1)
    assert torch.allclose(p1.velx, p2.velx, atol=1e-5) and torch.allclose(p1.vely, p2.vely, atol=1e-5)


def test_step_equals_base_model_on_projected_input():
    torch.manual_seed(0)
    solin = solin_model()
    base = Nucleus2MoEStaggeredBase(Nucleus2MoEStaggeredConfig(**SMALL)).eval()
    base.load_state_dict({k: v for k, v in solin.state_dict().items() if k != "_extra_state"}, strict=False)
    x = make_input()
    sp = torch.randn(2, solin.num_sim_params)
    with torch.no_grad():
        a = solin.step(x, sp)
        b = base.step(solin.project_input(x), sp)
    for f in ("sdf", "temperature", "velx", "vely"):
        assert torch.allclose(getattr(a, f), getattr(b, f), atol=1e-6), f


def test_output_ignores_the_expansion_part_of_the_input():
    # Adding a gradient field with the solver's boundary conditions to the physical input
    # velocity changes its divergence but not the projected input, so not the output.
    # The gradient has the size of the input velocity (as in the data, where the expansion
    # part is a few percent of it): a much larger one leaves float32 round-off of order
    # eps * |grad chi| after the projection. All dtypes float32 so the comparison is not
    # limited by bf16 attention.
    torch.manual_seed(0)
    fp32 = dict(attention_dtype="float32", moe_dtype="float32")
    # The attention layers create bf16 weights whatever the config says; cast them too
    # (as the analysis scripts do).
    model = Nucleus2MoEStaggeredSolinBase(Nucleus2MoEStaggeredSolinConfig(**SMALL, **VEL, **fp32)).float().eval()
    x = make_input()
    c = model.config
    chi = DX * torch.randn(2, 2, 64, 48, dtype=torch.float64)
    gx, gy = grad_faces_from_centers(chi, DX, DX)
    shifted = StaggeredFields(sdf=x.sdf, temperature=x.temperature,
                              velx=x.velx + (gx / c.velx_std).float(), vely=x.vely + (gy / c.vely_std).float())
    assert divergence_centers_from_faces(gx, gy, DX, DX).abs().max() > 10.0
    p, q = model.project_input(x), model.project_input(shifted)
    assert torch.allclose(p.velx, q.velx, atol=1e-5) and torch.allclose(p.vely, q.vely, atol=1e-5)
    sp = torch.randn(2, model.num_sim_params)
    with torch.no_grad():
        a, b = model.step(x, sp), model.step(shifted, sp)
    for f in ("sdf", "temperature", "velx", "vely"):
        assert torch.allclose(getattr(a, f), getattr(b, f), atol=1e-4), f


def test_registered_and_checkpoint_round_trip(tmp_path):
    from nucleus.models import load_model_from_checkpoint
    model = get_model("nucleus2_moe_staggered_solin", **SMALL, **VEL)
    assert model.config.velx_std == VEL["velx_std"]
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_model_from_checkpoint(str(path))
    assert config_to_dict(loaded.config) == config_to_dict(model.config)


def test_yaml_takes_velocity_constants_from_the_normalizer():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default", overrides=[
            "model_cfg=nucleus2/nucleus2_staggered_solin", "normalizer_cfg=staggered_standard", "data_dir=/tmp"])
    params = OmegaConf.to_container(cfg.model_cfg, resolve=True)["params"]
    for k in ("velx_mean", "velx_std", "vely_mean", "vely_std"):
        assert params[k] == cfg.normalizer_cfg[k], k
