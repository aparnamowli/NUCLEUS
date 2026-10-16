"""Tests for fixed_params: the staggered models can be blind to chosen simulation parameters."""
import pytest
import torch

from nucleus.models.nucleus2_moe_staggered import Nucleus2MoEStaggeredBase, Nucleus2MoEStaggeredConfig
from nucleus.models.nucleus2_moe_staggered_backbone import StaggeredFields, config_to_dict
from nucleus.models.nucleus2_moe_staggered_solin import (
    Nucleus2MoEStaggeredSolinBase,
    Nucleus2MoEStaggeredSolinConfig,
)

SMALL = dict(patch_size=16, embed_dim=32, num_heads=2, processor_blocks=1, num_experts=2,
             topk=1, moe_intermediate_dim=32, patching="Linear", activation_dtype="float32",
             attention_dtype="float32", moe_dtype="float32")


def make_input(b=2, t=8, h=64, w=48) -> StaggeredFields:
    return StaggeredFields(
        sdf=torch.randn(b, t, h, w), temperature=torch.randn(b, t, h, w),
        velx=torch.randn(b, t, h, w + 1), vely=torch.randn(b, t, h + 1, w),
    )


def model(fixed="", cls=Nucleus2MoEStaggeredBase, cfg=Nucleus2MoEStaggeredConfig, **kw):
    torch.manual_seed(0)
    return cls(cfg(**SMALL, fixed_params=fixed, **kw)).float().eval()


def index(m, name):
    names = m.expected_fluid_params + m.expected_heater_params + m.expected_global_params
    return names.index(name)


def test_default_sees_every_parameter():
    m = model()
    assert m.config.fixed_params == "" and m.fixed_param_index.numel() == 0
    sp = torch.randn(2, m.num_sim_params)
    assert torch.equal(m.fix_params(sp), sp)


def test_fixed_entries_are_zero_and_the_rest_unchanged():
    m = model("wallTemp, gravy")
    sp = torch.randn(2, m.num_sim_params)
    f = m.fix_params(sp)
    i, j = index(m, "wallTemp"), index(m, "gravy")
    assert torch.all(f[:, [i, j]] == 0)
    keep = [k for k in range(m.num_sim_params) if k not in (i, j)]
    assert torch.equal(f[:, keep], sp[:, keep])
    assert not torch.all(sp[:, i] == 0)  # the caller's tensor is not modified


@pytest.mark.parametrize("cls,cfg,kw", [
    (Nucleus2MoEStaggeredBase, Nucleus2MoEStaggeredConfig, {}),
    (Nucleus2MoEStaggeredBase, Nucleus2MoEStaggeredConfig, {"input_frames": 1}),
    (Nucleus2MoEStaggeredSolinBase, Nucleus2MoEStaggeredSolinConfig, {"input_frames": 1}),
])
def test_output_ignores_the_fixed_parameter(cls, cfg, kw):
    x = make_input()
    blind, seeing = model("wallTemp", cls, cfg, **kw), model("", cls, cfg, **kw)
    sp = torch.randn(2, blind.num_sim_params)
    sp2 = sp.clone()
    sp2[:, index(blind, "wallTemp")] += 3.0
    with torch.no_grad():
        a, b = blind.step(x, sp), blind.step(x, sp2)
        for f in ("sdf", "temperature", "velx", "vely"):
            assert torch.equal(getattr(a, f), getattr(b, f)), f
        assert not torch.equal(seeing.step(x, sp).sdf, seeing.step(x, sp2).sdf)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        model("wallTemperature")


def test_config_round_trip_old_configs_and_state_dict():
    m = model("wallTemp")
    d = config_to_dict(m.config)
    assert d["fixed_params"] == "wallTemp"
    assert Nucleus2MoEStaggeredBase.config_from_dict(d).fixed_params == "wallTemp"
    d.pop("fixed_params")  # a checkpoint saved before the field existed
    assert Nucleus2MoEStaggeredBase.config_from_dict(d).fixed_params == ""
    assert "fixed_param_index" not in m.state_dict()
    model("").load_state_dict(m.state_dict())  # same weights with and without the field
