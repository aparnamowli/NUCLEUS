"""Tests for input_frames: the staggered models see only the most recent input frames."""
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


def with_history(x: StaggeredFields, other: StaggeredFields, keep: int) -> StaggeredFields:
    """x with every frame before the last `keep` taken from other."""
    def mix(a, o):
        return torch.cat([o[:, :-keep], a[:, -keep:]], dim=1)
    return StaggeredFields(*(mix(getattr(x, f), getattr(other, f)) for f in ("sdf", "temperature", "velx", "vely")))


def model(n, cls=Nucleus2MoEStaggeredBase, cfg=Nucleus2MoEStaggeredConfig):
    torch.manual_seed(0)
    return cls(cfg(**SMALL, input_frames=n)).float().eval()


def test_default_keeps_the_whole_window():
    assert Nucleus2MoEStaggeredConfig(**SMALL).input_frames == 0
    x = make_input()
    k = model(0).keep_recent_frames(x)
    assert all(torch.equal(getattr(k, f), getattr(x, f)) for f in ("sdf", "temperature", "velx", "vely"))


def test_keep_recent_frames_repeats_the_oldest_kept():
    x = make_input()
    k = model(1).keep_recent_frames(x)
    for f in ("sdf", "temperature", "velx", "vely"):
        a, b = getattr(k, f), getattr(x, f)
        assert a.shape == b.shape
        assert torch.equal(a, b[:, -1:].expand_as(b))
    k3 = model(3).keep_recent_frames(x)
    assert torch.equal(k3.sdf[:, -3:], x.sdf[:, -3:])
    assert torch.equal(k3.sdf[:, :5], x.sdf[:, 5:6].expand(-1, 5, -1, -1))


def test_output_ignores_earlier_frames():
    x, y = make_input(), make_input()
    sp = torch.randn(2, model(1).num_sim_params)
    for n in (1, 3):
        m = model(n)
        with torch.no_grad():
            a, b = m.step(x, sp), m.step(with_history(x, y, n), sp)
        for f in ("sdf", "temperature", "velx", "vely"):
            assert torch.equal(getattr(a, f), getattr(b, f)), (n, f)
    m = model(0)
    with torch.no_grad():
        assert not torch.equal(m.step(x, sp).sdf, m.step(with_history(x, y, 1), sp).sdf)


def test_solin_model_inherits_input_frames():
    m = model(1, Nucleus2MoEStaggeredSolinBase, Nucleus2MoEStaggeredSolinConfig)
    x, y = make_input(), make_input()
    sp = torch.randn(2, m.num_sim_params)
    with torch.no_grad():
        a, b = m.step(x, sp), m.step(with_history(x, y, 1), sp)
    assert torch.equal(a.velx, b.velx) and torch.equal(a.sdf, b.sdf)


def test_config_round_trip_and_old_configs():
    m = model(1)
    d = config_to_dict(m.config)
    assert d["input_frames"] == 1
    assert Nucleus2MoEStaggeredBase.config_from_dict(d).input_frames == 1
    d.pop("input_frames")  # a checkpoint saved before the field existed
    assert Nucleus2MoEStaggeredBase.config_from_dict(d).input_frames == 0
