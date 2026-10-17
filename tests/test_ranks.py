"""Gate 0 item 5: exact PEFT placement/ranks, parameter counts, frozen head, rank-0 omission."""

import pytest
import torch

from rvm_lora.adapters import assert_exact_layout, build_peft, count_trainable, lora_layout

MODS = [f"roberta.encoder.layer.{l}.attention.self.{p}" for l in range(12) for p in ("query", "value")]


@pytest.fixture(scope="module")
def base_factory():
    from transformers import RobertaConfig, RobertaForSequenceClassification

    conf = RobertaConfig(num_labels=2)  # roberta-base geometry: 12 layers, hidden 768

    def make():
        torch.manual_seed(0)
        m = RobertaForSequenceClassification(conf)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    return make


def _check(base_factory, cfg, ranks, expected_params, record=None, label=None):
    pm = build_peft(base_factory(), ranks, cfg, run_seed=42)
    acct = assert_exact_layout(pm, ranks, cfg.modules, cfg.rank_unit_params)
    assert acct["adapter_params"] == expected_params
    assert not any(n.startswith("base_model.model.classifier") and p.requires_grad for n, p in pm.named_parameters())
    if record:
        record(**{label: acct["adapter_params"]})
    return pm


def test_architecture_param_counts(base_factory, cfg, record):
    for a in cfg.architectures():
        _check(base_factory, cfg, a.ranks, a.parameters)
    assert sorted({a.parameters for a in cfg.architectures()}) == [6144, 12288]
    _check(base_factory, cfg, {MODS[2]: 2, MODS[3]: 2}, 6144, record, "q2_plus_v2")
    _check(base_factory, cfg, {MODS[2]: 4, MODS[3]: 4}, 12288, record, "q4_plus_v4")


def test_uniform_budgets(base_factory, cfg, record):
    _check(base_factory, cfg, {m: 1 for m in MODS}, 36864, record, "B24")
    _check(base_factory, cfg, {m: 2 for m in MODS}, 73728, record, "B48")


def test_no_suffix_collision_and_rank_zero_omitted(base_factory, cfg):
    # layer 1 must never match layer 11; rank 0 entries must be omitted, not passed to PEFT
    ranks = {"roberta.encoder.layer.1.attention.self.query": 3, "roberta.encoder.layer.11.attention.self.value": 1,
             "roberta.encoder.layer.5.attention.self.query": 0}
    pm = build_peft(base_factory(), ranks, cfg, run_seed=1)
    lay = lora_layout(pm)
    assert set(lay) == {"roberta.encoder.layer.1.attention.self.query", "roberta.encoder.layer.11.attention.self.value"}
    assert lay["roberta.encoder.layer.1.attention.self.query"]["rank"] == 3
    assert all(v["rank"] > 0 for v in lay.values())
    assert count_trainable(pm)["lora_trainable"] == 4 * 1536
    # rsLoRA scaling alpha / sqrt(r)
    assert abs(lay["roberta.encoder.layer.1.attention.self.query"]["scaling"] - 4 / 3 ** 0.5) < 1e-12


def test_layout_assertion_catches_mismatch(base_factory, cfg):
    pm = build_peft(base_factory(), {MODS[0]: 2}, cfg, run_seed=0)
    with pytest.raises(AssertionError):
        assert_exact_layout(pm, {MODS[0]: 3}, cfg.modules, cfg.rank_unit_params)


def test_identical_zero_update_init_across_methods(base_factory, cfg):
    a = build_peft(base_factory(), {MODS[0]: 2, MODS[5]: 1}, cfg, run_seed=42)
    b = build_peft(base_factory(), {MODS[0]: 5}, cfg, run_seed=42)

    def A(pm, name):
        for n, mod in pm.named_modules():
            if n.endswith(name) and hasattr(mod, "lora_A"):
                return mod.lora_A["default"].weight.detach(), mod.lora_B["default"].weight.detach()

    Aa, Ba = A(a, MODS[0])
    Ab, Bb = A(b, MODS[0])
    assert torch.equal(Aa, Ab[:2]) and float(Ba.abs().max()) == 0 and float(Bb.abs().max()) == 0
    c = build_peft(base_factory(), {MODS[0]: 2}, cfg, run_seed=2025)
    assert not torch.equal(A(c, MODS[0])[0], Aa)
