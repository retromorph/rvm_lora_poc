"""CUDA smoke test on the real pinned roberta-base: 2 optimizer steps (8 x 4 microbatches), frozen head,
LoRA-only gradients, exact B24 accounting, real-model Jacobian vs JVP, and PEFT-native EVA budget.

Uses only rows of the head-warm-start and direction splits (never audit or official validation).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.cuda

need = [ROOT / "data" / "qnli_head.npz", ROOT / "data" / "qnli_directions.npz"]
if not torch.cuda.is_available() or not all(p.exists() for p in need):
    pytest.skip("requires CUDA and prepared splits", allow_module_level=True)


def _sub(enc, n):
    from rvm_lora.data import Encoded

    return Encoded(enc.idx[:n], enc.input_ids[:n], enc.attention_mask[:n], enc.labels[:n])


@pytest.fixture(scope="module")
def encs():
    from rvm_lora.data import load_encoded

    return load_encoded(need[0]), load_encoded(need[1])


def test_two_step_train_eval(cfg, encs, record):
    from rvm_lora.adapters import assert_exact_layout, build_peft
    from rvm_lora.modeling import configure_precision, evaluate, head_digest, load_backbone
    from rvm_lora.train import train_adapter

    configure_precision()
    head_enc, _ = encs
    model = load_backbone(0)
    h0 = head_digest(model)
    pm = build_peft(model, {m: 1 for m in cfg.modules}, cfg, run_seed=0)
    acct = assert_exact_layout(pm, {m: 1 for m in cfg.modules}, cfg.modules, cfg.rank_unit_params)
    assert acct["adapter_params"] == 36864
    torch.cuda.reset_peak_memory_stats()
    before = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_B" in n}
    res = train_adapter(pm, _sub(head_enc, 64), cfg, run_seed=0, steps=2)
    peak = torch.cuda.max_memory_allocated() / 2**20
    assert res["steps"] == 2 and res["microbatch"] == 8 and res["accumulation"] == 4
    changed = any(not torch.equal(before[n], p) for n, p in pm.named_parameters() if "lora_B" in n)
    assert changed, "LoRA parameters did not update"
    assert head_digest(pm.base_model.model) == h0
    ev = evaluate(pm, _sub(head_enc, 32))
    assert np.isfinite(ev["loss"])
    assert peak < 16 * 1024
    record(peak_gpu_memory_mb=peak, precision=res["precision"], train_wall_2steps_s=res["train_wall_s"],
           eval_loss_smoke=ev["loss"])


def test_real_model_jacobian_vs_jvp(cfg, encs, record):
    from torch.func import functional_call, jvp

    from rvm_lora.modeling import configure_precision, load_backbone
    from rvm_lora.probe import direction_pass, feature_pass

    configure_precision()
    _, dir_enc = encs
    model = load_backbone(0).eval()
    mods = [cfg.modules[0], cfg.modules[13], cfg.modules[23]]
    dirs = direction_pass(model, _sub(dir_enc, 16), mods, 2, 1e-4)
    sub = _sub(dir_enc, 8)
    feats = feature_pass(model, sub, mods, dirs["U"], dirs["V"], 1e-4)
    ids = torch.as_tensor(sub.input_ids, device="cuda")
    am = torch.as_tensor(sub.attention_mask, device="cuda")
    params = dict(model.named_parameters())
    worst = 0.0
    for m, name in enumerate(mods):
        W = params[name + ".weight"]
        D = torch.outer(dirs["U"][m, :, 0], dirs["V"][m, :, 0]).float().cuda()

        def z(s):
            out = functional_call(model, {**params, name + ".weight": W + s * D}, (),
                                  {"input_ids": ids, "attention_mask": am}).logits
            return out[:, 1] - out[:, 0]

        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):  # forward-mode AD is implemented for the math kernel
            _, t = jvp(z, (torch.zeros((), device="cuda"),), (torch.ones((), device="cuda"),))
        col = feats["J"][:, m * 2]
        ref = t.double().cpu().numpy()
        mask = np.abs(ref) > 1e-4 * np.abs(ref).max()
        worst = max(worst, float(np.max(np.abs(col - ref)[mask] / np.abs(ref)[mask])))
    assert worst < 1e-3
    assert np.all(np.abs(feats["J"]).sum(0) > 0)
    record(max_rel_err_real_model_vs_jvp=worst)


def test_eva_budget_and_init(cfg, encs, record):
    from peft import EvaConfig, get_peft_model

    from rvm_lora.adapters import count_trainable, lora_config, lora_layout
    from rvm_lora.baselines import eva_initialize
    from rvm_lora.modeling import batches, load_backbone

    _, dir_enc = encs
    model = load_backbone(0)
    pm = get_peft_model(model, lora_config({m: 1 for m in cfg.modules}, cfg, init_lora_weights="eva",
                                           eva_config=EvaConfig(rho=8.0)))
    sub = _sub(dir_enc, 128)
    bl = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
          for b in batches(sub, np.arange(len(sub)), 32, "cuda")]
    eva_initialize(pm, bl)
    for n, p in pm.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)
    lay = lora_layout(pm)
    total = sum(v["params"] for v in lay.values())
    assert total <= 36864 and max(v["rank"] for v in lay.values()) <= 8
    assert set(lay) <= set(cfg.modules)
    assert count_trainable(pm)["non_lora_trainable"] == 0
    for mod in pm.modules():
        if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}):
            assert float(mod.lora_B["default"].weight.abs().max()) == 0.0
    record(eva_params=total, eva_rank_sum=sum(v["rank"] for v in lay.values()),
           eva_max_rank=max(v["rank"] for v in lay.values()), eva_modules=len(lay))
