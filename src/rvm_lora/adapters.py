"""Exact PEFT LoRA placement, deterministic zero-update initialisation and parameter accounting.

* `target_modules` is a list of *full* module names (PEFT matches list entries by exact name or
  '.'-suffix; a full name can only match itself). `rank_pattern` keys are regex-escaped full names.
* rank 0 means the module is omitted from `target_modules`; PEFT never sees r=0.
* No `task_type` is passed, so PEFT does not add the classifier to `modules_to_save`; the head stays
  frozen. Assertions verify this.
* Non-EVA methods share one zero-update initialisation: for (run seed, module) a fixed 8 x 768
  Kaiming-uniform matrix (bound 1/sqrt(768), as PEFT's default) is drawn and its first r rows are
  used as lora_A; lora_B = 0. Overlapping (module, rank) prefixes are therefore identical across methods.
"""

from __future__ import annotations

import hashlib
import math
import re

import torch

LORA_KEYS = ("lora_A", "lora_B")


def lora_config(ranks: dict[str, int], cfg, init_lora_weights=True, eva_config=None):
    from peft import LoraConfig

    at = cfg["adapter_training"]
    ranks = {k: int(v) for k, v in ranks.items() if int(v) > 0}
    if not ranks:
        raise ValueError("empty architecture: no module has rank > 0 (head-only model, no PEFT wrapper)")
    targets = sorted(ranks)
    kwargs = dict(
        r=1 if eva_config is not None else max(ranks.values()),
        lora_alpha=at["lora_alpha"],
        lora_dropout=at["lora_dropout"],
        use_rslora=at["use_rslora"],
        bias="none",
        target_modules=targets,
        init_lora_weights=init_lora_weights,
    )
    if eva_config is None:
        kwargs["rank_pattern"] = {re.escape(k): v for k, v in ranks.items()}
    else:
        kwargs["eva_config"] = eva_config
    return LoraConfig(**kwargs)


def _module_seed(run_seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_seed}|{name}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


def deterministic_lora_init(peft_model, run_seed: int, max_rank: int = 8) -> None:
    """lora_A <- first r rows of a per-(seed, module) Kaiming-uniform 8 x in matrix; lora_B <- 0."""
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in getattr(module, "lora_A", {}):
            continue
        base_name = canonical_name(name)
        A = module.lora_A["default"].weight
        r, fan_in = A.shape
        g = torch.Generator().manual_seed(_module_seed(run_seed, base_name))
        bound = 1.0 / math.sqrt(fan_in)
        full = (torch.rand(max_rank, fan_in, generator=g, dtype=torch.float32) * 2 - 1) * bound
        with torch.no_grad():
            A.copy_(full[:r].to(A.device, A.dtype))
            module.lora_B["default"].weight.zero_()


def canonical_name(peft_name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if peft_name.startswith(prefix):
            peft_name = peft_name[len(prefix):]
            break
    return peft_name


def lora_layout(peft_model) -> dict[str, dict]:
    """Actual per-module LoRA rank, scaling and parameter count."""
    out = {}
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            out[canonical_name(name)] = {
                "rank": int(A.shape[0]),
                "params": int(A.numel() + B.numel()),
                "scaling": float(module.scaling["default"]),
                "in": int(A.shape[1]),
                "out": int(B.shape[0]),
            }
    return out


def count_trainable(model) -> dict:
    lora, other, names_other = 0, 0, []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in LORA_KEYS):
            lora += p.numel()
        else:
            other += p.numel()
            names_other.append(n)
    return {"lora_trainable": lora, "non_lora_trainable": other, "non_lora_names": names_other}


def assert_exact_layout(peft_model, expected: dict[str, int], eligible: list[str], unit_params: int) -> dict:
    """Hard assertions: exact module set and ranks, only eligible modules, LoRA-only gradients."""
    expected = {k: int(v) for k, v in expected.items() if int(v) > 0}
    layout = lora_layout(peft_model)
    actual = {k: v["rank"] for k, v in layout.items()}
    if actual != expected:
        raise AssertionError(f"LoRA layout mismatch:\n expected {expected}\n actual   {actual}")
    bad = set(actual) - set(eligible)
    if bad:
        raise AssertionError(f"LoRA placed on non-eligible modules {bad}")
    if any(v["rank"] <= 0 for v in layout.values()):
        raise AssertionError("rank-0 LoRA instance found")
    counts = count_trainable(peft_model)
    exp_params = unit_params * sum(expected.values())
    if counts["lora_trainable"] != exp_params:
        raise AssertionError(f"trainable LoRA params {counts['lora_trainable']} != {exp_params}")
    if counts["non_lora_trainable"] != 0:
        raise AssertionError(f"non-LoRA trainable params: {counts['non_lora_names']}")
    return {"layout": layout, "adapter_params": counts["lora_trainable"], "rank_sum": sum(expected.values())}


def build_peft(model, ranks: dict[str, int], cfg, run_seed: int):
    from peft import get_peft_model

    peft_model = get_peft_model(model, lora_config(ranks, cfg))
    deterministic_lora_init(peft_model, run_seed, cfg.max_rank)
    for n, p in peft_model.named_parameters():
        p.requires_grad_(any(k in n for k in LORA_KEYS))
    return peft_model
