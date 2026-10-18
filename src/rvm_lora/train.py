"""Fixed-protocol LoRA training with frozen backbone and head, and FP32 final-checkpoint evaluation."""

from __future__ import annotations

import time

import numpy as np
import torch

from .adapters import LORA_KEYS
from .data import Encoded
from .modeling import evaluate, head_digest, sample_order


def amp_dtype_for(device) -> torch.dtype | None:
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if str(device).startswith("cuda"):
        return torch.float16
    return None


def train_adapter(peft_model, train_enc: Encoded, cfg, run_seed: int, device="cuda", microbatch: int | None = None,
                  steps: int | None = None, log_every: int = 20, log_fn=None) -> dict:
    """AdamW, linear warmup/decay, effective batch 32 = microbatch x accumulation. No gradient clipping.

    Data order depends only on (run_seed, stream=202), so it is identical across methods.
    """
    at = cfg["adapter_training"]
    eff = int(at["effective_batch_size"])
    micro = int(microbatch or at["microbatch_size"])
    if eff % micro:
        raise ValueError("microbatch must divide effective batch")
    accum = eff // micro
    steps = int(steps or at["steps"])
    from transformers import get_linear_schedule_with_warmup

    params = [p for n, p in peft_model.named_parameters() if p.requires_grad]
    if not params or not all(any(k in n for k in LORA_KEYS) for n, p in peft_model.named_parameters() if p.requires_grad):
        raise AssertionError("trainable set must be LoRA-only and non-empty")
    opt = torch.optim.AdamW(params, lr=float(at["learning_rate"]), weight_decay=float(at["weight_decay"]))
    warm = int(round(at["warmup_steps"] * steps / at["steps"])) if steps != at["steps"] else int(at["warmup_steps"])
    sched = get_linear_schedule_with_warmup(opt, warm, steps)
    order = sample_order(len(train_enc), steps * eff, run_seed, stream=202)
    amp = amp_dtype_for(device)
    scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 else None
    torch.manual_seed(run_seed)  # dropout stream
    peft_model.train()
    head0 = head_digest(peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model)
    losses = []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        sel = order[step * eff:(step + 1) * eff]
        step_loss = 0.0
        for k in range(accum):
            mb = sel[k * micro:(k + 1) * micro]
            ids = torch.as_tensor(train_enc.input_ids[mb], device=device)
            am = torch.as_tensor(train_enc.attention_mask[mb], device=device)
            y = torch.as_tensor(train_enc.labels[mb], device=device)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits = peft_model(input_ids=ids, attention_mask=am).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y) / accum
            (scaler.scale(loss) if scaler else loss).backward()
            step_loss += float(loss.detach())
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if not np.isfinite(step_loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        losses.append(step_loss)
        if log_fn and (step + 1) % log_every == 0:
            log_fn({"train_loss": float(np.mean(losses[-log_every:]))}, step + 1)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    head1 = head_digest(peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model)
    if head0 != head1:
        raise AssertionError("classifier head changed during adapter training")
    return {"train_wall_s": wall, "train_loss_final": float(np.mean(losses[-20:])),
            "train_loss_first": float(np.mean(losses[:20])), "steps": steps, "microbatch": micro,
            "accumulation": accum, "precision": str(amp).replace("torch.", "") if amp else "float32",
            "head_digest": head0, "train_loss_curve": losses}


def train_with_oom_fallback(build_fn, train_enc, cfg, run_seed, device="cuda", log_fn=None, steps=None) -> tuple:
    """Try microbatch 8 x 4; on CUDA OOM rebuild from scratch and use 4 x 8 (logged)."""
    at = cfg["adapter_training"]
    events = []
    for micro in (int(at["microbatch_size"]), int(at["oom_fallback_microbatch"])):
        peft_model, meta = build_fn()
        try:
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            res = train_adapter(peft_model, train_enc, cfg, run_seed, device, micro, steps=steps, log_fn=log_fn)
            res["peak_gpu_memory_mb"] = (torch.cuda.max_memory_allocated() / 2**20) if str(device).startswith("cuda") else 0.0
            res["oom_events"] = events
            return peft_model, meta, res
        except torch.cuda.OutOfMemoryError as e:
            events.append({"microbatch": micro, "error": str(e)[:300]})
            del peft_model
            torch.cuda.empty_cache()
    raise RuntimeError(f"OOM even with fallback microbatch: {events}")


def eval_model(model, enc: Encoded, device="cuda") -> dict:
    return evaluate(model, enc, device)
