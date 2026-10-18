"""Backbone loading, frozen classifier head checkpoints and head warm start."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import MODEL_REVISION, Config, namespaced, sha256_file
from .data import Encoded

HEAD_PREFIX = "classifier."


def set_determinism(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_precision() -> None:
    # Evaluation and probes run in true FP32; training uses BF16 autocast (logged per run).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_backbone(seed: int, device: str | torch.device = "cuda", model_id: str = "FacebookAI/roberta-base",
                  revision: str | None = MODEL_REVISION, config_override=None):
    """RoBERTa sequence classifier; the classifier head is randomly initialised under `seed`."""
    from transformers import AutoModelForSequenceClassification
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    set_determinism(seed)
    if config_override is not None:
        from transformers import RobertaForSequenceClassification

        model = RobertaForSequenceClassification(config_override)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, num_labels=2)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def head_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(HEAD_PREFIX)}


def head_digest(model) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(head_state(model).items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def head_path(root: Path, task: str, seed: int) -> Path:
    return namespaced(root / "checkpoints") / f"head_{task}_seed{seed}.safetensors"


def save_head(model, path: Path) -> str:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in head_state(model).items()}, str(path))
    return sha256_file(path)


def load_head(model, path: Path) -> str:
    from safetensors.torch import load_file

    sd = load_file(str(path))
    missing = set(head_state(model)) - set(sd)
    if missing:
        raise RuntimeError(f"head checkpoint missing keys {missing}")
    model.load_state_dict(sd, strict=False)
    return sha256_file(path)


def batches(enc: Encoded, order: np.ndarray, size: int, device):
    for i in range(0, len(order), size):
        sel = order[i:i + size]
        yield {
            "input_ids": torch.as_tensor(enc.input_ids[sel], device=device),
            "attention_mask": torch.as_tensor(enc.attention_mask[sel], device=device),
            "labels": torch.as_tensor(enc.labels[sel], device=device),
        }


def sample_order(n: int, total: int, seed: int, stream: int) -> np.ndarray:
    """Concatenated epoch permutations of range(n) of length `total`, PCG64([seed, stream])."""
    rng = np.random.default_rng([int(seed), int(stream)])
    out = []
    while sum(len(o) for o in out) < total:
        out.append(rng.permutation(n))
    return np.concatenate(out)[:total]


def warm_start_head(model, head_enc: Encoded, cfg: Config, seed: int, device="cuda", amp_dtype=torch.bfloat16) -> dict:
    """200 AdamW steps on the classifier head only (backbone frozen), batch 32, lr 1e-3."""
    hw = cfg["head_warm_start"]
    head_params = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(HEAD_PREFIX))
    opt = torch.optim.AdamW(head_params, lr=float(hw["learning_rate"]))
    steps, bs = int(hw["steps"]), int(hw["batch_size"])
    order = sample_order(len(head_enc), steps * bs, seed, stream=101)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for step, batch in enumerate(batches(head_enc, order, bs, device)):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and str(device).startswith("cuda")):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return {"head_train_loss_first20": float(np.mean(losses[:20])), "head_train_loss_last20": float(np.mean(losses[-20:])),
            "head_steps": len(losses), "head_wall_s": wall}


@torch.no_grad()
def evaluate(model, enc: Encoded, device="cuda", batch_size: int = 128) -> dict:
    """FP32 evaluation: per-example logits and binary CE (nats) aggregated in float64."""
    model.eval()
    order = np.arange(len(enc))
    logits_all = []
    for batch in batches(enc, order, batch_size, device):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits_all.append(out.float().cpu())
    logits = torch.cat(logits_all).double().numpy()
    labels = enc.labels.astype(np.int64)
    lse = np.logaddexp(logits[:, 0], logits[:, 1])
    loss = lse - logits[np.arange(len(labels)), labels]
    pred = (logits[:, 1] > logits[:, 0]).astype(np.int64)
    return {"loss": float(np.mean(loss)), "accuracy": float(np.mean(pred == labels)), "per_example_loss": loss,
            "logits": logits, "idx": enc.idx.copy(), "labels": labels}
