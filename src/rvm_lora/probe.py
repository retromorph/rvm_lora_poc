"""Hooked probes on the frozen backbone (METHODS.md §2).

* `ProjectionTaps` records each eligible projection's input h and (graph-connected) output o.
  The embedding output is made to require grad so that every projection output is a non-leaf
  node of the graph even though all weights are frozen; delta_m = d z / d o_m for the binary
  logit difference z = l1 - l0 is then obtained with `torch.autograd.grad`. Because examples in a
  batch are independent in eval mode, grad of sum_i z_i gives every per-example delta_i.
* Direction generation: G_m = (1/N) sum_i (p_i - y_i) sum_t delta_i[t] h_i[t]^T (mean-loss
  gradient of W_m, out x in), then a rank-8 SVD with fixed signs.
* Features: J_ij = sum_t (delta_i[t] . u_mj)(v_mj . h_i[t]); Phi = sqrt(w) J, t = (y - p)/sqrt(w).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import Encoded
from .modeling import batches


class ProjectionTaps:
    def __init__(self, model, module_names: list[str]):
        self.model = model
        self.names = list(module_names)
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        mods = dict(model.named_modules())
        missing = [n for n in self.names if n not in mods]
        if missing:
            raise KeyError(f"modules not found: {missing}")
        emb = _embeddings(model)
        self.handles.append(emb.register_forward_hook(self._emb_hook))
        for n in self.names:
            self.handles.append(mods[n].register_forward_hook(self._make_hook(n)))

    @staticmethod
    def _emb_hook(module, inp, out):
        if isinstance(out, torch.Tensor) and not out.requires_grad:
            return out.detach().requires_grad_(True)
        return out

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.inputs[name] = inp[0].detach()
            self.outputs[name] = out
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _embeddings(model):
    base = model
    for attr in ("base_model", "model"):
        if hasattr(base, "roberta"):
            break
        base = getattr(base, attr, base)
    return base.roberta.embeddings


def _forward_deltas(model, taps: ProjectionTaps, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    taps.inputs.clear()
    taps.outputs.clear()
    with torch.enable_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits = logits if logits.dtype == torch.float64 else logits.float()
        z = logits[:, 1] - logits[:, 0]
        outs = [taps.outputs[n] for n in taps.names]
        grads = torch.autograd.grad(z.sum(), outs, allow_unused=False)
    deltas = {n: g.detach() for n, g in zip(taps.names, grads)}
    for n, d in deltas.items():
        if not torch.isfinite(d).all():
            raise FloatingPointError(f"non-finite delta for {n}")
    return z.detach(), deltas


def fix_svd_signs(U: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make the largest-|.| entry (first on ties) of each u column positive; flip the matching v."""
    piv = torch.argmax(U.abs(), dim=0)
    s = torch.sign(U[piv, torch.arange(U.shape[1])])
    s[s == 0] = 1
    return U * s, V * s


def direction_pass(model, enc: Encoded, module_names: list[str], n_dir: int, clip: float,
                   device="cuda", batch_size: int = 32) -> dict:
    """Mean-loss full-projection gradients on the direction set and their rank-n_dir SVD atoms."""
    model.eval()
    t0 = time.perf_counter()
    N = len(enc)
    mods = dict(model.named_modules())
    G = {n: torch.zeros(mods[n].weight.shape, dtype=torch.float64, device=device) for n in module_names}
    with ProjectionTaps(model, module_names) as taps:
        for batch in batches(enc, np.arange(N), batch_size, device):
            z, deltas = _forward_deltas(model, taps, batch)
            p = torch.sigmoid(z).clamp(clip, 1 - clip)  # clipped p for directions and features
            c = (p - batch["labels"].to(p.dtype)) / N
            for n in module_names:
                d = deltas[n]
                g = torch.einsum("bto,bti->oi", d * c.to(d.dtype)[:, None, None], taps.inputs[n].to(d.dtype))
                G[n] += g.double()
    grad_wall = _sync_time(t0, device)
    atoms_u, atoms_v, svals, zero_grad = [], [], [], []
    for n in module_names:
        U, S, Vh = torch.linalg.svd(G[n], full_matrices=False)
        u, v = fix_svd_signs(U[:, :n_dir], Vh[:n_dir].T)
        atoms_u.append(u.cpu())
        atoms_v.append(v.cpu())
        svals.append(S[:n_dir].cpu())
        if float(S[0]) == 0.0:
            zero_grad.append(n)
    if zero_grad:
        raise FloatingPointError(f"all-zero direction gradient for modules {zero_grad}")
    wall = _sync_time(t0, device)
    return {"U": torch.stack(atoms_u), "V": torch.stack(atoms_v), "S": torch.stack(svals),
            "grad_fro": torch.tensor([float(G[n].norm()) for n in module_names]),
            "n_examples": N, "passes": 1, "grad_wall_s": grad_wall, "wall_s": wall}


def feature_pass(model, enc: Encoded, module_names: list[str], U: torch.Tensor, V: torch.Tensor, clip: float,
                 device="cuda", batch_size: int = 32) -> dict:
    """Per-example logit-difference Jacobian rows J (n x M*n_dir) and the Gauss-Newton surrogate."""
    model.eval()
    t0 = time.perf_counter()
    M, _, n_dir = U.shape
    dt = next(model.parameters()).dtype
    Uf = U.to(device=device, dtype=dt)
    Vf = V.to(device=device, dtype=dt)
    J_rows, z_all = [], []
    with ProjectionTaps(model, module_names) as taps:
        for batch in batches(enc, np.arange(len(enc)), batch_size, device):
            z, deltas = _forward_deltas(model, taps, batch)
            cols = []
            for m, n in enumerate(module_names):
                du = deltas[n] @ Uf[m]                       # (B, T, n_dir)
                hv = taps.inputs[n].to(dt) @ Vf[m]           # (B, T, n_dir)
                cols.append((du * hv).sum(1))
            J_rows.append(torch.cat(cols, dim=1).cpu())
            z_all.append(z.cpu())
    J = torch.cat(J_rows).double().numpy()
    z = torch.cat(z_all).double().numpy()
    y = enc.labels.astype(np.float64)
    p = np.clip(1.0 / (1.0 + np.exp(-z)), clip, 1 - clip)
    w = p * (1 - p)
    Phi = np.sqrt(w)[:, None] * J
    t = (y - p) / np.sqrt(w)
    wall = _sync_time(t0, device)
    zero_cols = np.where(np.all(J == 0, axis=0))[0].tolist()
    if zero_cols:
        raise FloatingPointError(f"all-zero Jacobian columns {zero_cols[:10]}")
    return {"J": J, "z": z, "p": p, "w": w, "Phi": Phi, "t": t, "K": Phi.T @ Phi, "b": Phi.T @ t,
            "n_examples": len(enc), "passes": 1, "wall_s": wall}


def oplora_importance_pass(model, enc: Encoded, module_names: list[str], device="cuda", batch_size: int = 16) -> dict:
    """DPLoRA stage-1 importance I_m = (1/N) sum_n mean_ij (dL_n/dW_m)_ij^2 (per-example CE gradients).

    dL_n/dW_m = (p_n - y_n) delta_n^T h_n, and ||delta^T h||_F^2 = sum((delta delta^T) * (h h^T)).
    Unclipped p as in the reference (HF cross-entropy). Eval mode (dropout off), as in the reference.
    """
    model.eval()
    t0 = time.perf_counter()
    N = len(enc)
    mods = dict(model.named_modules())
    acc = {n: 0.0 for n in module_names}
    with ProjectionTaps(model, module_names) as taps:
        for batch in batches(enc, np.arange(N), batch_size, device):
            z, deltas = _forward_deltas(model, taps, batch)
            c2 = (torch.sigmoid(z) - batch["labels"].to(z.dtype)) ** 2
            for n in module_names:
                d = deltas[n].double()
                h = taps.inputs[n].double()
                fro2 = ((d @ d.transpose(1, 2)) * (h @ h.transpose(1, 2))).sum((1, 2))
                acc[n] += float((c2.double() * fro2).sum()) / mods[n].weight.numel()
    imp = {n: acc[n] / N for n in module_names}
    return {"importance": imp, "n_examples": N, "passes": 1, "wall_s": _sync_time(t0, device)}


def _sync_time(t0: float, device) -> float:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - t0
