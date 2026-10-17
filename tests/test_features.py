"""Gate 0 item 4: logit-difference Jacobian of rank-one q/v updates vs JVP and finite differences,
direction-gradient formula vs autograd, OPLoRA importance vs explicit per-example gradients."""

import numpy as np
import torch

from rvm_lora.data import Encoded
from rvm_lora.probe import direction_pass, feature_pass, fix_svd_signs, oplora_importance_pass


def _tiny(seed=0):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(seed)
    c = RobertaConfig(vocab_size=60, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=40, num_labels=2, type_vocab_size=1,
                      pad_token_id=1, attn_implementation="eager", initializer_range=0.5,
                      hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    m = RobertaForSequenceClassification(c).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _enc(n=6, T=10, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(3, 60, size=(n, T))
    am = np.ones((n, T), dtype=np.int64)
    for i in range(n):  # variable lengths with padding
        L = rng.integers(4, T + 1)
        ids[i, L:] = 1
        am[i, L:] = 0
    ids[:, 0] = 0
    return Encoded(np.arange(n), ids.astype(np.int64), am, rng.integers(0, 2, n).astype(np.int64))


MODS = [f"roberta.encoder.layer.{l}.attention.self.{p}" for l in range(2) for p in ("query", "value")]


def _z(model, enc, overrides=None):
    from torch.func import functional_call

    params = dict(model.named_parameters())
    if overrides:
        params = {**params, **overrides}
    out = functional_call(model, params, (), {"input_ids": torch.as_tensor(enc.input_ids),
                                             "attention_mask": torch.as_tensor(enc.attention_mask)})
    return out.logits[:, 1] - out.logits[:, 0]


def test_jacobian_matches_jvp_and_fd(record):
    from torch.func import jvp

    model = _tiny()
    enc = _enc()
    dirs = direction_pass(model, enc, MODS, n_dir=3, clip=1e-4, device="cpu", batch_size=4)
    feats = feature_pass(model, enc, MODS, dirs["U"], dirs["V"], clip=1e-4, device="cpu", batch_size=4)
    J = feats["J"]
    worst_jvp = worst_fd = 0.0
    n_cols = big = 0
    for m, name in enumerate(MODS):
        W = dict(model.named_parameters())[name + ".weight"]
        for j in range(3):
            D = torch.outer(dirs["U"][m, :, j], dirs["V"][m, :, j]).to(W.dtype)
            _, tang = jvp(lambda s: _z(model, enc, {name + ".weight": W + s * D}),
                          (torch.zeros((), dtype=W.dtype),), (torch.ones((), dtype=W.dtype),))
            eps = 1e-5
            fd = (_z(model, enc, {name + ".weight": W + eps * D}) - _z(model, enc, {name + ".weight": W - eps * D})) / (
                2 * eps)
            col = J[:, m * 3 + j]
            scale = max(float(np.abs(tang.numpy()).max()), 1e-12)  # column scale ("away from zero")
            r1 = float(np.max(np.abs(col - tang.numpy())) / scale)
            r2 = float(np.max(np.abs(col - fd.numpy())) / scale)
            worst_jvp, worst_fd = max(worst_jvp, r1), max(worst_fd, r2)
            n_cols += 1
            big += scale > 1e-3
    assert worst_jvp < 1e-3 and worst_fd < 1e-3
    assert big >= n_cols // 2, "Jacobian columns too small for a meaningful check"
    assert np.all(np.abs(J).sum(0) > 0), "zero Jacobian column (delta not propagated through frozen weights)"
    record(max_rel_err_vs_jvp=worst_jvp, max_rel_err_vs_central_fd=worst_fd, threshold=1e-3, columns=n_cols)


def test_direction_gradient_matches_autograd(record):
    model = _tiny(1)
    enc = _enc(seed=1)
    dirs = direction_pass(model, enc, MODS, n_dir=3, clip=1e-4, device="cpu", batch_size=3)
    # reference: autograd of the mean BCE w.r.t. the full weights (clipping inactive for this tiny model)
    Ws = [dict(model.named_parameters())[n + ".weight"] for n in MODS]
    for W in Ws:
        W.requires_grad_(True)
    z = _z(model, enc)
    p = torch.sigmoid(z)
    assert float(p.min()) > 1e-4 and float(p.max()) < 1 - 1e-4
    y = torch.as_tensor(enc.labels, dtype=z.dtype)
    loss = torch.nn.functional.binary_cross_entropy(p, y)
    grads = torch.autograd.grad(loss, Ws)
    worst = 0.0
    for m, g in enumerate(grads):
        U, S, Vh = torch.linalg.svd(g.double())
        u, v = fix_svd_signs(U[:, :3], Vh[:3].T)
        assert torch.allclose(S[:3], dirs["S"][m], rtol=1e-6, atol=1e-10)
        assert torch.allclose(u, dirs["U"][m], atol=1e-6) and torch.allclose(v, dirs["V"][m], atol=1e-6)
        worst = max(worst, float((S[:3] - dirs["S"][m]).abs().max()))
    for W in Ws:
        W.requires_grad_(False)
    # sign convention and unit norms
    piv = dirs["U"].abs().argmax(dim=1)
    assert bool((torch.gather(dirs["U"], 1, piv[:, None, :]) > 0).all())
    assert torch.allclose(dirs["U"].norm(dim=1), torch.ones(4, 3, dtype=dirs["U"].dtype))
    record(max_singular_value_err=worst)


def test_oplora_importance_matches_per_example_gradients():
    model = _tiny(2)
    enc = _enc(seed=2)
    imp = oplora_importance_pass(model, enc, MODS, device="cpu", batch_size=4)["importance"]
    Ws = {n: dict(model.named_parameters())[n + ".weight"] for n in MODS}
    acc = {n: 0.0 for n in MODS}
    for i in range(len(enc)):
        sub = Encoded(enc.idx[i:i + 1], enc.input_ids[i:i + 1], enc.attention_mask[i:i + 1], enc.labels[i:i + 1])
        for W in Ws.values():
            W.requires_grad_(True)
        z = _z(model, sub)
        logits = torch.stack([torch.zeros_like(z), z], 1)
        loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(sub.labels))
        gs = torch.autograd.grad(loss, list(Ws.values()))
        for n, g in zip(Ws, gs):
            acc[n] += float((g ** 2).mean())
        for W in Ws.values():
            W.requires_grad_(False)
    for n in MODS:
        assert np.isclose(imp[n], acc[n] / len(enc), rtol=1e-8)
