"""Phase 3 audit (GPU): evidence dictionary <-> trainable LoRA correspondence.

1. probe determinism: recompute direction atoms and evidence features with the stored head; compare with stored
2. real-model JVP of z along u_mj v_mj^T for the first two atoms of all 24 modules on 16 evidence rows vs stored J
   (checks projection, transposition, module order m*8+j)
3. PEFT orientation and scaling: a rank-r PEFT LoRA with A = V_r^T, B = U_r diag(s)/scaling reproduces the
   dictionary update sum_j s_j u_j v_j^T: finite-difference logit change vs J s (and vs. the swapped orientation)
4. deterministic LoRA init: zero initial update (logits bitwise equal to head-only), B = 0, shared rank prefixes
   across all 20 Gate-1 architectures, rsLoRA scaling alpha/sqrt(r), exact module set
5. tangent mismatch (diagnostic): at B = 0 the trainable update is (dB) A, whose row space is span(A_rows).
   For every Gate-1 module: ||P_A v_j||^2 for the selected prefix atoms, and the fraction of the
   direction-set mean-loss gradient captured by the LoRA tangent, ||G A^T (A A^T)^-1/2||_F^2 / ||G||_F^2,
   versus the best rank-r fraction sum_{j<=r} sigma_j^2 / ||G||_F^2 attained by the gradient-SVD prefix.
Read-only w.r.t. original artifacts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from rvm_lora import data as D  # noqa: E402
from rvm_lora.adapters import build_peft, lora_config, lora_layout  # noqa: E402
from rvm_lora.config import load_config  # noqa: E402
from rvm_lora.modeling import batches, configure_precision, evaluate, load_backbone, load_head  # noqa: E402
from rvm_lora.probe import ProjectionTaps, _forward_deltas, direction_pass, feature_pass  # noqa: E402

cfg = load_config(REPO / "configs/preregistered.yaml")
CK = REPO / "checkpoints/head_qnli_seed42.safetensors"
pdir = REPO / "artifacts/probes/qnli_s42"
configure_precision()
res = {}
model = load_backbone(42)
load_head(model, CK)
model.eval()
dirs_e = D.load_encoded(REPO / "data/qnli_directions.npz")
evid = D.load_encoded(REPO / "data/qnli_evidence.npz")
atoms = torch.load(pdir / "atoms.pt")
stored = np.load(pdir / "features.npz")

# 1. determinism
dirs = direction_pass(model, dirs_e, cfg.modules, cfg.n_dir, 1e-4)
feats = feature_pass(model, evid, cfg.modules, atoms["U"], atoms["V"], 1e-4)
res["probe_determinism"] = {
    "U_bitwise": bool(torch.equal(dirs["U"], atoms["U"])), "V_bitwise": bool(torch.equal(dirs["V"], atoms["V"])),
    "S_bitwise": bool(torch.equal(dirs["S"], atoms["S"])),
    "U_maxdiff": float((dirs["U"] - atoms["U"]).abs().max()),
    "J_bitwise": bool(np.array_equal(feats["J"], stored["J"])), "J_maxabs": float(np.abs(feats["J"] - stored["J"]).max()),
    "Phi_bitwise": bool(np.array_equal(feats["Phi"], stored["Phi"])), "t_bitwise": bool(np.array_equal(feats["t"], stored["t"]))}

# 2. JVP check on the real model with the real head
from torch.func import functional_call, jvp  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

n_rows = 16
ids = torch.as_tensor(evid.input_ids[:n_rows], device="cuda")
am = torch.as_tensor(evid.attention_mask[:n_rows], device="cuda")
params = dict(model.named_parameters())
worst = 0.0
per_mod = {}
for m, name in enumerate(cfg.modules):
    W = params[name + ".weight"]
    errs = []
    for j in range(2):
        Dm = torch.outer(atoms["U"][m, :, j], atoms["V"][m, :, j]).float().cuda()

        def z(s):
            out = functional_call(model, {**params, name + ".weight": W + s * Dm}, (),
                                  {"input_ids": ids, "attention_mask": am}).logits
            return out[:, 1] - out[:, 0]

        with sdpa_kernel(SDPBackend.MATH):
            _, tang = jvp(z, (torch.zeros((), device="cuda"),), (torch.ones((), device="cuda"),))
        ref = tang.double().cpu().numpy()
        col = stored["J"][:n_rows, m * cfg.n_dir + j]
        errs.append(float(np.max(np.abs(col - ref)) / max(np.abs(ref).max(), 1e-12)))
    per_mod[name] = errs
    worst = max(worst, max(errs))
res["jvp_vs_stored_J"] = {"max_rel_err_colscale": worst, "per_module": per_mod, "rows": n_rows}

# 3. PEFT orientation/scaling: LoRA with A = V_r^T, B = U_r diag(s) / scaling
probe_rows = D.Encoded(evid.idx[:64], evid.input_ids[:64], evid.attention_mask[:64], evid.labels[:64])
z0 = evaluate(model, probe_rows)["logits"]
z0 = z0[:, 1] - z0[:, 0]
orient = []
for name, r in [(cfg.modules[7], 2), (cfg.modules[10], 2), (cfg.modules[23], 3)]:
    m = cfg.modules.index(name)
    for swap in (False, True):
        base = load_backbone(42)
        load_head(base, CK)
        pm = build_peft(base, {name: r}, cfg, run_seed=42)
        lay = lora_layout(pm)[name]
        s = np.array([1e-3, -2e-3, 1.5e-3][:r])
        U = atoms["U"][m, :, :r].double()
        V = atoms["V"][m, :, :r].double()
        if swap:
            U, V = V, U  # wrong orientation (input/output swapped) as a negative control
        for mod_name, mod in pm.named_modules():
            if mod_name.endswith(name) and hasattr(mod, "lora_A"):
                with torch.no_grad():
                    mod.lora_A["default"].weight.copy_(V.T.float())
                    mod.lora_B["default"].weight.copy_((U * torch.as_tensor(s)[None, :] / lay["scaling"]).float())
        zl = evaluate(pm, probe_rows)["logits"]
        zl = zl[:, 1] - zl[:, 0]
        pred = stored["J"][:64, m * cfg.n_dir:m * cfg.n_dir + r] @ s
        rel = float(np.max(np.abs((zl - z0) - pred)) / max(np.abs(pred).max(), 1e-12))
        orient.append({"module": name, "rank": r, "swapped_orientation": swap, "scaling": lay["scaling"],
                       "expected_scaling": 4 / np.sqrt(r), "rel_err_dz_vs_Js": rel})
        del pm, base
res["peft_orientation"] = orient

# 4. deterministic init over the 20 Gate-1 architectures
init = {"zero_update_bitwise": [], "B_zero": [], "scalings_ok": [], "module_sets_ok": []}
prefix = {}
for a in cfg.architectures():
    base = load_backbone(42)
    load_head(base, CK)
    pm = build_peft(base, a.ranks, cfg, run_seed=42)
    lg = evaluate(pm, probe_rows)["logits"]
    init["zero_update_bitwise"].append(bool(np.array_equal(lg[:, 1] - lg[:, 0], z0)))
    lay = lora_layout(pm)
    init["module_sets_ok"].append(set(lay) == set(a.ranks) and all(lay[k]["rank"] == v for k, v in a.ranks.items()))
    init["scalings_ok"].append(all(abs(lay[k]["scaling"] - 4 / np.sqrt(v)) < 1e-12 for k, v in a.ranks.items()))
    for mn, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            key = mn.split("base_model.model.")[-1]
            A = mod.lora_A["default"].weight.detach().cpu()
            init["B_zero"].append(float(mod.lora_B["default"].weight.abs().max()) == 0.0)
            prefix.setdefault(key, []).append(A)
    del pm, base
pref_ok = all(all(torch.equal(x, xs[0][:x.shape[0]]) or torch.equal(xs[0], x[:xs[0].shape[0]]) for x in xs)
              for xs in prefix.values())
res["deterministic_init"] = {k: all(v) for k, v in init.items()}
res["deterministic_init"]["shared_rank_prefixes"] = pref_ok
res["deterministic_init"]["n_arch"] = 20

# 5. tangent mismatch diagnostic
mods = dict(model.named_modules())
G = {n: torch.zeros(mods[n].weight.shape, dtype=torch.float64, device="cuda") for n in cfg.modules}
N = len(dirs_e)
with ProjectionTaps(model, cfg.modules) as taps:
    for bt in batches(dirs_e, np.arange(N), 32, "cuda"):
        zz, deltas = _forward_deltas(model, taps, bt)
        p = torch.sigmoid(zz).clamp(1e-4, 1 - 1e-4)
        c = (p - bt["labels"].to(p.dtype)) / N
        for n in cfg.modules:
            G[n] += torch.einsum("bto,bti->oi", deltas[n] * c[:, None, None], taps.inputs[n]).double()
tang = []
seen = set()
for a in cfg.architectures():
    base = load_backbone(42)
    pm = build_peft(base, a.ranks, cfg, run_seed=42)
    for mn, mod in pm.named_modules():
        if not (hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {})):
            continue
        key = mn.split("base_model.model.")[-1]
        r = mod.lora_A["default"].weight.shape[0]
        if (key, r) in seen:
            continue
        seen.add((key, r))
        A = mod.lora_A["default"].weight.detach().double().cuda()       # r x in
        Q, _ = torch.linalg.qr(A.T)                                     # in x r, orthonormal row space of A
        m = cfg.modules.index(key)
        V = atoms["V"][m, :, :r].double().cuda()
        Gm = G[key]
        fro2 = float((Gm ** 2).sum())
        sv = torch.linalg.svdvals(Gm)
        tang.append({"module": key, "rank": r,
                     "atom_energy_in_lora_rowspace": [float((Q.T @ V[:, j]).norm() ** 2) for j in range(r)],
                     "random_expectation": r / 768,
                     "grad_frac_lora_tangent": float(((Gm @ Q) ** 2).sum()) / fro2,
                     "grad_frac_svd_prefix": float((sv[:r] ** 2).sum()) / fro2})
    del pm, base
res["tangent_mismatch"] = tang
res["tangent_mismatch_summary"] = {
    "mean_atom_energy_in_lora_rowspace": float(np.mean([np.mean(x["atom_energy_in_lora_rowspace"]) for x in tang])),
    "mean_grad_frac_lora_tangent": float(np.mean([x["grad_frac_lora_tangent"] for x in tang])),
    "mean_grad_frac_svd_prefix": float(np.mean([x["grad_frac_svd_prefix"] for x in tang]))}
(OUT / "phase3_lora_map.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: v for k, v in res.items() if k not in ("tangent_mismatch",)}, indent=1, default=str))
for x in tang[:12]:
    print(x["module"].split(".")[3], x["module"][-5:], x["rank"], [round(e, 4) for e in x["atom_energy_in_lora_rowspace"]],
          round(x["grad_frac_lora_tangent"], 4), round(x["grad_frac_svd_prefix"], 4))
