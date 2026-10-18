"""Verify the repaired optimize_evidence against an independent dense multistart search.

argv: <code_root with patched src> <repo with original artifacts> <out dir>
(1) 20 Gate-1 architectures; (2) every candidate at every step of the repaired real-data greedy search;
(3) all 100 seed-1931 null responses x all multi-column candidate groups (initial gains);
(4) 300 synthetic scale-disparate problems. Old (C3-only) optimiser values are reported alongside.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CODE, REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE / "src"))
import rvm_lora.evidence as ev  # noqa: E402
from rvm_lora.config import load_config  # noqa: E402
from rvm_lora.selectors import (  # noqa: E402
    Dictionary, RVMObjective, apply, candidate_groups, feasible, greedy_select, null_responses,
)

assert "GLOBAL_IMPROVEMENT_TOL" in dir(ev), "patched evidence module not loaded"
cfg = load_config(CODE / "configs/preregistered.yaml")
z = np.load(REPO / "artifacts/probes/qnli_s42/features.npz")
Phi, t = z["Phi"].astype(np.float64), z["t"].astype(np.float64)
D = Dictionary(Phi.T @ Phi, Phi.T @ t, cfg.modules, cfg.n_dir, cfg.rank_unit_params)
lo, hi = cfg.log_alpha_bounds
groups = candidate_groups(cfg.modules)
rng = np.random.default_rng(99)
res = {}


def dense_max(K, b, colmod, nm, n_random=100):
    def f(x):
        E, g = ev.evidence_and_grad(K, b, x, colmod, nm, True)
        return -E, -g
    grid1 = np.array([lo, -18, -14, -10, -6, -3, 0, 3, 6, 10, 14, 18, hi])
    starts = [np.full(nm, g) for g in grid1]
    if nm <= 2:
        starts += [np.array(p) for p in np.array(np.meshgrid(*[grid1] * nm)).reshape(nm, -1).T]
    starts += [rng.uniform(lo, hi, nm) for _ in range(n_random)]
    best = -np.inf
    for x0 in starts:
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * nm,
                     options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        best = max(best, -float(r.fun))
    return best


def old_opt(K, b, colmod, nm):
    def f(x):
        E, g = ev.evidence_and_grad(K, b, x, colmod, nm, True)
        return -E, -g
    best = -np.inf
    for x0 in (np.zeros(nm), np.full(nm, np.clip(np.log(max(np.median(np.diag(K)), 1e-300)), lo, hi))):
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * nm,
                     options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
        best = max(best, -float(r.fun))
    return best


def cmp(K, b, colmod, nm, n_random=100):
    t0 = time.perf_counter()
    new = ev.optimize_evidence(K, b, colmod, nm, bounds=(lo, hi)).evidence
    wall = time.perf_counter() - t0
    return {"new": new, "old": old_opt(K, b, colmod, nm), "dense": dense_max(K, b, colmod, nm, n_random), "wall": wall}


# (1)
arch = []
for a in cfg.architectures():
    ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
    K, b, colmod, active = D.sub(ranks)
    c = cmp(K, b, colmod, len(active), 300)
    arch.append({"arch": a.arch_id, **c})
res["arch"] = arch
res["arch_summary"] = {"max_dense_minus_new": max(r["dense"] - r["new"] for r in arch),
                       "n_new_below_dense_1e-7": sum(r["dense"] - r["new"] > 1e-7 for r in arch),
                       "n_old_below_dense_1e-6": sum(r["dense"] - r["old"] > 1e-6 for r in arch),
                       "max_wall_s": max(r["wall"] for r in arch)}
print("arch", res["arch_summary"], flush=True)

# (2) repaired greedy on real data; every candidate at every step vs dense
thr = json.loads((REPO / "artifacts/probes/qnli_s42/selection.json").read_text())["group_rvm"]["threshold"]
obj = RVMObjective(D, cfg.log_alpha_bounds)
t0 = time.perf_counter()
sel = greedy_select(obj, groups, cfg.max_rank, 24, thr)
res["greedy_wall_s_sequential"] = time.perf_counter() - t0
res["greedy_ranks_at_original_threshold"] = sel.rank_dict(cfg.modules)
steps = []
state = tuple([0] * 24)
cur = 0.0
for s in sel.trace:
    worst = 0.0
    nbad = 0
    for g in groups:
        if not feasible(state, g, cfg.max_rank, 24):
            continue
        new = apply(state, g)
        v = obj.value(new, parent=state)[0]
        K, b, colmod, active = D.sub(new)
        if K.shape[0] == 1:
            continue
        d = dense_max(K, b, colmod, len(active), 30)
        worst = max(worst, d - v)
        nbad += d - v > 1e-7
    steps.append({"iter": s["iter"], "group": s["best_group"], "gain_pp": s["gain_per_param"], "action": s["action"],
                  "max_dense_minus_repo": worst, "n_candidates_below_dense_1e-7": nbad})
    if s["action"] != "add":
        break
    state = tuple(s["ranks_after"])
    cur = s["objective_after_sweep"]
res["greedy_steps"] = steps
print("greedy", steps, flush=True)

# (3) null calibration groups
T = null_responses(Phi.shape[0], 100, cfg["probe"]["null_calibration_seed"])
worst = 0.0
nbad = nold = ntot = 0
t0 = time.perf_counter()
for i in range(100):
    Dn = D.with_b(Phi.T @ T[i])
    for g in groups:
        new = apply(tuple([0] * 24), g)
        K, b, colmod, active = Dn.sub(new)
        if K.shape[0] == 1:
            continue
        ntot += 1
        newv = ev.optimize_evidence(K, b, colmod, len(active), bounds=(lo, hi)).evidence
        d = dense_max(K, b, colmod, len(active), 10)
        o = old_opt(K, b, colmod, len(active))
        worst = max(worst, d - newv)
        nbad += d - newv > 1e-7
        nold += d - o > 1e-6
res["null_groups"] = {"n_problems": ntot, "max_dense_minus_new": worst, "n_new_below_dense_1e-7": nbad,
                      "n_old_below_dense_1e-6": nold, "wall_s": time.perf_counter() - t0}
print("nulls", res["null_groups"], flush=True)

# (4) synthetic scale-disparate problems (2-4 modules, 1-4 columns each, column scales spanning 1e-2..1e2)
syn = []
for k in range(300):
    nm = int(rng.integers(2, 5))
    sizes = rng.integers(1, 5, nm)
    colmod = np.repeat(np.arange(nm), sizes)
    p = len(colmod)
    n = 128
    X = rng.standard_normal((n, p)) @ (np.eye(p) + 0.5 * rng.standard_normal((p, p)))
    X *= 10.0 ** rng.uniform(-2, 2, p)[None, :]
    w = rng.standard_normal(p) * (rng.random(p) < 0.6) / np.linalg.norm(X, axis=0)
    y = X @ w * rng.uniform(0, 6) + rng.standard_normal(n)
    K, b = X.T @ X, X.T @ y
    c = cmp(K, b, colmod, nm, 60)
    syn.append({"k": k, "nm": nm, **c})
res["synthetic_summary"] = {"n": len(syn), "max_dense_minus_new": max(r["dense"] - r["new"] for r in syn),
                            "n_new_below_dense_1e-7": sum(r["dense"] - r["new"] > 1e-7 for r in syn),
                            "n_old_below_dense_1e-6": sum(r["dense"] - r["old"] > 1e-6 for r in syn),
                            "n_new_above_dense_1e-7": sum(r["new"] - r["dense"] > 1e-7 for r in syn)}
print("synthetic", res["synthetic_summary"], flush=True)
(OUT / "verify_fix.json").write_text(json.dumps(res, indent=2, default=str))
for r in arch:
    print(f"{r['arch']:14s} new={r['new']:+.9f} old={r['old']:+.9f} dense={r['dense']:+.9f} wall={r['wall']:.2f}s")
print(json.dumps(res["greedy_ranks_at_original_threshold"]))
