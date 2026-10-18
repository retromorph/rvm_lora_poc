"""Phase 4b: scope of the local-optimum defect in rvm_lora.evidence.optimize_evidence.

(1) validate the dense optima found for the 3 affected Gate-1 architectures with the direct n x n marginal
    likelihood (independent of the Gram/Cholesky form) and report conditioning;
(2) profile E along the segment between the recorded and the dense optimum (shows two separate basins);
(3) for every candidate group evaluated at every step of the stored real-data RVM greedy search: repo value vs
    dense multistart (does the greedy choice change?);
(4) null calibration: for the first 20 seed-1931 null responses, repo initial gains vs dense multistart.
Read-only w.r.t. original artifacts.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from rvm_lora.config import load_config  # noqa: E402
from rvm_lora.evidence import evidence_and_grad, evidence_direct  # noqa: E402
from rvm_lora.selectors import (  # noqa: E402
    Dictionary, RVMObjective, apply, candidate_groups, feasible, initial_gains, null_responses,
)

cfg = load_config(REPO / "configs/preregistered.yaml")
z = np.load(REPO / "artifacts/probes/qnli_s42/features.npz")
Phi, t = z["Phi"].astype(np.float64), z["t"].astype(np.float64)
K, b = Phi.T @ Phi, Phi.T @ t
D = Dictionary(K, b, cfg.modules, cfg.n_dir, cfg.rank_unit_params)
lo, hi = cfg.log_alpha_bounds
groups = candidate_groups(cfg.modules)
p4 = json.loads((OUT.parent / "phase4" / "phase4_evidence.json").read_text())
rng = np.random.default_rng(7)
res = {}


def dense_max(Ks, bs, colmod, nm, n_random=120):
    def f(x):
        E, g = evidence_and_grad(Ks, bs, x, colmod, nm, True)
        return -E, -g
    grid1 = np.array([lo, -15, -10, -5, -2, 0, 2, 5, 10, 15, hi])
    starts = [np.full(nm, g) for g in grid1]
    if nm <= 2:
        starts += [np.array(p) for p in np.array(np.meshgrid(*[grid1] * nm)).reshape(nm, -1).T]
    starts += [rng.uniform(lo, hi, nm) for _ in range(n_random)]
    best = -np.inf
    xb = None
    for x0 in starts:
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * nm,
                     options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        if -r.fun > best:
            best, xb = -float(r.fun), r.x.copy()
    return best, xb


# (1)+(2)
val = []
for row in p4["gate1_arch_evidence"]:
    if row["dense_minus_recorded"] < 1e-6:
        continue
    a = next(x for x in cfg.architectures() if x.arch_id == row["arch_id"])
    ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
    cols, colmod, active = D.columns(ranks)
    xr, xd = np.asarray(row["log_alpha_recorded"]), np.asarray(row["log_alpha_dense"])
    Ed = evidence_direct(Phi[:, cols], t, np.exp(xd[colmod]))
    Er = evidence_direct(Phi[:, cols], t, np.exp(xr[colmod]))
    C = np.eye(len(t)) + (Phi[:, cols] / np.exp(xd[colmod])[None, :]) @ Phi[:, cols].T
    prof = []
    for s in np.linspace(0, 1, 21):
        x = (1 - s) * xr + s * xd
        prof.append(float(evidence_direct(Phi[:, cols], t, np.exp(x[colmod]))))
    colnorm = {cfg.modules[m]: np.linalg.norm(Phi[:, m * 8:m * 8 + ranks[m]], axis=0).tolist() for m in active}
    val.append({"arch": row["arch_id"], "E_direct_at_recorded": Er, "E_direct_at_dense": Ed,
                "cond_C_at_dense": float(np.linalg.cond(C)), "profile_recorded_to_dense": prof,
                "column_norms": colnorm, "log_alpha_recorded": xr.tolist(), "log_alpha_dense": xd.tolist(),
                "q2_over_h_per_column": {cfg.modules[m]: [float(b[c] ** 2 / K[c, c]) for c in range(m * 8, m * 8 + ranks[m])]
                                         for m in active}})
res["affected_arch_validation"] = val

# (3) every candidate at every real-data greedy step
sel = json.loads((REPO / "artifacts/probes/qnli_s42/selection.json").read_text())
obj = RVMObjective(D, cfg.log_alpha_bounds)
steps = []
state = tuple([0] * 24)
cur = 0.0
for s in sel["group_rvm"]["trace"]:
    rows = []
    for g in groups:
        if not feasible(state, g, cfg.max_rank, 24):
            continue
        new = apply(state, g)
        v_repo, _ = obj.value(new, parent=state)
        Ks, bs, colmod, active = D.sub(new)
        v_dense, _ = dense_max(Ks, bs, colmod, len(active), n_random=40)
        rows.append({"group": g.key, "repo": v_repo, "dense": v_dense, "units": g.units})
    best_repo = max(rows, key=lambda r: (r["repo"] - cur) / r["units"])
    best_dense = max(rows, key=lambda r: (r["dense"] - cur) / r["units"])
    steps.append({"iter": s["iter"], "n_candidates": len(rows),
                  "n_repo_below_dense_1e-6": sum(r["dense"] - r["repo"] > 1e-6 for r in rows),
                  "max_underestimate": max(r["dense"] - r["repo"] for r in rows),
                  "best_group_repo": best_repo["group"], "best_group_dense": best_dense["group"],
                  "best_gain_pp_repo": (best_repo["repo"] - cur) / (best_repo["units"] * 1536),
                  "best_gain_pp_dense": (best_dense["dense"] - cur) / (best_dense["units"] * 1536),
                  "worst": sorted(rows, key=lambda r: r["repo"] - r["dense"])[:3]})
    if s.get("ranks_after") is None:
        break
    state = tuple(s["ranks_after"])
    cur = s["objective_after_sweep"]
res["greedy_candidates_vs_dense"] = steps

# (4) null calibration initial gains on 20 null responses
T = null_responses(Phi.shape[0], 100, cfg["probe"]["null_calibration_seed"])
nulls = []
for i in range(20):
    Dn = D.with_b(Phi.T @ T[i])
    o = RVMObjective(Dn, cfg.log_alpha_bounds)
    g_repo = initial_gains(o, groups, cfg.max_rank, 24)
    dens = []
    for g in groups:
        new = apply(tuple([0] * 24), g)
        Ks, bs, colmod, active = Dn.sub(new)
        if Ks.shape[0] == 1:
            dens.append(o.value(new)[0] / (g.units * 1536))
            continue
        v, _ = dense_max(Ks, bs, colmod, len(active), n_random=20)
        dens.append(v / (g.units * 1536))
    dens = np.asarray(dens)
    nulls.append({"i": i, "max_repo": float(g_repo.max()), "max_dense": float(dens.max()),
                  "n_groups_underestimated": int(np.sum(dens - g_repo > 1e-12))})
res["null_initial_gains_vs_dense"] = nulls
(OUT / "phase4b_localopt.json").write_text(json.dumps(res, indent=2, default=str))
for v in val:
    print(v["arch"], "Er", v["E_direct_at_recorded"], "Ed", v["E_direct_at_dense"], "cond", v["cond_C_at_dense"])
    print("  profile", [round(x, 3) for x in v["profile_recorded_to_dense"]])
    print("  colnorms", {k.split(".")[3] + k[-1]: [round(c, 2) for c in vv] for k, vv in v["column_norms"].items()})
    print("  q2/h", {k.split(".")[3] + k[-1]: [round(c, 2) for c in vv] for k, vv in v["q2_over_h_per_column"].items()})
for s in steps:
    print({k: s[k] for k in s if k != "worst"})
for n in nulls:
    print(n)
