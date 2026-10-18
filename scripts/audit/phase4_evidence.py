"""Phase 4 audit (CPU): RVM evidence, precision optimisation, Newton/Fisher scores, ridge CV, null calibration,
selection reproduction. Read-only w.r.t. the original artifacts; writes to the output directory only.

(a) every Gate-1 architecture: recorded evidence vs. (i) the direct n x n Gaussian marginal likelihood at the
    recorded precisions and (ii) a dense multi-start search of the same bounded objective (global-optimum
    check of the L-BFGS-B / 3-start implementation)
(b) independent re-implementation of gradient-norm, independent-Fisher and conditional-Newton scores
(c) ridge-lambda CV recomputed independently
(d) selection job re-run on the stored features into a scratch directory; diff against stored selection.json
(e) every state in the stored RVM trace: recorded objective vs dense multi-start
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from rvm_lora.config import load_config  # noqa: E402
from rvm_lora.evidence import evidence_and_grad, evidence_direct  # noqa: E402
from rvm_lora.selectors import Dictionary  # noqa: E402

cfg = load_config(REPO / "configs/preregistered.yaml")
pdir = REPO / "artifacts/probes/qnli_s42"
z = np.load(pdir / "features.npz")
Phi, t = z["Phi"].astype(np.float64), z["t"].astype(np.float64)
K, b = Phi.T @ Phi, Phi.T @ t
D = Dictionary(K, b, cfg.modules, cfg.n_dir, cfg.rank_unit_params)
lo, hi = cfg.log_alpha_bounds
scores = {s["arch_id"]: s for s in json.loads((REPO / "artifacts/gate1/precomputed_scores.json").read_text())["arch_scores"]}
lam = json.loads((REPO / "artifacts/gate1/precomputed_scores.json").read_text())["lambda"]
sel = json.loads((pdir / "selection.json").read_text())
res = {"label": "Phase-4 evidence/selector audit (read-only)", "lambda": lam}
rng = np.random.default_rng(12345)


def dense_max(Ks, bs, colmod, nm, n_random=300):
    """Best bounded evidence over a grid + random starts (L-BFGS-B from each start)."""
    def f(x):
        E, g = evidence_and_grad(Ks, bs, x, colmod, nm, True)
        return -E, -g
    grid1 = np.array([lo, -15, -10, -5, -2, 0, 2, 5, 10, 15, hi])
    starts = []
    if nm <= 2:
        starts += [np.array(p) for p in np.array(np.meshgrid(*[grid1] * nm)).reshape(nm, -1).T]
    else:
        starts += [np.full(nm, g) for g in grid1]
        for i in range(nm):  # one module free, others excluded / included
            for g in grid1:
                x = np.full(nm, hi)
                x[i] = g
                starts.append(x)
                x2 = np.full(nm, 0.0)
                x2[i] = hi
                starts.append(x2)
    starts += [rng.uniform(lo, hi, nm) for _ in range(n_random)]
    best = (-np.inf, None)
    for x0 in starts:
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * nm,
                     options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        if -r.fun > best[0]:
            best = (-float(r.fun), r.x.copy())
    return best


# (a) Gate-1 architectures
arch_rows = []
t0 = time.perf_counter()
for a in cfg.architectures():
    ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
    Ks, bs, colmod, active = D.sub(ranks)
    rec = scores[a.arch_id]
    la_rec = np.asarray(rec["rvm_log_alpha"])
    E_rec_recomputed, _ = evidence_and_grad(Ks, bs, la_rec, colmod, len(active), False)
    cols, _, _ = D.columns(ranks)
    E_direct = evidence_direct(Phi[:, cols], t, np.exp(la_rec[colmod]))
    Ebest, xbest = dense_max(Ks, bs, colmod, len(active))
    arch_rows.append({"arch_id": a.arch_id, "E_recorded": rec["group_rvm"], "E_at_recorded_alpha": E_rec_recomputed,
                      "E_direct_nxn_at_recorded_alpha": E_direct, "E_dense_multistart": Ebest,
                      "dense_minus_recorded": Ebest - rec["group_rvm"], "log_alpha_recorded": la_rec.tolist(),
                      "log_alpha_dense": xbest.tolist()})
res["gate1_arch_evidence"] = arch_rows
res["gate1_arch_evidence_summary"] = {
    "max_rel_err_gram_vs_direct": max(abs(r["E_at_recorded_alpha"] - r["E_direct_nxn_at_recorded_alpha"]) /
                                      max(1.0, abs(r["E_direct_nxn_at_recorded_alpha"])) for r in arch_rows),
    "max_dense_minus_recorded": max(r["dense_minus_recorded"] for r in arch_rows),
    "n_arch_dense_better_by_1e-6": sum(r["dense_minus_recorded"] > 1e-6 for r in arch_rows),
    "wall_s": time.perf_counter() - t0}

# (b) independent scores
ind = []
for a in cfg.architectures():
    ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
    cols, _, _ = D.columns(ranks)
    P = Phi[:, cols]
    bA = P.T @ t
    KA = P.T @ P
    gn = float(bA @ bA)
    fi = float(0.5 * np.sum(bA ** 2 / (np.diag(KA) + lam)))
    nw = float(0.5 * bA @ np.linalg.solve(KA + lam * np.eye(len(cols)), bA))
    rec = scores[a.arch_id]
    ind.append({"arch_id": a.arch_id, "gn_rel_err": abs(gn - rec["gradient_norm"]) / abs(gn),
                "fisher_rel_err": abs(fi - rec["independent_fisher"]) / abs(fi),
                "newton_rel_err": abs(nw - rec["conditional_newton"]) / abs(nw),
                "columns": cols.tolist()})
res["independent_scores_max_rel_err"] = {k: max(r[k] for r in ind) for k in ["gn_rel_err", "fisher_rel_err", "newton_rel_err"]}
res["arch_columns"] = {r["arch_id"]: r["columns"] for r in ind}

# (c) ridge CV
grid = cfg["probe"]["nonbayes_ridge_grid"]
n = Phi.shape[0]
fold = np.arange(n) % 3
cv = {}
for lmb in grid:
    errs = []
    for f in range(3):
        tr, te = fold != f, fold == f
        s = np.linalg.lstsq(np.vstack([Phi[tr], np.sqrt(lmb) * np.eye(Phi.shape[1])]),
                            np.concatenate([t[tr], np.zeros(Phi.shape[1])]), rcond=None)[0]
        errs.append(float(np.mean((t[te] - Phi[te] @ s) ** 2)))
    cv[lmb] = float(np.mean(errs))
res["ridge_cv_independent"] = {"cv_mse": cv, "argmin": min(cv, key=cv.get),
                               "recorded": sel["conditional_newton"]["cv"]}

# (d) selection job reproduction
scr = OUT / "selection_rerun"
scr.mkdir(exist_ok=True)
(scr / "features.npz").write_bytes((pdir / "features.npz").read_bytes())
t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "rvm_lora.selection_job", str(scr), str(scr / "selection.json"), "--gate1",
                "--config", str(REPO / "configs/preregistered.yaml")], check=True, cwd=REPO,
               env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "PYTHONPATH": str(REPO / "src")})
new = json.loads((scr / "selection.json").read_text())
cmp = {}
for k in ["group_rvm", "conditional_newton", "rvm_no_logdet"]:
    cmp[k] = {"ranks_equal": new[k]["ranks"] == sel[k]["ranks"],
              "objective_equal": new[k]["objective"] == sel[k]["objective"],
              "threshold_equal": new[k].get("threshold") == sel[k].get("threshold")}
cmp["fresh_null_equal"] = new["group_rvm_fresh_null"]["selected_units"] == sel["group_rvm_fresh_null"]["selected_units"]
cmp["newton_diag_threshold_equal"] = new["conditional_newton"]["diagnostic_null_threshold"] == \
    sel["conditional_newton"]["diagnostic_null_threshold"]
cmp["arch_scores_equal"] = all(
    x["group_rvm"] == y["group_rvm"] and x["conditional_newton"] == y["conditional_newton"]
    for x, y in zip(new["architecture_scores"], sel["architecture_scores"]))
cmp["rerun_wall_s"] = time.perf_counter() - t0
res["selection_rerun_comparison"] = cmp
res["thresholded_newton_equals_rvm"] = sel["conditional_newton"]["diagnostic_thresholded_ranks"] == sel["group_rvm"]["ranks"]
res["rvm_selected"] = sel["group_rvm"]["ranks"]
res["thresholded_newton_selected"] = sel["conditional_newton"]["diagnostic_thresholded_ranks"]
res["rvm_trace"] = [{k: s.get(k) for k in ["iter", "best_group", "gain_per_param", "action", "ranks_after",
                                           "boundary_pruned", "deletions", "log_alpha"]} for s in sel["group_rvm"]["trace"]]
res["newton_thr_trace"] = None
res["thresholds"] = {"rvm": sel["group_rvm"]["threshold"], "newton_diag": sel["conditional_newton"]["diagnostic_null_threshold"],
                     "no_logdet": sel["rvm_no_logdet"]["threshold"]}
res["fresh_null"] = {"activations": sel["group_rvm_fresh_null"]["activations"],
                     "selected_units": sel["group_rvm_fresh_null"]["selected_units"]}
res["numerical_issues"] = sel["group_rvm"]["numerical_issues"]

# (e) RVM trace states vs dense multistart
states = []
for s in sel["group_rvm"]["trace"]:
    if s.get("ranks_after") is None:
        continue
    ranks = tuple(s["ranks_after"])
    Ks, bs, colmod, active = D.sub(ranks)
    Eb, xb = dense_max(Ks, bs, colmod, len(active), n_random=150)
    states.append({"iter": s["iter"], "ranks": list(ranks), "recorded": s["objective_after_sweep"], "dense": Eb,
                   "dense_minus_recorded": Eb - s["objective_after_sweep"]})
res["rvm_trace_states_dense"] = states
(OUT / "phase4_evidence.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: res[k] for k in ["gate1_arch_evidence_summary", "independent_scores_max_rel_err", "ridge_cv_independent",
                                      "selection_rerun_comparison", "thresholded_newton_equals_rvm", "rvm_selected",
                                      "thresholded_newton_selected", "thresholds", "rvm_trace_states_dense"]},
                 indent=1, default=str))
for r in arch_rows:
    print(f"{r['arch_id']:14s} rec={r['E_recorded']:+.6f} direct={r['E_direct_nxn_at_recorded_alpha']:+.6f} "
          f"dense={r['E_dense_multistart']:+.6f} diff={r['dense_minus_recorded']:+.2e}")
print(json.dumps(res["rvm_trace"], indent=0, default=str)[:3000])
