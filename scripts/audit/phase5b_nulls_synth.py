"""Parallel part of the fix verification: (3) all 100 seed-1931 null responses x 36 multi-column groups and
(4) 300 synthetic scale-disparate problems; repaired optimiser vs an independent dense multistart."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

CODE, REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE / "src"))
import rvm_lora.evidence as ev  # noqa: E402
from rvm_lora.config import load_config  # noqa: E402
from rvm_lora.selectors import Dictionary, apply, candidate_groups, null_responses  # noqa: E402

assert "GLOBAL_IMPROVEMENT_TOL" in dir(ev)
cfg = load_config(CODE / "configs/preregistered.yaml")
lo, hi = cfg.log_alpha_bounds
z = np.load(REPO / "artifacts/probes/qnli_s42/features.npz")
Phi, t = z["Phi"].astype(np.float64), z["t"].astype(np.float64)
D = Dictionary(Phi.T @ Phi, Phi.T @ t, cfg.modules, cfg.n_dir, cfg.rank_unit_params)
groups = candidate_groups(cfg.modules)
T = null_responses(Phi.shape[0], 100, cfg["probe"]["null_calibration_seed"])


def dense(K, b, cm, nm, seed):
    rng = np.random.default_rng(seed)
    def f(x):
        E, g = ev.evidence_and_grad(K, b, x, cm, nm, True)
        return -E, -g
    grid = np.array([lo, -16, -10, -5, 0, 5, 10, 16, hi])
    starts = [np.array(p) for p in np.array(np.meshgrid(*[grid] * nm)).reshape(nm, -1).T] if nm <= 2 else \
        [np.full(nm, g) for g in grid] + [rng.uniform(lo, hi, nm) for _ in range(60)]
    return max(-minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * nm).fun for x0 in starts)


def old(K, b, cm, nm):
    def f(x):
        E, g = ev.evidence_and_grad(K, b, x, cm, nm, True)
        return -E, -g
    return max(-minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * nm,
                         options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9}).fun
               for x0 in (np.zeros(nm), np.full(nm, np.clip(np.log(max(np.median(np.diag(K)), 1e-300)), lo, hi))))


def null_job(i):
    Dn = D.with_b(Phi.T @ T[i])
    rows = []
    for g in groups:
        K, b, cm, act = Dn.sub(apply(tuple([0] * 24), g))
        if K.shape[0] == 1:
            continue
        rows.append((ev.optimize_evidence(K, b, cm, len(act), bounds=(lo, hi)).evidence, old(K, b, cm, len(act)),
                     dense(K, b, cm, len(act), i)))
    a = np.array(rows)
    return {"i": i, "n": len(rows), "max_dense_minus_new": float((a[:, 2] - a[:, 0]).max()),
            "n_new_below": int(((a[:, 2] - a[:, 0]) > 1e-7).sum()), "n_old_below": int(((a[:, 2] - a[:, 1]) > 1e-6).sum())}


def synth_job(k):
    rng = np.random.default_rng(1000 + k)
    nm = int(rng.integers(2, 5))
    cm = np.repeat(np.arange(nm), rng.integers(1, 5, nm))
    p = len(cm)
    X = rng.standard_normal((128, p)) @ (np.eye(p) + 0.5 * rng.standard_normal((p, p)))
    X *= 10.0 ** rng.uniform(-2, 2, p)[None, :]
    w = rng.standard_normal(p) * (rng.random(p) < 0.6) / np.linalg.norm(X, axis=0)
    y = X @ w * rng.uniform(0, 6) + rng.standard_normal(128)
    K, b = X.T @ X, X.T @ y
    new = ev.optimize_evidence(K, b, cm, nm, bounds=(lo, hi)).evidence
    return {"k": k, "nm": nm, "new": new, "old": old(K, b, cm, nm), "dense": dense(K, b, cm, nm, k)}


if __name__ == "__main__":
    with Pool(8) as pool:
        nulls = pool.map(null_job, range(100))
        syn = pool.map(synth_job, range(300))
    res = {"null_groups": {"n_problems": sum(r["n"] for r in nulls),
                           "max_dense_minus_new": max(r["max_dense_minus_new"] for r in nulls),
                           "n_new_below_dense_1e-7": sum(r["n_new_below"] for r in nulls),
                           "n_old_below_dense_1e-6": sum(r["n_old_below"] for r in nulls)},
           "synthetic": {"n": len(syn), "max_dense_minus_new": max(r["dense"] - r["new"] for r in syn),
                         "n_new_below_dense_1e-7": sum(r["dense"] - r["new"] > 1e-7 for r in syn),
                         "n_old_below_dense_1e-6": sum(r["dense"] - r["old"] > 1e-6 for r in syn),
                         "n_new_above_dense_1e-7": sum(r["new"] - r["dense"] > 1e-7 for r in syn)}}
    OUT.write_text(json.dumps(res, indent=2, default=lambda o: o.item()))
    print(json.dumps(res, indent=1, default=lambda o: o.item()))
