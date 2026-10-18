"""CPU-only structure-selection job (no torch/CUDA import, so fork pools are safe).

Run as a subprocess with single-threaded BLAS:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m rvm_lora.selection_job <probe_dir> <out.json> [--gate1]

Inputs: <probe_dir>/features.npz (Phi, t, K, b) written by the GPU probe. Outputs: thresholds, null
diagnostics, selected structures with traces and timings for group RVM, conditional Newton and the
no-logdet ablation; with --gate1 also the precomputed scores of all 20 architectures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .selectors import (
    Dictionary,
    NewtonObjective,
    NoLogdetObjective,
    RVMObjective,
    architecture_scores,
    calibrate_threshold,
    candidate_groups,
    evaluate_null,
    greedy_select,
    select_ridge_lambda,
)
from .tracking import write_json


def _rvm(bounds):
    def make(D):
        return RVMObjective(D, bounds)
    return make


def _newton(lam):
    def make(D):
        return NewtonObjective(D, lam)
    return make


def _nolog(bounds):
    def make(D):
        return NoLogdetObjective(D, bounds)
    return make


def run(probe_dir: Path, out: Path, gate1: bool, config_path: str | None, n_workers: int) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    pr = cfg["probe"]
    z = np.load(probe_dir / "features.npz")
    Phi, t = z["Phi"].astype(np.float64), z["t"].astype(np.float64)
    K, b = Phi.T @ Phi, Phi.T @ t
    D = Dictionary(K, b, cfg.modules, cfg.n_dir, cfg.rank_unit_params)
    groups = candidate_groups(cfg.modules)
    budget = int(cfg["stage_b"]["budget_rank_units"])
    max_rank = cfg.max_rank
    bounds = cfg.log_alpha_bounds
    res: dict = {"probe_dir": str(probe_dir), "n_workers": n_workers, "budget_units": budget,
                 "n_candidate_groups": len(groups), "modules": cfg.modules}

    # ---- conditional Newton: ridge CV, greedy to the full B24 budget (PREREGISTRATION §4 "same budget";
    # stopping is part of the Bayesian mechanism under test, see docs/PROTOCOL_CLARIFICATIONS.md C1)
    t0 = time.perf_counter()
    lam, cv = select_ridge_lambda(Phi, t, pr["nonbayes_ridge_grid"], pr["nonbayes_ridge_cv_folds"])
    selN = greedy_select(NewtonObjective(D, lam), groups, max_rank, budget, -float("inf"))
    newton_wall = time.perf_counter() - t0
    # diagnostic only: Newton's own null threshold and what it would stop at
    t0 = time.perf_counter()
    thrN, maxN, _ = calibrate_threshold(_newton(lam), Phi, D, groups, max_rank, budget, pr["null_calibration_count"],
                                        pr["null_calibration_seed"], pr["null_percentile"], n_workers)
    selNt = greedy_select(NewtonObjective(D, lam), groups, max_rank, budget, thrN)
    res["conditional_newton"] = {"lambda": lam, "cv": cv, "threshold": None,
                                 "ranks": selN.rank_dict(cfg.modules), "rank_units": selN.rank_units,
                                 "objective": selN.objective_value, "stop_reason": selN.stop_reason,
                                 "trace": selN.trace, "cpu_wall_s": newton_wall,
                                 "diagnostic_null_threshold": thrN, "diagnostic_null_max_gains": maxN.tolist(),
                                 "diagnostic_thresholded_ranks": selNt.rank_dict(cfg.modules),
                                 "diagnostic_thresholded_units": selNt.rank_units,
                                 "diagnostic_wall_s": time.perf_counter() - t0}

    # ---- group RVM: null threshold (seed 1931), greedy with deletion
    t0 = time.perf_counter()
    thrR, maxR, _ = calibrate_threshold(_rvm(bounds), Phi, D, groups, max_rank, budget, pr["null_calibration_count"],
                                        pr["null_calibration_seed"], pr["null_percentile"], n_workers)
    objR = RVMObjective(D, bounds)
    selR = greedy_select(objR, groups, max_rank, budget, thrR, n_workers=n_workers)
    res["group_rvm"] = {"threshold": thrR, "threshold_units": "log-evidence gain per adapter parameter",
                        "null_max_gains": maxR.tolist(), "ranks": selR.rank_dict(cfg.modules),
                        "rank_units": selR.rank_units, "objective": selR.objective_value,
                        "stop_reason": selR.stop_reason, "trace": selR.trace, "numerical_issues": objR.all_issues(),
                        "n_objective_evals": selR.n_objective_evals, "cpu_wall_s": time.perf_counter() - t0}

    # ---- diagnostics (not part of any method's timed cost)
    t0 = time.perf_counter()
    act, units, _ = evaluate_null(_rvm(bounds), Phi, D, groups, max_rank, budget, thrR,
                                  pr["null_evaluation_count"], pr["null_evaluation_seed"], n_workers)
    res["group_rvm_fresh_null"] = {"activations": act, "count": pr["null_evaluation_count"],
                                   "seed": pr["null_evaluation_seed"], "selected_units": units,
                                   "wall_s": time.perf_counter() - t0}
    t0 = time.perf_counter()
    thr0, max0, _ = calibrate_threshold(_nolog(bounds), Phi, D, groups, max_rank, budget, pr["null_calibration_count"],
                                        pr["null_calibration_seed"], pr["null_percentile"], n_workers)
    sel0 = greedy_select(NoLogdetObjective(D, bounds), groups, max_rank, budget, thr0)
    res["rvm_no_logdet"] = {"threshold": thr0, "ranks": sel0.rank_dict(cfg.modules), "rank_units": sel0.rank_units,
                            "objective": sel0.objective_value, "stop_reason": sel0.stop_reason, "trace": sel0.trace,
                            "exploratory": True, "cpu_wall_s": time.perf_counter() - t0}

    if gate1:
        t0 = time.perf_counter()
        scores = []
        for a in cfg.architectures():
            ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
            s = architecture_scores(D, ranks, lam, bounds)
            scores.append({"arch_id": a.arch_id, "family": a.family, "layers": list(a.layers),
                           "parameters": a.parameters, "ranks": a.ranks, **s})
        res["architecture_scores"] = scores
        res["architecture_scores_lambda"] = lam
        res["architecture_scores_wall_s"] = time.perf_counter() - t0
    write_json(out, res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir")
    ap.add_argument("out")
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    run(Path(a.probe_dir), Path(a.out), a.gate1, a.config, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
