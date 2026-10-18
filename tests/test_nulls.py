"""Selector behaviour, null calibration determinism, budget/prefix constraints, pooled == sequential."""

import numpy as np

from rvm_lora.config import eligible_modules
from rvm_lora.selectors import (
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

MODS = eligible_modules()


def _dict(seed=0, signal=True, n=256):
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((n, 192)) * np.tile(1 / np.arange(1, 9), 24)[None, :]
    s = np.zeros(192)
    if signal:
        s[[0, 1, 9, 80]] = [1.0, 0.6, 0.8, 0.7]
    t = Phi @ s + rng.standard_normal(n)
    return Phi, t, Dictionary(Phi.T @ Phi, Phi.T @ t, MODS, 8, 1536)


def test_candidate_space():
    g = candidate_groups(MODS)
    assert len(g) == 24 * 2 + 12
    assert len({x.key for x in g}) == len(g)


def test_rvm_selects_signal_and_respects_constraints(record):
    Phi, t, D = _dict()
    G = candidate_groups(MODS)
    thr, maxima, _ = calibrate_threshold(RVMObjective, Phi, D, G, 8, 24, 30, 1931, 95)
    res = greedy_select(RVMObjective(D), G, 8, 24, thr)
    assert sum(res.ranks) <= 24 and max(res.ranks) <= 8
    assert res.ranks[0] >= 1 and res.ranks[1] >= 1 and res.ranks[10] >= 1
    assert res.stop_reason in {"null_threshold", "budget_exhausted", "cycle_detected"}
    assert all(r >= 0 for r in res.ranks)
    record(selected_units=res.rank_units, stop=res.stop_reason, threshold=thr)


def test_pool_equals_sequential():
    Phi, t, D = _dict(1)
    G = candidate_groups(MODS)
    thr_a, m_a, _ = calibrate_threshold(RVMObjective, Phi, D, G, 8, 24, 16, 1931, 95, n_workers=0)
    thr_b, m_b, _ = calibrate_threshold(RVMObjective, Phi, D, G, 8, 24, 16, 1931, 95, n_workers=4)
    assert thr_a == thr_b and np.array_equal(m_a, m_b)
    a = greedy_select(RVMObjective(D), G, 8, 24, thr_a, n_workers=0)
    b = greedy_select(RVMObjective(D), G, 8, 24, thr_a, n_workers=4)
    assert a.ranks == b.ranks and a.objective_value == b.objective_value


def test_null_behaviour(record):
    Phi, t, D = _dict(2, signal=False)
    G = candidate_groups(MODS)
    thr, maxima, _ = calibrate_threshold(RVMObjective, Phi, D, G, 8, 24, 40, 1931, 95)
    act, units, _ = evaluate_null(RVMObjective, Phi, D, G, 8, 24, thr, 40, 1932)
    assert act <= 10  # ~5% expected under the complete null
    zero = greedy_select(RVMObjective(D.with_b(np.zeros(192))), G, 8, 24, thr)
    assert zero.rank_units == 0 and zero.stop_reason == "null_threshold"
    thr2, maxima2, _ = calibrate_threshold(RVMObjective, Phi, D, G, 8, 24, 40, 1931, 95)
    assert thr == thr2 and np.array_equal(maxima, maxima2)  # deterministic seeds
    record(threshold=thr, fresh_activations=act, count=40)


def test_newton_and_ablation_fill_budget_without_threshold():
    Phi, t, D = _dict(3)
    G = candidate_groups(MODS)
    lam, info = select_ridge_lambda(Phi, t, [1e-3, 1e-2, 1e-1, 1, 10, 100], 3)
    assert lam in info["cv_mse"]
    n = greedy_select(NewtonObjective(D, lam), G, 8, 24, -np.inf)
    assert n.rank_units == 24 and n.stop_reason == "budget_exhausted"
    z = greedy_select(NoLogdetObjective(D), G, 8, 24, -np.inf)
    assert z.rank_units == 24


def test_architecture_scores_consistent():
    Phi, t, D = _dict(4)
    ranks = tuple([2, 2] + [0] * 22)
    s = architecture_scores(D, ranks, 1.0)
    b = D.b[[0, 1, 8, 9]]
    assert np.isclose(s["gradient_norm"], b @ b)
    assert s["conditional_newton"] > 0 and np.isfinite(s["group_rvm"])
    assert len(s["rvm_log_alpha"]) == 2
