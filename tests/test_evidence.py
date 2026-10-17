"""Gate 0 items 1-3: evidence identities, singleton analytic optimum, conditional/group increments."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from rvm_lora.evidence import (
    conditional_fixed_increment,
    evidence_and_grad,
    evidence_direct,
    optimize_evidence,
    singleton_bounded,
    singleton_optimum,
)

RTOL, ATOL = 1e-5, 1e-7


def _problem(seed=0, n=80, cols=12, modules=4):
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((n, cols)) @ (np.eye(cols) + 0.3 * rng.standard_normal((cols, cols)))
    s = rng.standard_normal(cols) * (rng.random(cols) < 0.5)
    t = Phi @ s * 0.2 + rng.standard_normal(n)
    colmod = np.repeat(np.arange(modules), cols // modules)
    return Phi, t, colmod


def _rel(a, b):
    return abs(a - b) / max(abs(b), ATOL / RTOL)


def test_gram_evidence_matches_direct(record):
    Phi, t, colmod = _problem()
    K, b = Phi.T @ Phi, Phi.T @ t
    worst = 0.0
    rng = np.random.default_rng(1)
    for _ in range(30):
        la = rng.uniform(-6, 6, size=4)
        E, _ = evidence_and_grad(K, b, la, colmod, 4)
        Ed = evidence_direct(Phi, t, np.exp(la)[colmod])
        assert np.isclose(E, Ed, rtol=RTOL, atol=ATOL), (E, Ed)
        worst = max(worst, _rel(E, Ed))
    # sequential increments (Cholesky/Gram) against direct differences
    la = rng.uniform(-3, 3, size=4)
    prev_g = prev_d = 0.0
    inc_err = 0.0
    for k in range(1, 5):
        cols = colmod < k
        Eg, _ = evidence_and_grad(K[np.ix_(cols, cols)], b[cols], la[:k], colmod[cols], k)
        Ed = evidence_direct(Phi[:, cols], t, np.exp(la)[colmod[cols]])
        inc_err = max(inc_err, _rel(Eg - prev_g, Ed - prev_d))
        assert np.isclose(Eg - prev_g, Ed - prev_d, rtol=RTOL, atol=ATOL)
        prev_g, prev_d = Eg, Ed
    # numerical limits: all alpha at the exclusion bound -> E ~ 0; tiny alpha stays finite
    E_hi, _ = evidence_and_grad(K, b, np.full(4, 23.02585092994), colmod, 4)
    E_lo, _ = evidence_and_grad(K, b, np.full(4, -23.02585092994), colmod, 4)
    assert abs(E_hi) < 1e-6 and np.isfinite(E_lo)
    record(max_rel_err_evidence=worst, max_rel_err_increments=inc_err, E_at_upper_bound=E_hi, rtol=RTOL, atol=ATOL)


def test_gradient_matches_finite_difference(record):
    Phi, t, colmod = _problem(2)
    K, b = Phi.T @ Phi, Phi.T @ t
    la = np.array([0.3, -1.0, 2.0, 0.5])
    _, g = evidence_and_grad(K, b, la, colmod, 4)
    eps = 1e-6
    fd = np.array([(evidence_and_grad(K, b, la + eps * e, colmod, 4)[0] -
                    evidence_and_grad(K, b, la - eps * e, colmod, 4)[0]) / (2 * eps) for e in np.eye(4)])
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-6)
    record(max_abs_grad_err=float(np.max(np.abs(g - fd))))


def test_singleton_analytic_vs_numeric(record):
    rng = np.random.default_rng(3)
    worst = 0.0
    n_excluded = 0
    for _ in range(200):
        h = float(rng.uniform(0.1, 50))
        q = float(rng.normal() * np.sqrt(h) * rng.uniform(0.2, 4))
        d_star, a_star = singleton_optimum(q, h)
        f = lambda la: -0.5 * (q * q / (h + np.exp(la)) - np.log1p(h / np.exp(la)))  # noqa: E731
        r = minimize_scalar(f, bounds=(-23, 23), method="bounded", options={"xatol": 1e-10})
        num = max(-r.fun, 0.0)
        if q * q <= h:
            n_excluded += 1
            assert d_star == 0.0 and np.isinf(a_star)
            assert num < 1e-6
        else:
            assert np.isclose(d_star, num, rtol=1e-6, atol=1e-9)
            worst = max(worst, abs(d_star - num))
        # bounded closed form used by the selector agrees with the generic L-BFGS-B optimiser
        Eb, _ = singleton_bounded(q, h)
        fit = optimize_evidence(np.array([[h]]), np.array([q]), np.array([0]), 1)
        assert np.isclose(Eb, fit.evidence, rtol=1e-6, atol=1e-9)
    assert n_excluded > 10
    record(max_abs_err=worst, excluded_cases=n_excluded, cases=200)


def test_singleton_ranking_equals_q2_over_h(record):
    """Negative novelty control: optimised singleton evidence orders eligible atoms exactly as q^2/h."""
    rng = np.random.default_rng(4)
    n = 100
    Phi = rng.standard_normal((n, 30))
    t = Phi[:, :3] @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(n)
    S = np.array([0, 1])
    aS = np.array([2.0, 5.0])
    C = np.eye(n) + (Phi[:, S] / aS) @ Phi[:, S].T
    Ci_t = np.linalg.solve(C, t)
    Ci_P = np.linalg.solve(C, Phi)
    taus, deltas = [], []
    for j in range(2, 30):
        q, h = Phi[:, j] @ Ci_t, Phi[:, j] @ Ci_P[:, j]
        d, _ = singleton_optimum(q, h)
        # numeric conditional optimum with existing precisions fixed
        f = lambda la: -conditional_fixed_increment(Phi.T @ Phi, Phi.T @ t, S, aS, np.array([j]),  # noqa: E731
                                                    np.array([np.exp(la)]))
        num = max(-minimize_scalar(f, bounds=(-23, 23), method="bounded", options={"xatol": 1e-10}).fun, 0.0)
        assert np.isclose(d, num, rtol=1e-5, atol=1e-8)
        if q * q / h > 1:
            taus.append(q * q / h)
            deltas.append(d)
    order_tau = np.argsort(-np.asarray(taus), kind="stable")
    order_e = np.argsort(-np.asarray(deltas), kind="stable")
    assert len(taus) >= 3
    assert list(order_tau) == list(order_e)
    record(eligible_atoms=len(taus), identical_ordering=True)


def test_conditional_fixed_increment_equals_difference(record):
    Phi, t, colmod = _problem(5, n=70, cols=12, modules=4)
    K, b = Phi.T @ Phi, Phi.T @ t
    rng = np.random.default_rng(6)
    worst = 0.0
    for G in (np.array([9]), np.array([9, 10]), np.array([6, 7, 8])):  # rank-1, block rank-2, rank-3
        S = np.arange(0, 6)
        aS = np.exp(rng.uniform(-2, 2, size=S.size))
        aG = np.exp(rng.uniform(-2, 2)) * np.ones(G.size)
        d = conditional_fixed_increment(K, b, S, aS, G, aG)
        full = np.concatenate([S, G])
        direct = evidence_direct(Phi[:, full], t, np.concatenate([aS, aG])) - evidence_direct(Phi[:, S], t, aS)
        assert np.isclose(d, direct, rtol=RTOL, atol=ATOL)
        worst = max(worst, _rel(d, direct))
    # re-optimised definition >= fixed-precision definition (refit can only help)
    S_cols, G_cols = np.arange(0, 6), np.arange(6, 9)
    cm = colmod[:9]
    fit_S = optimize_evidence(K[np.ix_(S_cols, S_cols)], b[S_cols], cm[:6], 2)
    full = np.arange(9)
    fit_SG = optimize_evidence(K[np.ix_(full, full)], b[full], cm, 3)
    fixed = max(conditional_fixed_increment(K, b, S_cols, np.exp(fit_S.log_alpha)[cm[:6]], G_cols,
                                            np.exp(la) * np.ones(3)) for la in np.linspace(-10, 10, 201))
    assert fit_SG.evidence - fit_S.evidence >= fixed - 1e-6
    record(max_rel_err=worst)


def test_degenerate_cases(record):
    rng = np.random.default_rng(7)
    n = 50
    x = rng.standard_normal(n)
    Phi = np.stack([x, x, rng.standard_normal(n), np.zeros(n)], axis=1)  # duplicate + zero column
    t = 0.5 * x + rng.standard_normal(n)
    K, b = Phi.T @ Phi, Phi.T @ t
    cm = np.array([0, 0, 1, 2])
    fit = optimize_evidence(K, b, cm, 3)
    assert np.isfinite(fit.evidence) and fit.evidence > 0
    # zero-gradient module: evidence is exactly flat in its precision (contributes nothing at any alpha)
    vals = [evidence_and_grad(K, b, np.r_[fit.log_alpha[:2], la], cm, 3)[0] for la in (-20.0, 0.0, 20.0)]
    assert np.ptp(vals) < 1e-9
    assert abs(evidence_and_grad(K, b, fit.log_alpha, cm, 3)[1][2]) < 1e-12
    E, g = evidence_and_grad(K, b, np.full(3, 23.02585092994), cm, 3)  # all-infinite precision
    assert abs(E) < 1e-6
    fit0 = optimize_evidence(K, np.zeros(4), cm, 3)  # no-improvement case
    assert fit0.evidence <= 1e-6 and fit0.boundary[:2].all()  # module 2 is flat (zero column)
    record(duplicate_column_evidence=fit.evidence, zero_target_evidence=fit0.evidence)


@pytest.mark.parametrize("seed", [0, 1])
def test_optimizer_deterministic(seed):
    Phi, t, colmod = _problem(seed)
    K, b = Phi.T @ Phi, Phi.T @ t
    a = optimize_evidence(K, b, colmod, 4)
    c = optimize_evidence(K, b, colmod, 4)
    assert a.evidence == c.evidence and np.array_equal(a.log_alpha, c.log_alpha)
