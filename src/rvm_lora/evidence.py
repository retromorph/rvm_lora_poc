"""Float64 Type-II marginal evidence for the grouped-ARD linear-Gaussian surrogate.

Model (METHODS.md §3): t = Phi_S s + eps, eps ~ N(0, I), s_k ~ N(0, 1/alpha_{m(k)}), with one
precision per active module shared by that module's prefix columns. Relative to the empty model

    E(S, alpha) = 1/2 b^T (K + A)^{-1} b - 1/2 [logdet(K + A) - logdet A],   K = Phi^T Phi, b = Phi^T t.

We evaluate it through the well-conditioned form M = I + D K D with D = A^{-1/2} (eigenvalues >= 1),
using Cholesky factors only (no explicit inverses for the objective):

    logdet(K + A) - logdet A = logdet M,   b^T (K + A)^{-1} b = (D b)^T M^{-1} (D b).

Gradient w.r.t. log alpha_k (column k), summed over the columns of each module:

    dE/dlog alpha_k = 1/2 (1 - (M^{-1})_kk - ((M^{-1} D b)_k)^2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

DEFAULT_LOG_ALPHA_BOUNDS = (-23.02585092994, 23.02585092994)  # alpha in [1e-10, 1e10] (YAML)
BOUNDARY_TOL = 1e-6  # log-alpha distance to the upper bound that counts as boundary pruning
MAX_RELATIVE_JITTER = 1e-8


class NumericalError(RuntimeError):
    pass


def chol_with_jitter(M: np.ndarray, issues: list | None = None) -> tuple[np.ndarray, float]:
    """Lower Cholesky factor; adds the smallest diagonal jitter that succeeds and records it.

    Raises NumericalError if more than MAX_RELATIVE_JITTER * mean(diag) is needed.
    """
    M = np.asarray(M, dtype=np.float64)
    try:
        return np.linalg.cholesky(M), 0.0
    except np.linalg.LinAlgError:
        pass
    scale = float(np.mean(np.abs(np.diag(M)))) or 1.0
    for k in range(-14, int(np.log10(MAX_RELATIVE_JITTER)) + 1):
        jitter = scale * 10.0**k
        try:
            L = np.linalg.cholesky(M + jitter * np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            continue
        if issues is not None:
            issues.append({"event": "cholesky_jitter", "jitter": jitter, "dim": M.shape[0]})
        return L, jitter
    raise NumericalError(f"Cholesky failed even with relative jitter {MAX_RELATIVE_JITTER}")


def expand_alpha(log_alpha_modules: np.ndarray, col_module: np.ndarray) -> np.ndarray:
    """Per-column alpha from per-module log-precisions; col_module[k] indexes into log_alpha_modules."""
    return np.exp(np.asarray(log_alpha_modules, dtype=np.float64)[col_module])


def evidence_and_grad(
    K: np.ndarray, b: np.ndarray, log_alpha_modules: np.ndarray, col_module: np.ndarray,
    n_modules: int | None = None, with_grad: bool = True, issues: list | None = None,
) -> tuple[float, np.ndarray | None]:
    """E(S, alpha) and dE/dlog(alpha_m) for the selected columns (K, b already restricted to S)."""
    d = K.shape[0]
    if d == 0:
        return 0.0, np.zeros(0)
    n_modules = int(col_module.max()) + 1 if n_modules is None else n_modules
    dvec = np.exp(-0.5 * np.asarray(log_alpha_modules, dtype=np.float64)[col_module])
    M = (dvec[:, None] * K) * dvec[None, :]
    M[np.diag_indices(d)] += 1.0
    L, _ = chol_with_jitter(M, issues)
    db = dvec * b
    c = solve_triangular(L, db, lower=True, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    E = 0.5 * float(c @ c) - 0.5 * logdet
    if not with_grad:
        return E, None
    Linv = solve_triangular(L, np.eye(d), lower=True, check_finite=False)
    minv_diag = np.sum(Linv * Linv, axis=0)
    w = Linv.T @ c  # M^{-1} D b
    g_col = 0.5 * (1.0 - minv_diag - w * w)
    g = np.bincount(col_module, weights=g_col, minlength=n_modules)
    return E, g


def evidence_direct(Phi: np.ndarray, t: np.ndarray, alpha_cols: np.ndarray) -> float:
    """Reference: log N(t | 0, I + Phi A^-1 Phi^T) - log N(t | 0, I) via n x n slogdet/solve."""
    n = Phi.shape[0]
    C = np.eye(n) + (Phi / alpha_cols[None, :]) @ Phi.T
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        raise NumericalError("covariance not PD")
    quad = float(t @ np.linalg.solve(C, t))
    return -0.5 * logdet - 0.5 * quad + 0.5 * float(t @ t)


def singleton_optimum(q: float, h: float) -> tuple[float, float]:
    """Analytic maximiser for one column: returns (Delta*, alpha*). alpha* = inf when q^2 <= h."""
    if h <= 0:
        return 0.0, np.inf
    tau = q * q / h
    if tau <= 1.0:
        return 0.0, np.inf
    return 0.5 * (tau - 1.0 - np.log(tau)), h * h / (q * q - h)


def singleton_bounded(q: float, h: float, bounds: tuple[float, float] = DEFAULT_LOG_ALPHA_BOUNDS
                      ) -> tuple[float, float]:
    """Exact bounded optimum of E for one column from empty support: returns (E*, log alpha*).

    E(alpha) = 1/2 [q^2/(h + alpha) - log(1 + h/alpha)] is unimodal in log alpha with stationary point
    alpha* = h^2/(q^2 - h) when q^2 > h, else increasing towards the exclusion bound.
    """
    lo, hi = bounds
    if h <= 0:
        return 0.0, hi
    if q * q > h:
        la = float(np.clip(np.log(h * h / (q * q - h)), lo, hi))
    else:
        la = hi
    a = np.exp(la)
    return 0.5 * (q * q / (h + a) - np.log1p(h / a)), la


def conditional_fixed_increment(
    K: np.ndarray, b: np.ndarray, S: np.ndarray, alpha_S: np.ndarray, G: np.ndarray, alpha_G: np.ndarray,
) -> float:
    """Delta(G | S) with existing precisions fixed (METHODS §3), computed in Gram space via Woodbury."""
    S = np.asarray(S, dtype=int)
    G = np.asarray(G, dtype=int)
    if S.size:
        MS = K[np.ix_(S, S)] + np.diag(alpha_S)
        LS, _ = chol_with_jitter(MS)
        KSG = K[np.ix_(S, G)]
        X = solve_triangular(LS, KSG, lower=True)
        y = solve_triangular(LS, b[S], lower=True)
        bG = b[G] - X.T @ y
        KG = K[np.ix_(G, G)] - X.T @ X
    else:
        bG = b[G].copy()
        KG = K[np.ix_(G, G)].copy()
    MG = KG + np.diag(alpha_G)
    LG, _ = chol_with_jitter(MG)
    c = solve_triangular(LG, bG, lower=True)
    # logdet(I + A^-1 K_G) = logdet(A + K_G) - logdet A
    logdet = 2.0 * np.sum(np.log(np.diag(LG))) - np.sum(np.log(alpha_G))
    return 0.5 * float(c @ c) - 0.5 * logdet


@dataclass
class EvidenceFit:
    evidence: float
    log_alpha: np.ndarray  # per module, in the order given
    boundary: np.ndarray  # bool per module: driven to the exclusion bound
    n_starts: int
    converged: bool
    issues: list = field(default_factory=list)


GLOBAL_IMPROVEMENT_TOL = 1e-9  # an added start / refinement must beat the C3-start optimum by this many nats
COORD_GRID_STEP = 0.5  # log-alpha spacing of the per-module global line search


def _module_start_values(K: np.ndarray, b: np.ndarray, col_module: np.ndarray, n_modules: int,
                         lo: float, hi: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per-module scale-aware start values: the module's own mean column scale (log mean diag K_m) and the
    bounded empty-support singleton optima log(h^2/(q^2-h)) of its columns (Tipping-Faul, METHODS §1)."""
    dK = np.diag(K)
    own = np.empty(n_modules)
    cands = []
    for m in range(n_modules):
        cols = np.flatnonzero(col_module == m)
        own[m] = np.log(max(float(np.mean(dK[cols])), 1e-300))
        vals = [own[m]]
        for c in cols:
            h, q2 = float(dK[c]), float(b[c]) ** 2
            if h > 0 and q2 > h:
                vals.append(np.log(h * h / (q2 - h)))
        cands.append(np.unique(np.clip(vals, lo, hi)))
    return np.clip(own, lo, hi), cands


def optimize_evidence(
    K: np.ndarray, b: np.ndarray, col_module: np.ndarray, n_modules: int,
    warm_start: np.ndarray | None = None,
    bounds: tuple[float, float] = DEFAULT_LOG_ALPHA_BOUNDS,
    maxiter: int = 200,
) -> EvidenceFit:
    """Type-II maximisation of E over one log-precision per module (L-BFGS-B, deterministic starts).

    Starts of docs/PROTOCOL_CLARIFICATIONS.md C3: (i) the warm start if given, (ii) log alpha = 0,
    (iii) log alpha = log(median diag K). These alone return local optima when a module's prefix columns,
    or different modules, differ in scale by orders of magnitude (reports/CORRECTNESS_AUDIT.md, A1), so the
    protocol's *maximised* evidence is approached with further deterministic starts and a refinement:
    (iv) every module at its own column scale; (v) every module at its most / least inclusive singleton
    optimum; (vi) one module at each of its scale-aware values with the others at (v)-inclusive or excluded;
    then (vii) block-coordinate global line search (grid step 0.5 in log alpha over the YAML bounds) from the
    best point, polished by L-BFGS-B. The C3 optimum is kept unless (iv)-(vii) improve it by more than
    GLOBAL_IMPROVEMENT_TOL. No randomness is involved.
    """
    if K.shape[0] == 0:
        return EvidenceFit(0.0, np.zeros(0), np.zeros(0, bool), 0, True)
    lo, hi = bounds
    diag_med = float(np.median(np.diag(K)))
    starts = []
    if warm_start is not None:
        starts.append(np.clip(np.asarray(warm_start, dtype=np.float64), lo, hi))
    starts.append(np.zeros(n_modules))
    starts.append(np.full(n_modules, np.clip(np.log(max(diag_med, 1e-300)), lo, hi)))
    n_c3 = len(starts)
    own, cands = _module_start_values(K, b, col_module, n_modules, lo, hi)
    incl = np.array([c[0] for c in cands])
    excl = np.array([c[-1] for c in cands])
    extra = [own, incl, excl]
    for m in range(n_modules):
        for v in cands[m]:
            x = incl.copy()
            x[m] = v
            extra.append(x)
            x = np.full(n_modules, hi)
            x[m] = v
            extra.append(x)
    seen = {tuple(np.round(s, 12)) for s in starts}
    for x in extra:
        k = tuple(np.round(x, 12))
        if k not in seen:
            seen.add(k)
            starts.append(x)
    issues: list = []

    def f(x):
        E, g = evidence_and_grad(K, b, x, col_module, n_modules, True, issues)
        return -E, -g

    def value(x):
        return evidence_and_grad(K, b, x, col_module, n_modules, False, issues)[0]

    def lbfgs(x0):
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * n_modules,
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9})
        return -float(res.fun), np.asarray(res.x, dtype=np.float64), bool(res.success)

    best = None
    converged_any = False
    for i, x0 in enumerate(starts):
        val, x, ok = lbfgs(x0)
        if not np.isfinite(val):
            issues.append({"event": "nonfinite_evidence", "start": x0.tolist()})
            continue
        converged_any |= ok
        tol = 1e-12 if (best is None or i < n_c3) else GLOBAL_IMPROVEMENT_TOL
        if best is None or val > best[0] + tol:
            best = (val, x)
    if best is None:
        raise NumericalError("evidence optimisation failed for all starts")
    # (vii) block-coordinate global line search over each module's log-precision, then polish
    E, x = best
    grid = np.unique(np.concatenate([np.arange(lo, hi, COORD_GRID_STEP), [hi]]))
    xr, Er = x.copy(), E
    for _ in range(3):
        moved = False
        for m in range(n_modules):
            vals = []
            for gv in grid:
                xt = xr.copy()
                xt[m] = gv
                vals.append(value(xt))
            k = int(np.argmax(vals))
            if vals[k] > Er + GLOBAL_IMPROVEMENT_TOL:
                xr[m], Er, moved = grid[k], float(vals[k]), True
        if not moved:
            break
    if Er > E + GLOBAL_IMPROVEMENT_TOL:
        val, xp, ok = lbfgs(xr)
        if np.isfinite(val) and val >= Er:
            Er, xr = val, xp
        issues.append({"event": "coordinate_refinement_improved", "gain": Er - E})
        E, x = Er, xr
    if E < 0.0:  # exclusion of every module (alpha -> inf) attains ~0; report the bound value honestly
        issues.append({"event": "negative_optimum", "value": E})
    boundary = x >= hi - BOUNDARY_TOL
    return EvidenceFit(E, x, boundary, len(starts), converged_any, issues)


def ridge_quadratic(K: np.ndarray, b: np.ndarray, lam: float) -> float:
    """Regularised quadratic improvement 1/2 b^T (K + lam I)^{-1} b (conditional Newton score)."""
    d = K.shape[0]
    if d == 0:
        return 0.0
    L, _ = chol_with_jitter(K + lam * np.eye(d))
    c = solve_triangular(L, b, lower=True, check_finite=False)
    return 0.5 * float(c @ c)


def nologdet_quadratic(K: np.ndarray, b: np.ndarray, col_module: np.ndarray, n_modules: int,
                       bounds: tuple[float, float] = DEFAULT_LOG_ALPHA_BOUNDS) -> float:
    """Ablation objective: max over alpha of 1/2 b^T (K + A)^{-1} b without the logdet penalty.

    The quadratic term is monotonically decreasing in every alpha, so the maximiser is the lower
    bound alpha = exp(lo) for all modules (a near-unregularised least-squares improvement).
    """
    d = K.shape[0]
    if d == 0:
        return 0.0
    return ridge_quadratic(K, b, float(np.exp(bounds[0])))
