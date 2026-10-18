"""Structure selectors over the fixed gradient-SVD dictionary (METHODS.md §3-4).

All selectors share the same Gram statistics K = Phi^T Phi and b = Phi^T t (float64), the same
candidate-group search space and the same cost model (rank_unit_params per rank unit).

Parallelism: optional fork-based process pools evaluate independent candidates. Every task carries
its full input (state + explicit warm start), so pooled and sequential runs give identical results.
Pools must only be created in processes that have not initialised CUDA (see `selection_job.py`).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .evidence import (
    DEFAULT_LOG_ALPHA_BOUNDS,
    nologdet_quadratic,
    optimize_evidence,
    ridge_quadratic,
    singleton_bounded,
)

Ranks = tuple[int, ...]
GAIN_TOL = 1e-9  # minimum evidence improvement for a backward deletion to be accepted


@dataclass(frozen=True)
class Group:
    key: str
    increments: tuple[tuple[int, int], ...]  # (module index, rank increment)

    @property
    def units(self) -> int:
        return sum(k for _, k in self.increments)


@dataclass
class Dictionary:
    """Gram statistics of the 24 x 8 rank-one dictionary. Column index = m * n_dir + j."""

    K: np.ndarray
    b: np.ndarray
    modules: list[str]
    n_dir: int
    unit_params: int

    @property
    def n_modules(self) -> int:
        return len(self.modules)

    def with_b(self, b: np.ndarray) -> "Dictionary":
        return Dictionary(self.K, np.asarray(b, dtype=np.float64), self.modules, self.n_dir, self.unit_params)

    def columns(self, ranks: Ranks) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """(column indices, local module index per column, active global module indices)."""
        cols, colmod, active = [], [], []
        for m, r in enumerate(ranks):
            if r > 0:
                local = len(active)
                active.append(m)
                for j in range(r):
                    cols.append(m * self.n_dir + j)
                    colmod.append(local)
        return np.asarray(cols, dtype=int), np.asarray(colmod, dtype=int), active

    def sub(self, ranks: Ranks) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        cols, colmod, active = self.columns(ranks)
        return self.K[np.ix_(cols, cols)], self.b[cols], colmod, active


def candidate_groups(modules: list[str]) -> list[Group]:
    """Fixed, deterministically ordered search space: +1/+2 per module, then a q+1,v+1 pair per layer."""
    groups: list[Group] = []
    for m, name in enumerate(modules):
        groups.append(Group(f"{name}:+1", ((m, 1),)))
        groups.append(Group(f"{name}:+2", ((m, 2),)))
    layers: dict[int, dict[str, int]] = {}
    for m, name in enumerate(modules):
        layer = int(name.split(".")[3])
        layers.setdefault(layer, {})[name.rsplit(".", 1)[1]] = m
    for layer in sorted(layers):
        q, v = layers[layer]["query"], layers[layer]["value"]
        groups.append(Group(f"layer{layer}:query+1,value+1", ((q, 1), (v, 1))))
    return groups


def apply(ranks: Ranks, g: Group) -> Ranks:
    r = list(ranks)
    for m, k in g.increments:
        r[m] += k
    return tuple(r)


def feasible(ranks: Ranks, g: Group, max_rank: int, budget_units: int) -> bool:
    if sum(ranks) + g.units > budget_units:
        return False
    return all(ranks[m] + k <= max_rank for m, k in g.increments)


# --------------------------------------------------------------------------- objectives
class Objective:
    """Cached set function value(ranks). `_value(ranks, warm)` must be a pure function of its inputs."""

    name = "abstract"
    allow_delete = False

    def __init__(self, D: Dictionary):
        self.D = D
        self.cache: dict[Ranks, tuple[float, dict]] = {}
        self.n_evals = 0

    def warm_for(self, ranks: Ranks, parent: Ranks | None):
        return None

    def value(self, ranks: Ranks, parent: Ranks | None = None) -> tuple[float, dict]:
        if ranks not in self.cache:
            self.n_evals += 1
            self.cache[ranks] = self._value(ranks, self.warm_for(ranks, parent))
        return self.cache[ranks]

    def prefetch(self, states: list[Ranks], parent: Ranks, pool) -> None:
        todo = [s for s in dict.fromkeys(states) if s not in self.cache]
        if not todo or pool is None:
            return
        tasks = [(s, self.warm_for(s, parent)) for s in todo]
        for s, out in zip(todo, pool.map(_pool_eval, tasks, chunksize=max(1, len(tasks) // 32))):
            self.n_evals += 1
            self.cache[s] = out

    def _value(self, ranks: Ranks, warm) -> tuple[float, dict]:
        raise NotImplementedError

    def boundary_modules(self, ranks: Ranks) -> list[int]:
        return []


class RVMObjective(Objective):
    """Globally re-optimised grouped-ARD log evidence max_alpha E(S, alpha)."""

    name = "group_rvm"
    allow_delete = True

    def __init__(self, D: Dictionary, bounds: tuple[float, float] = DEFAULT_LOG_ALPHA_BOUNDS):
        super().__init__(D)
        self.bounds = bounds
        self.issues: list = []

    def warm_for(self, ranks, parent):
        if parent is None or parent not in self.cache:
            return None
        pa = self.cache[parent][1]["log_alpha"]
        return {m: a for m, a in pa.items() if ranks[m] > 0}

    def _value(self, ranks, warm):
        K, b, colmod, active = self.D.sub(ranks)
        if not active:
            return 0.0, {"log_alpha": {}, "boundary": [], "converged": True, "issues": []}
        if K.shape[0] == 1:  # exact closed form for a single column (bounded Tipping-Faul optimum)
            E, la = singleton_bounded(float(b[0]), float(K[0, 0]), self.bounds)
            bnd = [active[0]] if la >= self.bounds[1] - 1e-6 else []
            return E, {"log_alpha": {active[0]: la}, "boundary": bnd, "converged": True, "issues": []}
        x0 = None
        if warm:
            diag_med = float(np.median(np.diag(K)))
            default = float(np.clip(np.log(max(diag_med, 1e-300)), *self.bounds))
            x0 = np.array([warm.get(m, default) for m in active])
        fit = optimize_evidence(K, b, colmod, len(active), warm_start=x0, bounds=self.bounds)
        la = {m: float(x) for m, x in zip(active, fit.log_alpha)}
        bnd = [m for m, flag in zip(active, fit.boundary) if flag]
        return fit.evidence, {"log_alpha": la, "boundary": bnd, "converged": fit.converged,
                              "issues": [{**i, "ranks": list(ranks)} for i in fit.issues]}

    def boundary_modules(self, ranks):
        return list(self.value(ranks)[1]["boundary"])

    def all_issues(self) -> list:
        return [i for _, aux in self.cache.values() for i in aux.get("issues", [])]


class NewtonObjective(Objective):
    """Conditional Newton: ridge-regularised quadratic improvement, no logdet complexity term."""

    name = "conditional_newton"

    def __init__(self, D: Dictionary, lam: float):
        super().__init__(D)
        self.lam = float(lam)

    def _value(self, ranks, warm):
        K, b, _, _ = self.D.sub(ranks)
        return ridge_quadratic(K, b, self.lam), {}


class NoLogdetObjective(Objective):
    """Offline ablation: RVM objective with the logdet penalty removed (alpha -> lower bound)."""

    name = "rvm_no_logdet"

    def __init__(self, D: Dictionary, bounds: tuple[float, float] = DEFAULT_LOG_ALPHA_BOUNDS):
        super().__init__(D)
        self.bounds = bounds

    def _value(self, ranks, warm):
        K, b, colmod, active = self.D.sub(ranks)
        return nologdet_quadratic(K, b, colmod, len(active), self.bounds), {}


# --------------------------------------------------------------------------- fork-pool plumbing
_WORKER: dict = {}


def _pool_eval(task):
    ranks, warm = task
    return _WORKER["obj"]._value(ranks, warm)


def make_pool(n_workers: int, **state):
    """Fork a pool *after* publishing `state` so workers inherit it. Returns None if n_workers <= 1."""
    _WORKER.clear()
    _WORKER.update(state)
    if n_workers <= 1:
        return None
    return mp.get_context("fork").Pool(n_workers)


# --------------------------------------------------------------------------- greedy search
@dataclass
class SelectionResult:
    selector: str
    ranks: Ranks
    objective_value: float
    stop_reason: str
    threshold: float
    trace: list = field(default_factory=list)
    wall_s: float = 0.0
    n_objective_evals: int = 0

    def rank_dict(self, modules: list[str]) -> dict[str, int]:
        return {modules[m]: r for m, r in enumerate(self.ranks) if r > 0}

    @property
    def rank_units(self) -> int:
        return int(sum(self.ranks))


def initial_gains(obj: Objective, groups: list[Group], max_rank: int, budget_units: int) -> np.ndarray:
    """Gain per parameter of every feasible group from empty support (used for null calibration)."""
    zero: Ranks = tuple([0] * obj.D.n_modules)
    base, _ = obj.value(zero)
    out = []
    for g in groups:
        if not feasible(zero, g, max_rank, budget_units):
            continue
        v, _ = obj.value(apply(zero, g), parent=zero)
        out.append((v - base) / (g.units * obj.D.unit_params))
    return np.asarray(out, dtype=np.float64)


def greedy_select(
    obj: Objective, groups: list[Group], max_rank: int, budget_units: int, threshold: float,
    max_iter: int = 200, record_trace: bool = True, n_workers: int = 0,
) -> SelectionResult:
    """Forward addition by best gain-per-parameter above `threshold`; for RVM also boundary pruning
    and one backward-delete sweep per iteration; cycle detection and an iteration limit."""
    t0 = time.perf_counter()
    pool = make_pool(n_workers, obj=obj) if n_workers > 1 else None
    try:
        res = _greedy(obj, groups, max_rank, budget_units, threshold, max_iter, record_trace, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    res.wall_s = time.perf_counter() - t0
    return res


def _greedy(obj, groups, max_rank, budget_units, threshold, max_iter, record_trace, pool) -> SelectionResult:
    D = obj.D
    ranks: Ranks = tuple([0] * D.n_modules)
    cur, _ = obj.value(ranks)
    visited = {ranks}
    trace: list = []
    stop = "iteration_limit"
    for it in range(max_iter):
        feas = [(order, g) for order, g in enumerate(groups) if feasible(ranks, g, max_rank, budget_units)]
        if not feas:
            stop = "budget_exhausted"
            break
        obj.prefetch([apply(ranks, g) for _, g in feas], ranks, pool)
        best = None
        for order, g in feas:
            new = apply(ranks, g)
            v, _ = obj.value(new, parent=ranks)
            gain_pp = (v - cur) / (g.units * D.unit_params)
            if best is None or gain_pp > best[0]:  # strict '>' keeps the earliest group on ties
                best = (gain_pp, order, g, new, v)
        gain_pp, _, g, new, v = best
        step = {"iter": it, "best_group": g.key, "gain_per_param": gain_pp, "threshold": threshold,
                "objective_before": cur}
        if not gain_pp > threshold:
            step["action"] = "stop_threshold"
            trace.append(step)
            stop = "null_threshold"
            break
        ranks, cur = new, v
        step.update(action="add", objective_after_add=cur)
        if obj.allow_delete:
            pruned = obj.boundary_modules(ranks)
            if pruned:
                r = list(ranks)
                for m in pruned:
                    r[m] = 0
                ranks = tuple(r)
                cur, _ = obj.value(ranks)
                step["boundary_pruned"] = [D.modules[m] for m in pruned]
            obj.prefetch([tuple(r - (i == m) for i, r in enumerate(ranks))
                          for m in range(D.n_modules) if ranks[m] > 0], ranks, pool)
            deletions = []
            for m in range(D.n_modules):  # one backward-delete sweep in canonical order
                if ranks[m] == 0:
                    continue
                r = list(ranks)
                r[m] -= 1
                cand = tuple(r)
                v2, _ = obj.value(cand, parent=ranks)
                if v2 > cur + GAIN_TOL:
                    deletions.append({"module": D.modules[m], "objective_gain": v2 - cur})
                    ranks, cur = cand, v2
            if deletions:
                step["deletions"] = deletions
            if isinstance(obj, RVMObjective):
                step["log_alpha"] = {D.modules[m]: a for m, a in obj.value(ranks)[1]["log_alpha"].items()}
        step["ranks_after"] = list(ranks)
        step["objective_after_sweep"] = cur
        if record_trace:
            trace.append(step)
        if ranks in visited:
            stop = "cycle_detected"
            break
        visited.add(ranks)
    return SelectionResult(obj.name, ranks, cur, stop, threshold, trace, 0.0, obj.n_evals)


# --------------------------------------------------------------------------- null calibration
def null_responses(n: int, count: int, seed: int) -> np.ndarray:
    """count x n matrix of N(0, I) pseudo-responses from numpy PCG64 with the given seed."""
    return np.random.default_rng(seed).standard_normal((count, n))


def _null_max_gain(i: int) -> float:
    W = _WORKER
    obj = W["make_obj"](W["D"].with_b(W["Phi"].T @ W["T"][i]))
    return float(initial_gains(obj, W["groups"], W["max_rank"], W["budget"]).max())


def _null_full_run(i: int) -> int:
    W = _WORKER
    obj = W["make_obj"](W["D"].with_b(W["Phi"].T @ W["T"][i]))
    return greedy_select(obj, W["groups"], W["max_rank"], W["budget"], W["threshold"],
                         record_trace=False).rank_units


def _map_nulls(fn, n_workers: int, count: int, **state) -> list:
    pool = make_pool(n_workers, **state)
    try:
        if pool is None:
            return [fn(i) for i in range(count)]
        return pool.map(fn, range(count), chunksize=max(1, count // (4 * n_workers)))
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def calibrate_threshold(make_obj: Callable[[Dictionary], Objective], Phi: np.ndarray, D: Dictionary,
                        groups: list[Group], max_rank: int, budget_units: int, count: int, seed: int,
                        percentile: float, n_workers: int = 0) -> tuple[float, np.ndarray, float]:
    """Empirical `percentile` of the maximum initial gain-per-parameter over `count` null responses."""
    t0 = time.perf_counter()
    T = null_responses(Phi.shape[0], count, seed)
    maxima = np.asarray(_map_nulls(_null_max_gain, n_workers, count, make_obj=make_obj, D=D, Phi=Phi, T=T,
                                   groups=groups, max_rank=max_rank, budget=budget_units), dtype=np.float64)
    thr = float(np.percentile(maxima, percentile))  # numpy default 'linear' interpolation
    return thr, maxima, time.perf_counter() - t0


def evaluate_null(make_obj: Callable[[Dictionary], Objective], Phi: np.ndarray, D: Dictionary,
                  groups: list[Group], max_rank: int, budget_units: int, threshold: float,
                  count: int, seed: int, n_workers: int = 0) -> tuple[int, list, float]:
    """Run the full selector on `count` fresh null responses; count activations (>=1 module selected)."""
    t0 = time.perf_counter()
    T = null_responses(Phi.shape[0], count, seed)
    units = _map_nulls(_null_full_run, n_workers, count, make_obj=make_obj, D=D, Phi=Phi, T=T, groups=groups,
                       max_rank=max_rank, budget=budget_units, threshold=threshold)
    return int(sum(u > 0 for u in units)), [int(u) for u in units], time.perf_counter() - t0


# --------------------------------------------------------------------------- ridge CV
def select_ridge_lambda(Phi: np.ndarray, t: np.ndarray, grid: list[float], folds: int) -> tuple[float, dict]:
    """3-fold CV of quadratic-surrogate predictive error inside the evidence set only.

    Rows are assigned to folds by `row_index % folds` (the evidence rows are already in the frozen
    random split order); Phi and t are fixed (computed once with the fixed head). The model is
    ridge regression of t on all dictionary columns. Ties go to the larger lambda.
    """
    n = Phi.shape[0]
    fold = np.arange(n) % folds
    errors = {}
    for lam in grid:
        errs = []
        for f in range(folds):
            tr, te = fold != f, fold == f
            Ktr = Phi[tr].T @ Phi[tr]
            btr = Phi[tr].T @ t[tr]
            s = np.linalg.solve(Ktr + lam * np.eye(Ktr.shape[0]), btr)
            r = t[te] - Phi[te] @ s
            errs.append(float(r @ r) / int(te.sum()))
        errors[float(lam)] = float(np.mean(errs))
    best = min(sorted(errors, reverse=True), key=lambda k: errors[k])
    return best, {"cv_mse": errors, "folds": folds, "fold_rule": "row_index % folds", "selected": best}


# --------------------------------------------------------------------------- Gate-1 scores
def architecture_scores(D: Dictionary, ranks: Ranks, lam: float,
                        bounds: tuple[float, float] = DEFAULT_LOG_ALPHA_BOUNDS) -> dict:
    """Four precomputed scores for a full fixed architecture (METHODS §4)."""
    K, b, colmod, active = D.sub(ranks)
    fit = optimize_evidence(K, b, colmod, len(active), bounds=bounds)
    return {
        "gradient_norm": float(b @ b),
        "independent_fisher": float(0.5 * np.sum(b * b / (np.diag(K) + lam))),
        "conditional_newton": ridge_quadratic(K, b, lam),
        "group_rvm": fit.evidence,
        "rvm_log_alpha": [float(x) for x in fit.log_alpha],
        "rvm_boundary": [bool(x) for x in fit.boundary],
        "rvm_converged": bool(fit.converged),
        "rvm_issues": fit.issues,
    }
