"""External baselines: uniform LoRA, PEFT-native EVA and the OPLoRA (DPLoRA stage-1) ILP allocation.

OPLoRA reference: https://github.com/cjpark-ai/DPLoRA @ 6666d8bdb37ebd153a8916b0464c8a3878976aae (MIT),
`pruning/stage1/initial_rank_allocation.py`:
  importance I_l = (1/N) sum_n mean_ij (dL_n/dW_l)_ij^2           (measure_layer_importance)
  gain(l, r)    = I_l * (1 - exp(-r / min(d_in, d_out)))          (estimate_performance_gain)
  cost(r)       = r * (d_in + d_out)                              (pruning_utils.calculate_cost)
  ILP: one rank per module, total cost <= budget, every module type gets >= 1 rank; maximise sum gain.
The reference solves the ILP with PuLP/CBC at zero gap; we solve the same ILP exactly by dynamic
programming (verified against brute force and scipy.optimize.milp in tests).
"""

from __future__ import annotations

import math
import time

import numpy as np

OPLORA_REPO = "https://github.com/cjpark-ai/DPLoRA"
OPLORA_SHA = "6666d8bdb37ebd153a8916b0464c8a3878976aae"
OPLORA_LICENSE = "MIT"


def uniform_ranks(modules: list[str], r: int) -> dict[str, int]:
    return {m: r for m in modules}


def module_type(name: str) -> str:
    return name.rsplit(".", 1)[1]


def oplora_gain(importance: float, r: int, min_dim: int = 768) -> float:
    return importance * (1.0 - math.exp(-r / min_dim))


def oplora_allocate(importance: dict[str, float], budget_units: int, rank_choices: list[int],
                    min_dim: int = 768) -> tuple[dict[str, int], float]:
    """Exact ILP optimum by DP over (modules, used units, type-coverage mask).

    Ties (exactly equal objective) are broken towards the lexicographically earlier assignment in the
    canonical module order with smaller ranks first; exact ties have probability ~0 with real data.
    """
    names = list(importance)
    types = sorted({module_type(n) for n in names})
    tbit = {t: 1 << i for i, t in enumerate(types)}
    full = (1 << len(types)) - 1
    # dp[(units, mask)] = (value, assignment tuple)
    dp: dict[tuple[int, int], tuple[float, tuple]] = {(0, 0): (0.0, ())}
    for n in names:
        nxt: dict[tuple[int, int], tuple[float, tuple]] = {}
        for (u, mask), (val, assign) in dp.items():
            for r in rank_choices:
                if u + r > budget_units:
                    continue
                key = (u + r, mask | (tbit[module_type(n)] if r >= 1 else 0))
                cand = (val + oplora_gain(importance[n], r, min_dim), assign + (r,))
                if key not in nxt or cand[0] > nxt[key][0] + 1e-15 * max(1.0, abs(cand[0])):
                    nxt[key] = cand
        dp = nxt
    finals = [(v, a) for (u, mask), (v, a) in dp.items() if mask == full]
    if not finals:
        raise ValueError("OPLoRA ILP infeasible")
    best = max(finals, key=lambda x: x[0])
    return {n: r for n, r in zip(names, best[1]) if r > 0}, best[0]


def oplora_milp(importance: dict[str, float], budget_units: int, rank_choices: list[int], min_dim: int = 768):
    """Reference MILP (scipy/HiGHS) used only in tests to cross-check the DP."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    names = list(importance)
    R = list(rank_choices)
    nv = len(names) * len(R)
    c = -np.array([oplora_gain(importance[n], r, min_dim) for n in names for r in R])
    A, lo, hi = [], [], []
    for i in range(len(names)):
        row = np.zeros(nv)
        row[i * len(R):(i + 1) * len(R)] = 1
        A.append(row), lo.append(1), hi.append(1)
    A.append(np.array([r for _ in names for r in R], dtype=float)), lo.append(0), hi.append(budget_units)
    for t in sorted({module_type(n) for n in names}):
        row = np.array([r if module_type(n) == t else 0 for n in names for r in R], dtype=float)
        A.append(row), lo.append(1), hi.append(np.inf)
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1))
    x = np.round(res.x).astype(int)
    ranks = {}
    for i, n in enumerate(names):
        r = R[int(np.argmax(x[i * len(R):(i + 1) * len(R)]))]
        if r > 0:
            ranks[n] = r
    return ranks, -res.fun


def eva_initialize(peft_model, direction_batches: list[dict], show_progress_bar: bool = False) -> dict:
    """PEFT-native EVA (explained-variance rank redistribution + SVD init of lora_A).

    Batches contain only input_ids/attention_mask (no `labels`), so EVA's token mask is the attention
    mask; sequence-level labels would otherwise be broadcast against tokens by the default
    `prepare_model_inputs_fn_language_modeling`.
    """
    from peft import initialize_lora_eva_weights

    t0 = time.perf_counter()
    initialize_lora_eva_weights(peft_model, dataloader=direction_batches, show_progress_bar=show_progress_bar)
    return {"eva_wall_s": time.perf_counter() - t0}
