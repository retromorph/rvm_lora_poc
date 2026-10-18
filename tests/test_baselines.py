"""OPLoRA stage-1 ILP port: exact DP vs brute force and scipy MILP; budget/coverage constraints."""

import itertools

import numpy as np

from rvm_lora.baselines import oplora_allocate, oplora_gain, oplora_milp, uniform_ranks

MODS = [f"roberta.encoder.layer.{l}.attention.self.{p}" for l in range(12) for p in ("query", "value")]


def _brute(imp, budget, R):
    names = list(imp)
    best = None
    for combo in itertools.product(R, repeat=len(names)):
        if sum(combo) > budget:
            continue
        if not any(r for n, r in zip(names, combo) if n.endswith("query")):
            continue
        if not any(r for n, r in zip(names, combo) if n.endswith("value")):
            continue
        v = sum(oplora_gain(imp[n], r) for n, r in zip(names, combo))
        if best is None or v > best[0]:
            best = (v, combo)
    return {n: r for n, r in zip(names, best[1]) if r}, best[0]


def test_dp_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        names = MODS[:5]
        imp = {n: float(rng.lognormal()) for n in names}
        a, va = oplora_allocate(imp, 6, [0, 1, 2, 3])
        b, vb = _brute(imp, 6, [0, 1, 2, 3])
        assert np.isclose(va, vb, rtol=1e-12) and a == b


def test_dp_matches_milp_full_size(record):
    rng = np.random.default_rng(1)
    agree = 0
    for _ in range(5):
        imp = {n: float(rng.lognormal(sigma=0.5)) for n in MODS}
        a, va = oplora_allocate(imp, 24, list(range(9)))
        b, vb = oplora_milp(imp, 24, list(range(9)))
        assert np.isclose(va, vb, rtol=1e-9)
        assert sum(a.values()) <= 24 and max(a.values()) <= 8
        assert any(n.endswith("query") for n in a) and any(n.endswith("value") for n in a)
        agree += a == b
    record(milp_assignment_agreement=agree, cases=5)


def test_uniform():
    assert sum(uniform_ranks(MODS, 1).values()) * 1536 == 36864
    assert sum(uniform_ranks(MODS, 2).values()) * 1536 == 73728
