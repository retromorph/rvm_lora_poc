"""Gate evaluator: PASS / FAIL / INCONCLUSIVE / BLOCKED / NOT_RUN paths on synthetic metrics, bootstrap
determinism, and reproducibility of reports/gates.json from logged artifacts when present."""

import json
from pathlib import Path

import numpy as np
import pytest

from rvm_lora import gates as G

ROOT = Path(__file__).resolve().parents[1]


def _rows(cfg, rvm_good=True, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in cfg.architectures():
        U = float(rng.uniform(0, spread))
        noise = rng.normal(0, 1)
        rows.append({"arch_id": a.arch_id, "parameters": a.parameters, "U": U,
                     "group_rvm": U * 100 if rvm_good else -U * 100 + noise,
                     "conditional_newton": noise, "gradient_norm": noise * 2 + 0.1, "independent_fisher": -noise})
    return rows


def _pairs(cfg):
    out = []
    for a in cfg.architectures():
        if a.family == "two_layer_rank2":
            i, j = a.layers
            out.append((a.arch_id, f"L{i:02d}_qv_r2", f"L{j:02d}_qv_r2"))
    return out


def test_gate1_pass(cfg):
    rows = _rows(cfg, True)
    # make interactions informative with matching signs
    by = {r["arch_id"]: r for r in rows}
    for pid, a, b in _pairs(cfg):
        by[pid]["U"] = by[a]["U"] + by[b]["U"] + 0.02
        by[pid]["group_rvm"] = by[a]["group_rvm"] + by[b]["group_rvm"] + 1.0
    g = G.evaluate_gate1(rows, _pairs(cfg), {"fresh_activations": 4}, cfg["stage_a"])
    assert g["status"] == G.PASS, g["criteria"]


def test_gate1_fail_on_concordance(cfg):
    g = G.evaluate_gate1(_rows(cfg, False), _pairs(cfg), {"fresh_activations": 4}, cfg["stage_a"])
    assert g["status"] == G.FAIL


def test_gate1_fail_on_null(cfg):
    rows = _rows(cfg, True)
    g = G.evaluate_gate1(rows, _pairs(cfg), {"fresh_activations": 11}, cfg["stage_a"])
    assert [c for c in g["criteria"] if c["criterion"] == "fresh_null_activations"][0]["status"] == G.FAIL
    assert g["status"] == G.FAIL


def test_gate1_inconclusive_few_pairs(cfg):
    rows = _rows(cfg, True, spread=0.004)  # all |dU| < 0.005 -> no informative pairs
    by = {r["arch_id"]: r for r in rows}
    # keep top-3 margin positive so only the pair-count criterion is non-PASS
    g = G.evaluate_gate1(rows, _pairs(cfg), {"fresh_activations": 0}, cfg["stage_a"])
    crit = {c["criterion"]: c["status"] for c in g["criteria"]}
    assert crit["informative_cost_matched_pairs"] == G.INCONCLUSIVE
    assert crit["nonadditivity_sign_correct"] == G.NOT_EVALUABLE
    assert g["status"] in (G.INCONCLUSIVE, G.FAIL)
    assert by


def test_gate1_missing_runs(cfg):
    rows = _rows(cfg)
    rows[3]["U"] = None
    assert G.evaluate_gate1(rows, _pairs(cfg), None, cfg["stage_a"])["status"] == G.INCONCLUSIVE
    assert G.evaluate_gate1(rows, _pairs(cfg), None, cfg["stage_a"], "infrastructure")["status"] == G.BLOCKED


def test_concordance_ties_are_conservative():
    rows = [{"arch_id": "a", "parameters": 1, "U": 0.1, "s": 1.0}, {"arch_id": "b", "parameters": 1, "U": 0.0, "s": 1.0}]
    assert G.pairwise_concordance(rows, "s", 0.005)["concordance"] == 0.0


def _g2(cfg, rvm_shift=-0.02, n=200, newton_shift=0.0, seed=0):
    sb = cfg["stage_b"]
    rng = np.random.default_rng(seed)
    runs, per = {}, {}
    base = {t: rng.uniform(0.2, 0.5, n) for t in sb["tasks"]}
    shifts = {"uniform_r1": 0.0, "eva": 0.005, "oplora": 0.01, "conditional_newton": newton_shift,
              "group_rvm": rvm_shift, "uniform_r2": -0.015}
    for m in sb["methods"]:
        for t in sb["tasks"]:
            for s in sb["seeds"]:
                arr = base[t] + shifts[m] + rng.normal(0, 0.01, n)
                per[(m, t, s)] = arr
                runs[(m, t, s)] = {"status": "COMPLETED", "loss": float(arr.mean()), "accuracy": 0.9,
                                   "adapter_params": 73728 if m == "uniform_r2" else 36864,
                                   "selection_wall_s": 5.0 if m == "group_rvm" else 0.0, "train_wall_s": 100.0}
    struct = {(t, s): {"group_rvm": {"a": 1}, "conditional_newton": {"b": 1}} for t in sb["tasks"] for s in sb["seeds"]}
    fid = {m: True for m in sb["methods"]}
    return runs, per, struct, fid


def test_gate2_paths(cfg):
    sb = cfg["stage_b"]
    runs, per, st, fid = _g2(cfg)
    g = G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)
    assert g["status"] == G.PASS and g["route_a"]
    runs, per, st, fid = _g2(cfg, rvm_shift=0.01)
    assert G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)["status"] == G.FAIL
    runs, per, st, fid = _g2(cfg)
    fid["oplora"] = False
    assert G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)["status"] == G.INCONCLUSIVE
    runs, per, st, fid = _g2(cfg)
    runs[("eva", "qnli", 42)]["status"] = G.BLOCKED
    assert G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)["status"] == G.BLOCKED
    assert G.evaluate_gate2({}, sb, {}, {}, G.FAIL)["status"] == G.NOT_RUN
    # same structures as Newton -> specificity fails even with a better loss
    runs, per, st, fid = _g2(cfg)
    st = {k: {"group_rvm": {"a": 1}, "conditional_newton": {"a": 1}} for k in st}
    assert G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)["status"] == G.FAIL
    # overhead breach
    runs, per, st, fid = _g2(cfg)
    for k in runs:
        if k[0] == "group_rvm":
            runs[k]["selection_wall_s"] = 30.0
    assert G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)["status"] == G.FAIL


def test_gate2_route_b(cfg):
    sb = cfg["stage_b"]
    runs, per, st, fid = _g2(cfg, rvm_shift=-0.012, newton_shift=-0.004)
    # make RVM worse than the best B24 baseline by less than Route A needs, but near uniform_r2
    g = G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)
    assert g["route_b"] and g["status"] == G.PASS


def test_bootstrap_deterministic(cfg):
    sb = cfg["stage_b"]
    runs, per, st, fid = _g2(cfg)
    a = G.paired_bootstrap({(t, s): per[("group_rvm", t, s)] for t in sb["tasks"] for s in sb["seeds"]},
                           {(t, s): per[("uniform_r1", t, s)] for t in sb["tasks"] for s in sb["seeds"]},
                           sb["tasks"], sb["seeds"], 300, 991, 5.0)
    b = G.paired_bootstrap({(t, s): per[("group_rvm", t, s)] for t in sb["tasks"] for s in sb["seeds"]},
                           {(t, s): per[("uniform_r1", t, s)] for t in sb["tasks"] for s in sb["seeds"]},
                           sb["tasks"], sb["seeds"], 300, 991, 5.0)
    assert a == b and a["lower"] <= a["point"] <= a["upper"]


def test_overall_decision():
    assert G.overall_decision("PASS", "PASS", "PASS") == "GO_TO_3B_REPLICATION"
    assert G.overall_decision("PASS", "FAIL", "NOT_RUN_GATE_FAILURE") == "NO_GO"
    assert G.overall_decision("PASS", "INCONCLUSIVE", "NOT_RUN_GATE_FAILURE") == "INCONCLUSIVE"
    assert G.overall_decision("FAIL", "BLOCKED", "BLOCKED") == "INCONCLUSIVE"
    assert G.overall_decision("BLOCKED", "BLOCKED", "BLOCKED") == "BLOCKED"


@pytest.mark.post_hoc
def test_gates_json_reproducible_from_artifacts(cfg, record):
    """If real Gate-1 artifacts exist, recomputing the evaluator must reproduce reports/gates.json exactly."""
    gj = ROOT / "reports" / "gates.json"
    g1p = ROOT / "artifacts" / "gate1" / "gate1.json"
    if not (gj.exists() and g1p.exists()):
        pytest.skip("no logged Gate-1 artifacts yet")
    from rvm_lora.pipeline import gate1_rows

    rows, pairs, head = gate1_rows(cfg)
    sel = json.loads((ROOT / "artifacts" / "probes" / "qnli_s42" / "selection.json").read_text())
    g1 = G.evaluate_gate1(rows, pairs, {"fresh_activations": sel["group_rvm_fresh_null"]["activations"]},
                          cfg["stage_a"])
    stored = json.loads(gj.read_text())["gate1"]
    assert g1["status"] == stored["status"]
    for a, b in zip(g1["criteria"], stored["criteria"]):
        assert a["criterion"] == b["criterion"] and a["status"] == b["status"]
        assert a["value"] == b["value"]
    record(reproduced_gate1_status=g1["status"])
