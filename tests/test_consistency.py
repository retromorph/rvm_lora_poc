"""Post-hoc cross-file consistency: YAML thresholds <-> gates.json cutoffs, CSVs <-> evaluator inputs,
run index <-> MLflow metrics, explicit status for every planned run (no zero-filled scores)."""

import csv
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.post_hoc
LIVE_URI = os.environ.get("MLFLOW_TRACKING_URI")


def _need(*paths):
    for p in paths:
        if not (ROOT / p).exists():
            pytest.skip(f"{p} not produced yet")


def _csv(p):
    with open(ROOT / p) as f:
        return list(csv.DictReader(f))


def test_thresholds_match_yaml(cfg):
    _need("reports/gates.json")
    g = json.loads((ROOT / "reports/gates.json").read_text())
    sa = cfg["stage_a"]
    want = {"informative_cost_matched_pairs": sa["min_informative_cost_matched_pairs"],
            "rvm_pairwise_concordance": sa["min_rvm_concordance"],
            "rvm_concordance_minus_best_nonbayes": sa["min_concordance_margin_over_best_nonbayes"],
            "rvm_six_choice_top3_utility_margin_nats": sa["min_six_choice_mean_utility_margin_nats"],
            "fresh_null_activations": sa["max_fresh_null_activations"],
            "nonadditivity_sign_correct": sa["conditional_interactions"]["min_correct_if_rule_applicable"]}
    for c in g["gate1"].get("criteria", []):
        if c["criterion"] in want:
            assert c["cutoff"] == want[c["criterion"]], c["criterion"]
    assert g["meta"]["config_sha256"] == cfg.sha256


def test_architecture_csv_matches_gate1(cfg):
    _need("reports/architecture_scores.csv", "artifacts/gate1/gate1.json")
    rows = {r["arch_id"]: r for r in _csv("reports/architecture_scores.csv")}
    g1 = json.loads((ROOT / "artifacts/gate1/gate1.json").read_text())
    assert len(rows) == 20 == len(g1["rows"])
    for r in g1["rows"]:
        c = rows[r["arch_id"]]
        for k in ("gradient_norm", "independent_fisher", "conditional_newton", "group_rvm", "U", "audit_loss"):
            if r[k] is None:
                assert c[k] == "", "missing values must stay empty, never zero-filled"
            else:
                assert float(c[k]) == r[k], (r["arch_id"], k)


def test_run_index_complete_and_explicit(cfg):
    _need("reports/run_index.csv", "reports/gates.json")
    rows = _csv("reports/run_index.csv")
    assert len(rows) == cfg["maximum_full_adapter_training_runs"] == 44
    allowed = {"COMPLETED", "FAILED", "RUNNING", "NOT_RUN_GATE_FAILURE", "BLOCKED", "PLANNED_NOT_STARTED"}
    for r in rows:
        assert r["status"] in allowed
        if r["status"] != "COMPLETED":
            assert r["eval_loss"] == "", "non-completed runs must not carry a score"
    g = json.loads((ROOT / "reports/gates.json").read_text())
    if g["gate1"]["status"] != "PASS":
        assert all(r["status"] == "NOT_RUN_GATE_FAILURE" for r in rows if r["gate"] == "gate2")


@pytest.mark.skipif(not (LIVE_URI or "").startswith("http"), reason="needs the live MLflow server")
def test_run_index_matches_mlflow(cfg):
    _need("reports/run_index.csv")
    from rvm_lora.tracking import Tracker

    tr = Tracker(LIVE_URI)
    n = 0
    for r in _csv("reports/run_index.csv"):
        if r["status"] != "COMPLETED":
            continue
        run = tr.run_info(r["mlflow_run_id"])
        key = "audit_loss" if r["gate"] == "gate1" else "official_validation_loss"
        assert run.data.metrics[key] == float(r["eval_loss"])
        assert run.data.tags["status"] == "COMPLETED" and run.info.status == "FINISHED"
        assert int(run.data.metrics["adapter_params"]) == int(float(r["adapter_params"]))
        n += 1
    assert n > 0
