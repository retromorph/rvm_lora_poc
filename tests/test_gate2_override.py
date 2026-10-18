"""Exploratory Gate-2 override: without the explicit override a non-PASS Gate 1 still blocks Gate 2 exactly as
before; with it, the unchanged criteria are computed, labelled counterfactual, and the Gate-1 record, its status
and the preregistered decision are left untouched."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rvm_lora import cli
from rvm_lora import config as C
from rvm_lora import gates as G
from rvm_lora import pipeline as P
from rvm_lora import tracking
from rvm_lora.config import sha256_file
from rvm_lora.tracking import mirror_dir, run_key, write_json

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(cfg, seed=0, n=50):
    sb = cfg["stage_b"]
    rng = np.random.default_rng(seed)
    runs, per = {}, {}
    shifts = {"uniform_r1": 0.0, "eva": 0.005, "oplora": 0.01, "conditional_newton": 0.0, "group_rvm": -0.02,
              "uniform_r2": -0.015}
    base = {t: rng.uniform(0.2, 0.5, n) for t in sb["tasks"]}
    for m in sb["methods"]:
        for t in sb["tasks"]:
            for s in sb["seeds"]:
                arr = base[t] + shifts[m] + rng.normal(0, 0.01, n)
                per[(m, t, s)] = arr
                runs[(m, t, s)] = {"status": "COMPLETED", "loss": float(arr.mean()), "accuracy": 0.9,
                                   "adapter_params": 73728 if m == "uniform_r2" else 36864,
                                   "selection_wall_s": 5.0 if m == "group_rvm" else 0.0, "train_wall_s": 100.0}
    struct = {(t, s): {"group_rvm": {"a": 1}, "conditional_newton": {"b": 1}} for t in sb["tasks"] for s in sb["seeds"]}
    return runs, per, struct, {m: True for m in sb["methods"]}


# ------------------------------------------------------------------------------------------ evaluator
def test_failed_gate1_still_blocks_gate2_even_with_complete_runs(cfg):
    runs, per, st, fid = _synthetic(cfg)
    for g1 in (G.FAIL, G.INCONCLUSIVE, G.BLOCKED):
        g = G.evaluate_gate2(runs, cfg["stage_b"], fid, st, g1, per)
        assert g["status"] == G.NOT_RUN and g["criteria"] == []


def test_preregistered_gate2_unchanged_by_refactor(cfg):
    runs, per, st, fid = _synthetic(cfg)
    assert G.evaluate_gate2(runs, cfg["stage_b"], fid, st, G.PASS, per) == \
        G.gate2_criteria(runs, cfg["stage_b"], fid, st, per)


def test_exploratory_evaluation_is_counterfactual_and_labelled(cfg):
    sb = cfg["stage_b"]
    runs, per, st, fid = _synthetic(cfg)
    real = G.evaluate_gate2(runs, sb, fid, st, G.PASS, per)
    ex = G.evaluate_gate2_exploratory(runs, sb, fid, st, G.FAIL, per)
    assert "status" not in ex  # cannot be mistaken for, or assembled as, the preregistered Gate-2 status
    assert ex["execution_type"] == G.EXPLORATORY_OVERRIDE and ex["gate1_status"] == G.FAIL
    assert ex["preregistered_gate2_status"] == G.NOT_RUN
    assert ex["exploratory_gate2_standalone_result"] == real["status"]
    assert [c["value"] for c in ex["criteria"]] == [c["value"] for c in real["criteria"]]
    assert [c["status"] for c in ex["criteria"]] == [c["status"] for c in real["criteria"]]
    assert all(c["execution_type"] == G.EXPLORATORY_OVERRIDE for c in ex["criteria"])
    assert ex["bootstrap"] == real["bootstrap"]
    with pytest.raises(KeyError):
        G.assemble({"status": G.PASS}, {"status": G.FAIL}, ex, {})
    # whatever the counterfactual says, a failed Gate 1 keeps the preregistered decision at NO_GO
    assert G.overall_decision(G.PASS, G.FAIL, G.NOT_RUN) == "NO_GO"


# ------------------------------------------------------------------------------------------ pipeline
@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Exploratory namespace `exp` in a temp root, with a reference namespace `ref` holding a FAIL Gate 1."""
    for mod in (C, P, tracking):
        monkeypatch.setattr(mod, "RUN_NAMESPACE", "exp")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "ART", tmp_path / "artifacts" / "exp")
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "baseline_fidelity.json", tmp_path / "configs" / "baseline_fidelity.json")
    rec = tmp_path / "artifacts" / "ref" / "gate1" / "gate1.json"
    write_json(rec, {"gate1": {"status": "FAIL", "criteria": []}})
    return tmp_path, rec


def _forbid(*_a, **_k):
    raise AssertionError("Gate-2 work started without a passing Gate 1 or an explicit override")


def test_train_b_refuses_failed_gate1_without_override(cfg, sandbox, monkeypatch):
    write_json(P.ART / "gate1" / "gate1.json", {"gate1": {"status": "FAIL"}})
    for f in ("warm_head", "probe", "adapter_run", "_record_empty", "ensure_validation"):
        monkeypatch.setattr(P, f, _forbid)
    with pytest.raises(RuntimeError, match="Gate 2 must not run"):
        P.train_b(cfg, tracker=None)


def test_train_b_refuses_missing_gate1_without_override(cfg, sandbox, monkeypatch):
    monkeypatch.setattr(P, "warm_head", _forbid)
    with pytest.raises(FileNotFoundError):  # this namespace has no Gate 1 of its own
        P.train_b(cfg, tracker=None)


def test_override_needs_its_own_namespace(sandbox, monkeypatch):
    root, rec = sandbox
    monkeypatch.setattr(P, "ART", rec.parents[1])  # same namespace as the record
    with pytest.raises(RuntimeError, match="own RVM_RUN_NAMESPACE"):
        P.gate1_override(rec)
    monkeypatch.setattr(P, "ART", root / "artifacts" / "exp")
    monkeypatch.setattr(P, "RUN_NAMESPACE", "")
    with pytest.raises(RuntimeError, match="own RVM_RUN_NAMESPACE"):
        P.gate1_override(rec)


def test_override_carries_fail_and_leaves_record_untouched(sandbox):
    _, rec = sandbox
    before = sha256_file(rec)
    ov = P.gate1_override(rec)
    assert ov["gate1_status"] == "FAIL" and ov["execution_type"] == "exploratory_override"
    assert ov["stage"] == "gate2" and ov["gate1_record_sha256"] == before == sha256_file(rec)
    assert ov["reference_artifacts"] == "artifacts/ref"


def test_train_b_with_override_tags_every_stage(cfg, sandbox, monkeypatch):
    _, rec = sandbox
    ov = P.gate1_override(rec)
    calls = []
    monkeypatch.setattr(P, "warm_head", lambda *a, **k: calls.append(("head", a[1:3], k)))
    monkeypatch.setattr(P, "probe", lambda *a, **k: calls.append(("probe", a[1:3], k)))
    monkeypatch.setattr(P, "verify_reference", lambda t, s, ref: {"task": t, "seed": s, "ref": str(ref)})
    monkeypatch.setattr(P, "ensure_validation", lambda *a: None)
    monkeypatch.setattr(P, "gate2_methods_ranks", lambda c, t, s: {m: ({"x": 1}, 1.0) for m in c["stage_b"]["methods"]})
    monkeypatch.setattr(P, "adapter_run", lambda *a, **k: calls.append(("run", a[3:6], k)) or {})
    P.train_b(cfg, tracker=None, override=ov)
    heads = [c for c in calls if c[0] == "head"]
    runs = [c for c in calls if c[0] == "run"]
    assert len(heads) == 4 and all(c[2]["gate"] == "gate2" and c[2]["extra_tags"] is ov for c in heads)
    assert all(c[2]["extra_tags"]["gate1_status"] == "FAIL" for c in calls)
    assert len(runs) == 24 and all(c[2]["extra_tags"]["execution_type"] == "exploratory_override" for c in runs)
    chk = json.loads((P.ART / "gate2" / "reference_check.json").read_text())
    assert len(chk["checks"]) == 4 and chk["override"]["gate1_status"] == "FAIL"


def test_verify_reference_detects_head_mismatch(sandbox):
    root, _ = sandbox
    ref = root / "artifacts" / "ref"
    write_json(ref / "heads" / "qnli_s42.json", {"head_checkpoint_sha256": "A", "head_ids_sha256": "I",
                                                 "head_examples": 2000})
    write_json(P.head_meta_path("qnli", 42), {"head_checkpoint_sha256": "A", "head_ids_sha256": "I",
                                              "head_examples": 2000})
    assert P.verify_reference("qnli", 42, ref)["all_equal"] is True
    assert P.verify_reference("sst2", 42, ref)["reference"] is None  # nothing to reproduce
    write_json(P.head_meta_path("qnli", 42), {"head_checkpoint_sha256": "B", "head_ids_sha256": "I",
                                              "head_examples": 2000})
    with pytest.raises(P.ReferenceMismatchError):
        P.verify_reference("qnli", 42, ref)


def _write_logged_gate2(cfg, root):
    runs, per, st, _ = _synthetic(cfg)
    sb = cfg["stage_b"]
    for t in sb["tasks"]:
        for s in sb["seeds"]:
            write_json(P.probe_dir(t, s) / "selection.json",
                       {"group_rvm": {"ranks": {"a": 1}}, "conditional_newton": {"ranks": {"b": 1}},
                        "rvm_no_logdet": {"ranks": {}}})
    for (m, t, s), r in runs.items():
        key = run_key(cfg.sha256, "gate2", m, t, s, m)
        d = mirror_dir(root, "gate2", key)
        write_json(d / "result.json", {"status": "COMPLETED", "metrics": {
            "official_validation_loss": r["loss"], "official_validation_accuracy": r["accuracy"],
            "adapter_params": r["adapter_params"], "selection_wall_s": r["selection_wall_s"],
            "train_wall_s": r["train_wall_s"]}})
        np.savez(d / "eval_validation.npz", loss=per[(m, t, s)])
    return runs, per, st


def test_eval_gate2_without_override_stays_not_run(cfg, sandbox):
    root, _ = sandbox
    _write_logged_gate2(cfg, root)
    write_json(P.ART / "gate1" / "gate1.json", {"gate1": {"status": "FAIL"}})
    g = P.eval_gate2(cfg)
    assert g["status"] == G.NOT_RUN
    assert not (P.ART / "gate2" / "gate2_exploratory.json").exists()


def test_eval_gate2_with_override_writes_only_exploratory_output(cfg, sandbox):
    root, rec = sandbox
    runs, per, st = _write_logged_gate2(cfg, root)
    ov = P.gate1_override(rec)
    g = P.eval_gate2(cfg, ov)
    fid = json.loads((ROOT / "configs" / "baseline_fidelity.json").read_text())["faithful"]
    real = G.evaluate_gate2(runs, cfg["stage_b"], fid, st, G.PASS, per)
    assert g["exploratory_gate2_standalone_result"] == real["status"] and g["gate1_status"] == "FAIL"
    assert (P.ART / "gate2" / "gate2_exploratory.json").exists() and not (P.ART / "gate2" / "gate2.json").exists()
    assert json.loads(rec.read_text())["gate1"]["status"] == "FAIL"


# ------------------------------------------------------------------------------------------ CLI
@pytest.mark.parametrize("argv", [
    ["train-b", "--exploratory-override-gate1"],  # override without naming the Gate-1 record
    ["train-b", "--gate1-record", "artifacts/v2/gate1/gate1.json"],  # record without the explicit override
    ["all", "--exploratory-override-gate1", "--gate1-record", "artifacts/v2/gate1/gate1.json"],
    ["report-exploratory"],
])
def test_cli_override_must_be_explicit(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)
