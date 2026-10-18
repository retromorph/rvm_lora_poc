"""Regression (audit A2): cached heads/probes/scores/runs are reused only with matching provenance, and a
namespaced (corrected) run can never collide with the original run's keys or directories."""

import json
from pathlib import Path

import pytest

from rvm_lora import config as C
from rvm_lora import pipeline as P
from rvm_lora.tracking import mirror_dir, run_key, write_json


def test_run_key_namespace_salting_keeps_v1_keys():
    import hashlib

    v1 = run_key("sha", "gate1", "m", "qnli", 42, "L00", namespace="")
    assert v1 == hashlib.sha256(b"sha|gate1|m|qnli|42|L00").hexdigest()[:20]  # original keys unchanged
    v2 = run_key("sha", "gate1", "m", "qnli", 42, "L00", namespace="v2")
    assert v2 != v1 and v2 == run_key("sha", "gate1", "m", "qnli", 42, "L00", namespace="v2")


def test_namespaced_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "RUN_NAMESPACE", "v2")
    assert C.namespaced(tmp_path / "artifacts") == tmp_path / "artifacts" / "v2"
    assert mirror_dir(tmp_path, "gate1", "k") == tmp_path / "artifacts" / "v2" / "runs" / "gate1" / "k"
    from rvm_lora.modeling import head_path

    assert head_path(tmp_path, "qnli", 42).parent == tmp_path / "checkpoints" / "v2"
    monkeypatch.setattr(C, "RUN_NAMESPACE", "")
    assert mirror_dir(tmp_path, "gate1", "k") == tmp_path / "artifacts" / "runs" / "gate1" / "k"


def test_check_cached():
    P.check_cached("x", {"head_checkpoint_sha256": "a", "run_namespace": "v2"}, head_checkpoint_sha256="a",
                   run_namespace="v2")
    P.check_cached("x", {"head_checkpoint_sha256": "a"}, run_namespace="v1")  # pre-namespace outputs are v1
    with pytest.raises(P.StaleCacheError):
        P.check_cached("x", {"head_checkpoint_sha256": "a"}, head_checkpoint_sha256="b")
    with pytest.raises(P.StaleCacheError):
        P.check_cached("x", {"head_checkpoint_sha256": "a"}, run_namespace="v2")


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Original (un-namespaced) layout in a temp root, independent of RVM_RUN_NAMESPACE in the environment."""
    from rvm_lora import tracking

    for mod in (C, P, tracking):
        monkeypatch.setattr(mod, "RUN_NAMESPACE", "")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "ART", tmp_path / "artifacts")
    write_json(P.head_meta_path("qnli", 42), {"head_checkpoint_sha256": "NEWHEAD", "run_namespace": "v1"})
    return tmp_path


def test_probe_cache_refuses_stale_head(cfg, sandbox):
    write_json(P.probe_dir("qnli", 42) / "probe_result.json", {"head_checkpoint_sha256": "OLDHEAD"})
    with pytest.raises(P.StaleCacheError):
        P.probe(cfg, "qnli", 42, gate1=True, tracker=None)


def test_head_only_audit_cache_refuses_stale_head(cfg, sandbox):
    write_json(P.ART / "gate1" / "head_only_audit.json", {"head_checkpoint_sha256": "OLDHEAD", "audit_loss": 0.69})
    with pytest.raises(P.StaleCacheError):
        P.head_only_audit(cfg, tracker=None)


def test_completed_run_reuse_requires_same_head(cfg, sandbox):
    key = run_key(cfg.sha256, "gate1", "fixed_architecture", "qnli", 42, "L00_qv_r2", namespace="")
    write_json(mirror_dir(sandbox, "gate1", key) / "result.json", {"status": "COMPLETED", "metrics": {}})
    with pytest.raises(P.StaleCacheError):  # v1-style mirror without a head hash is never silently reused
        P.adapter_run(cfg, None, "gate1", "fixed_architecture", "qnli", 42, "L00_qv_r2", {}, "audit", "audit", None)
    write_json(mirror_dir(sandbox, "gate1", key) / "result.json",
               {"status": "COMPLETED", "metrics": {}, "head_checkpoint_sha256": "NEWHEAD", "run_namespace": "v1"})
    assert P.adapter_run(cfg, None, "gate1", "fixed_architecture", "qnli", 42, "L00_qv_r2", {}, "audit", "audit",
                         None)["status"] == "COMPLETED"


def test_precomputed_scores_must_match_probe(cfg, sandbox, monkeypatch):
    rep = sandbox / "reports"
    monkeypatch.setattr(P, "REPORTS", rep)
    write_json(rep / "protocol_freeze.json", {"config_sha256": cfg.sha256})
    write_json(P.probe_dir("qnli", 42) / "probe_result.json",
               {"selection_file_sha256": "SELNEW", "head_checkpoint_sha256": "NEWHEAD"})
    write_json(P.ART / "gate1" / "precomputed_scores.json",
               {"selection_file_sha256": "SELOLD", "head_checkpoint_sha256": "NEWHEAD"})
    with pytest.raises(P.StaleCacheError):
        P.oracle_a(cfg, tracker=None)


def test_gate0_reuse_requires_current_code(tmp_path):
    from rvm_lora.cli import gate0_current

    p = tmp_path / "gate0.json"
    assert not gate0_current(p, "abc")
    p.write_text(json.dumps({"status": "PASS", "git_sha": "old"}))
    assert not gate0_current(p, "abc")
    p.write_text(json.dumps({"status": "PASS", "git_sha": "abc"}))
    assert gate0_current(p, "abc")
    p.write_text(json.dumps({"status": "FAIL", "git_sha": "abc"}))
    assert not gate0_current(Path(p), "abc")
