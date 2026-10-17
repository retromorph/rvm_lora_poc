"""Idempotent run keys, attempt numbering and MLflow round-trip (local SQLite store; plus the live
server when MLFLOW_TRACKING_URI is set)."""

import os

import pytest

from rvm_lora.tracking import Tracker, mirror_dir, read_mirror, run_key, write_json


def test_run_key_deterministic():
    a = run_key("sha", "gate1", "m", "qnli", 42, "L00")
    assert a == run_key("sha", "gate1", "m", "qnli", 42, "L00")
    assert len({a, run_key("sha", "gate1", "m", "qnli", 2025, "L00"), run_key("sha2", "gate1", "m", "qnli", 42, "L00"),
                run_key("sha", "gate2", "m", "qnli", 42, "L00")}) == 4


def test_mirror_roundtrip(tmp_path):
    k = run_key("s", "gate1", "m", "t", 1, "a")
    write_json(mirror_dir(tmp_path, "gate1", k) / "result.json", {"status": "COMPLETED", "x": 1.5})
    assert read_mirror(tmp_path, "gate1", k)["x"] == 1.5
    assert read_mirror(tmp_path, "gate1", "missing") is None


def _roundtrip(uri, record=None):
    tr = Tracker(uri)
    key = run_key("test-protocol", "gate0", "mlflow_smoke", "none", 0, os.urandom(4).hex())
    rid1, a1 = tr.start("gate0", key, {"method": "mlflow_smoke", "task": "none", "seed": 0}, {"p": 1})
    tr.log_metrics(rid1, {"m": 0.25})
    rid2, a2 = tr.start("gate0", key, {"method": "mlflow_smoke", "task": "none", "seed": 0}, {"p": 1})
    assert (a1, a2) == (1, 2)
    assert tr.run_info(rid1).data.tags["status"] == "INTERRUPTED_SUPERSEDED"
    tr.log_metrics(rid2, {"m": 0.5})
    tr.finish(rid2, "COMPLETED")
    runs = tr.find_runs("gate0", key)
    assert len(runs) == 2
    got = tr.run_info(rid2)
    assert got.data.metrics["m"] == 0.5 and got.info.status == "FINISHED"
    if record:
        record(uri=uri, run_id=rid2)


def test_mlflow_local_sqlite(tmp_path):
    _roundtrip(f"sqlite:///{tmp_path}/mlflow.db")


@pytest.mark.skipif(not os.environ.get("MLFLOW_TRACKING_URI"), reason="no live MLflow server configured")
def test_mlflow_live_server(record):
    _roundtrip(os.environ["MLFLOW_TRACKING_URI"], record)
