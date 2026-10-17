"""Gate 0 item 6 (data): splits deterministic, disjoint, run-seed independent, train-only."""

import json

import numpy as np
import pytest
import torch

from rvm_lora.data import load_manifest, make_split, manifest_path, validate_split

ROOT_TESTS = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n_train", [104743, 67349])  # QNLI / SST-2 official train sizes
def test_split_sizes_disjoint_and_stable(cfg, n_train, record):
    idx = np.arange(n_train)
    torch.manual_seed(1)
    np.random.seed(1)
    a = make_split(idx, cfg)
    torch.manual_seed(2025)
    np.random.seed(2025)
    _ = np.random.rand(10)
    b = make_split(idx, cfg)
    assert a == b  # independent of global/model seeds
    checks = validate_split(a, cfg, n_train)
    assert checks["disjoint"]
    assert len(a["adapter_train"]) == 8000 and set(a["directions"]) <= set(a["adapter_train"])
    assert not set(a["audit"]) & set(a["adapter_train"])
    record(**{f"n{n_train}": {k: len(v) for k, v in a.items()}})


def test_validate_split_rejects_overlap(cfg):
    a = make_split(np.arange(20000), cfg)
    bad = dict(a)
    bad["audit"] = a["audit"][:-1] + [a["head"][0]]
    with pytest.raises(AssertionError):
        validate_split(bad, cfg, 20000)


def test_prepared_manifests_if_present(cfg, record):
    found = {}
    for task in cfg.tasks:
        p = manifest_path(ROOT_TESTS, task)
        if not p.exists():
            continue
        m = load_manifest(ROOT_TESTS, task)
        assert m["source_split"] == "train"
        validate_split(m["ids"], cfg, m["official_train_rows"])
        found[task] = m["sha256"]["audit"]
    record(manifests_checked=list(found), audit_sha256=found)
    if not found:
        pytest.skip("splits not prepared yet")
    json.dumps(found)
