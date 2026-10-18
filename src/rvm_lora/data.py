"""Frozen GLUE splits, manifests and tokenisation.

Split rule (PREREGISTRATION §1): for each task independently, shuffle the official *training* `idx`
values with numpy PCG64(split_seed); the first 10,000 form the training pool, the next 1,500 the
internal audit set. Inside the pool (in shuffled order): 2,000 head, 512 direction, 512 evidence,
6,976 other. The adapter-training pool is the 8,000 non-head rows. Official validation is never
touched by `make_split`; it is loaded only by the Gate-2 evaluator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATASET_REVISION, MODEL_REVISION, Config

TEXT_FIELDS = {"qnli": ("question", "sentence"), "sst2": ("sentence", None)}
SPLIT_NAMES = ("head", "directions", "evidence", "other", "audit")


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def make_split(train_idx: np.ndarray | list[int], cfg: Config) -> dict[str, list[int]]:
    """Deterministic partition of official training `idx` values. Independent of any run seed."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if len(np.unique(train_idx)) != len(train_idx):
        raise ValueError("official training idx values are not unique")
    perm = np.random.default_rng(int(cfg["split_seed"])).permutation(len(train_idx))
    shuffled = train_idx[perm]
    P, A = int(cfg["train_pool_size"]), int(cfg["audit_pool_size"])
    pool, audit = shuffled[:P], shuffled[P:P + A]
    h, d, e = int(cfg["head_size"]), int(cfg["directions_size"]), int(cfg["evidence_size"])
    out = {
        "head": pool[:h],
        "directions": pool[h:h + d],
        "evidence": pool[h + d:h + d + e],
        "other": pool[h + d + e:],
        "audit": audit,
    }
    out = {k: [int(x) for x in v] for k, v in out.items()}
    out["adapter_train"] = out["directions"] + out["evidence"] + out["other"]
    return out


def validate_split(split: dict[str, list[int]], cfg: Config, n_train: int) -> dict:
    """Assert sizes, disjointness and membership in the official training range. Returns checks."""
    sizes = {"head": cfg["head_size"], "directions": cfg["directions_size"], "evidence": cfg["evidence_size"],
             "other": cfg["other_adapter_training_size"], "audit": cfg["audit_pool_size"],
             "adapter_train": cfg["adapter_train_size"]}
    for k, n in sizes.items():
        if len(split[k]) != n:
            raise AssertionError(f"split {k} has {len(split[k])} rows, expected {n}")
    union: set[int] = set()
    for k in SPLIT_NAMES:
        s = set(split[k])
        if len(s) != len(split[k]):
            raise AssertionError(f"duplicates inside {k}")
        if union & s:
            raise AssertionError(f"split {k} overlaps an earlier split")
        union |= s
    if set(split["adapter_train"]) & set(split["audit"]) or set(split["adapter_train"]) & set(split["head"]):
        raise AssertionError("adapter-training pool overlaps audit/head")
    if min(union) < 0 or max(union) >= n_train:
        raise AssertionError("split idx outside official training range")
    return {"disjoint": True, "sizes_ok": True, "within_official_train": True}


def build_manifest(task: str, split: dict[str, list[int]], n_train: int, dataset_fingerprint: str) -> dict:
    return {
        "task": task,
        "dataset_repository": "nyu-mll/glue",
        "dataset_revision": DATASET_REVISION,
        "source_split": "train",
        "official_train_rows": n_train,
        "dataset_fingerprint": dataset_fingerprint,
        "split_seed_rule": "numpy.random.default_rng(split_seed).permutation(n_train) over official train idx",
        "ids": split,
        "sha256": {k: ids_sha256(v) for k, v in split.items()},
    }


def manifest_path(root: Path, task: str) -> Path:
    return root / "reports" / "splits" / f"{task}_split_manifest.json"


def load_manifest(root: Path, task: str) -> dict:
    with open(manifest_path(root, task)) as f:
        return json.load(f)


# --------------------------------------------------------------------------- HF loading
def load_glue(task: str, split: str):
    from datasets import load_dataset

    return load_dataset("nyu-mll/glue", task, split=split, revision=DATASET_REVISION)


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("FacebookAI/roberta-base", revision=MODEL_REVISION)


@dataclass
class Encoded:
    """Tokenised rows (fixed 128-token padding) aligned with `idx` order."""

    idx: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.idx)

    def subset(self, ids: list[int]) -> "Encoded":
        pos = {int(i): k for k, i in enumerate(self.idx)}
        sel = np.asarray([pos[int(i)] for i in ids], dtype=np.int64)
        return Encoded(self.idx[sel], self.input_ids[sel], self.attention_mask[sel], self.labels[sel])


def encode(task: str, ds, tokenizer, max_length: int) -> Encoded:
    a, b = TEXT_FIELDS[task]
    texts_a = list(ds[a])
    texts_b = list(ds[b]) if b else None
    enc = tokenizer(texts_a, texts_b, truncation=True, max_length=max_length, padding="max_length",
                    return_tensors="np")
    return Encoded(np.asarray(ds["idx"], dtype=np.int64), enc["input_ids"].astype(np.int64),
                   enc["attention_mask"].astype(np.int64), np.asarray(ds["label"], dtype=np.int64))


def cache_path(root: Path, task: str, split: str) -> Path:
    return root / "data" / f"{task}_{split}_encoded.npz"


def save_encoded(path: Path, e: Encoded) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, idx=e.idx, input_ids=e.input_ids, attention_mask=e.attention_mask, labels=e.labels)


def load_encoded(path: Path) -> Encoded:
    z = np.load(path)
    return Encoded(z["idx"], z["input_ids"], z["attention_mask"], z["labels"])
