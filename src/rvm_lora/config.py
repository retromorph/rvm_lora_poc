"""Typed, validated view of `configs/preregistered.yaml`.

The YAML is the frozen source of truth. This module only *reads* it and asserts
internal consistency; it never supplies alternative defaults for scientific values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "preregistered.yaml"
PREREG_MD = REPO_ROOT / "docs" / "PREREGISTRATION.md"

# Pinned external revisions (resolved from the Hub on 2026-10-17 before any outcome was read).
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"  # FacebookAI/roberta-base
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"  # nyu-mll/glue


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Architecture:
    """A fixed Gate-1 architecture: exact (module name -> rank) map."""

    arch_id: str
    family: str
    layers: tuple[int, ...]
    ranks: dict[str, int] = field(hash=False)
    parameters: int

    @property
    def rank_units(self) -> int:
        return sum(self.ranks.values())


def module_name(layer: int, proj: str) -> str:
    return f"roberta.encoder.layer.{layer}.attention.self.{proj}"


def eligible_modules(num_layers: int = 12, projections: tuple[str, ...] = ("query", "value")) -> list[str]:
    """Canonical module order used everywhere: layer-major, then projection order from YAML."""
    return [module_name(layer, p) for layer in range(num_layers) for p in projections]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(hash=False)
    path: Path
    sha256: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    # convenient typed accessors -------------------------------------------------
    @property
    def tasks(self) -> list[str]:
        return list(self.raw["tasks"])

    @property
    def modules(self) -> list[str]:
        m = self.raw["model"]
        return eligible_modules(m["num_layers"], tuple(m["eligible_projections"]))

    @property
    def rank_unit_params(self) -> int:
        return int(self.raw["model"]["rank_unit_parameters"])

    @property
    def n_dir(self) -> int:
        return int(self.raw["model"]["candidate_directions_per_module"])

    @property
    def max_rank(self) -> int:
        return int(self.raw["model"]["max_rank_per_module"])

    @property
    def log_alpha_bounds(self) -> tuple[float, float]:
        lo, hi = self.raw["probe"]["rvm_precision_log_bounds"]
        return float(lo), float(hi)

    def architectures(self) -> list[Architecture]:
        """The exact 20-architecture Gate-1 manifest, in a fixed ID order."""
        sa = self.raw["stage_a"]["architectures"]
        out: list[Architecture] = []
        unit = self.rank_unit_params
        s2 = sa["single_layer_rank2"]
        for layer in s2["layers"]:
            ranks = {module_name(layer, "query"): s2["rank_query"], module_name(layer, "value"): s2["rank_value"]}
            out.append(Architecture(f"L{layer:02d}_qv_r2", "single_layer_rank2", (layer,), ranks,
                                    unit * sum(ranks.values())))
        s4 = sa["single_layer_rank4"]
        for layer in s4["layers"]:
            ranks = {module_name(layer, "query"): s4["rank_query"], module_name(layer, "value"): s4["rank_value"]}
            out.append(Architecture(f"L{layer:02d}_qv_r4", "single_layer_rank4", (layer,), ranks,
                                    unit * sum(ranks.values())))
        tl = sa["two_layer_rank2"]
        for a, b in tl["pairs"]:
            ranks = {}
            for layer in (a, b):
                ranks[module_name(layer, "query")] = tl["rank_query_per_layer"]
                ranks[module_name(layer, "value")] = tl["rank_value_per_layer"]
            out.append(Architecture(f"L{a:02d}+L{b:02d}_qv_r2", "two_layer_rank2", (a, b), ranks,
                                    unit * sum(ranks.values())))
        return out


class ConfigError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(raw: dict[str, Any]) -> None:
    """Internal-consistency assertions; raises ConfigError on any contradiction."""
    _check(raw["status"] == "frozen_before_outcomes", "config not frozen")
    _check(raw["backbone"] == "FacebookAI/roberta-base", "unexpected backbone")
    _check(raw["dataset_repository"] == "nyu-mll/glue", "unexpected dataset repo")
    _check(sorted(raw["tasks"]) == ["qnli", "sst2"], "unexpected tasks")
    parts = raw["head_size"] + raw["directions_size"] + raw["evidence_size"] + raw["other_adapter_training_size"]
    _check(parts == raw["train_pool_size"], f"train pool partition {parts} != {raw['train_pool_size']}")
    _check(raw["adapter_train_size"] == raw["train_pool_size"] - raw["head_size"], "adapter pool size mismatch")
    m = raw["model"]
    _check(m["eligible_module_count"] == m["num_layers"] * len(m["eligible_projections"]), "module count")
    _check(m["rank_unit_parameters"] == 2 * m["hidden_size"], "rank unit params != 2*hidden")
    _check(m["max_rank_per_module"] == m["candidate_directions_per_module"], "max rank != directions")
    at = raw["adapter_training"]
    _check(at["microbatch_size"] * at["gradient_accumulation"] == at["effective_batch_size"], "eff batch")
    _check(at["oom_fallback_microbatch"] * at["oom_fallback_accumulation"] == at["effective_batch_size"],
           "fallback eff batch")
    sb = raw["stage_b"]
    unit = m["rank_unit_parameters"]
    _check(sb["budget_parameters"] == sb["budget_rank_units"] * unit, "B24 params")
    _check(sb["comparison_parameters"] == sb["comparison_rank_units"] * unit, "B48 params")
    _check(sb["budget_rank_units"] == m["eligible_module_count"], "B24 != 1 rank per module")
    _check(len(sb["methods"]) * len(sb["tasks"]) * len(sb["seeds"]) == sb["max_main_runs"], "Gate-2 run count")
    sa = raw["stage_a"]["architectures"]
    total = 0
    for fam, spec in sa.items():
        n = len(spec["layers"]) if "layers" in spec else len(spec["pairs"])
        _check(n == spec["count"], f"{fam} count")
        total += n
        per_layer = spec.get("rank_query", spec.get("rank_query_per_layer")) + spec.get(
            "rank_value", spec.get("rank_value_per_layer"))
        n_layers = 1 if "layers" in spec else 2
        _check(per_layer * n_layers * unit == spec["exact_parameters"], f"{fam} exact params")
    _check(total == raw["stage_a"]["architecture_count"], "architecture count")
    _check(total + sb["max_main_runs"] == raw["maximum_full_adapter_training_runs"], "max runs")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    validate(raw)
    return Config(raw=raw, path=path, sha256=sha256_file(path))
