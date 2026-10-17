"""MLflow tracking with deterministic run keys, idempotent reuse and a local JSON mirror.

Run key = sha256(protocol_sha|gate|method|task|seed|arch_id)[:20]. A run whose local mirror says
COMPLETED (and whose MLflow run is FINISHED) is reused, never silently replaced. Any other prior
attempt is kept and a new attempt number is used; metrics of different attempts are never mixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

EXPERIMENTS = {"gate0": "rvm-lora/gate0", "gate1": "rvm-lora/gate1", "gate2": "rvm-lora/gate2"}


def run_key(protocol_sha: str, gate: str, method: str, task: str, seed: int, arch_id: str) -> str:
    raw = f"{protocol_sha}|{gate}|{method}|{task}|{seed}|{arch_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def mirror_dir(root: Path, gate: str, key: str) -> Path:
    return root / "artifacts" / "runs" / gate / key


def read_mirror(root: Path, gate: str, key: str) -> dict | None:
    p = mirror_dir(root, gate, key) / "result.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    import numpy as np

    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Tracker:
    """Thin MLflow client wrapper; requires a reachable tracking server (no silent local fallback)."""

    def __init__(self, uri: str | None = None):
        import mlflow
        from mlflow.tracking import MlflowClient

        self.uri = uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(self.uri)
        self.client = MlflowClient(self.uri)
        self._exp_ids: dict[str, str] = {}

    def experiment_id(self, gate: str) -> str:
        name = EXPERIMENTS[gate]
        if name not in self._exp_ids:
            exp = self.client.get_experiment_by_name(name)
            self._exp_ids[name] = exp.experiment_id if exp else self.client.create_experiment(name)
        return self._exp_ids[name]

    def find_runs(self, gate: str, key: str) -> list:
        return self.client.search_runs([self.experiment_id(gate)], filter_string=f"tags.run_key = '{key}'",
                                       max_results=100)

    def start(self, gate: str, key: str, tags: dict, params: dict) -> tuple[str, int]:
        prior = self.find_runs(gate, key)
        attempt = 1 + max([int(r.data.tags.get("attempt", "0")) for r in prior] or [0])
        for r in prior:  # mark unfinished prior attempts explicitly (never deleted)
            if r.info.status == "RUNNING":
                self.client.set_tag(r.info.run_id, "status", "INTERRUPTED_SUPERSEDED")
                self.client.set_terminated(r.info.run_id, status="KILLED")
        all_tags = {**{k: str(v) for k, v in tags.items()}, "run_key": key, "attempt": str(attempt),
                    "hostname": socket.gethostname(), "status": "RUNNING"}
        run = self.client.create_run(self.experiment_id(gate), tags=all_tags,
                                     run_name=f"{tags.get('method')}-{tags.get('task')}-s{tags.get('seed')}-"
                                              f"{tags.get('arch_id', '')}-a{attempt}")
        for k, v in params.items():
            self.client.log_param(run.info.run_id, k, str(v)[:6000])
        return run.info.run_id, attempt

    def log_metrics(self, run_id: str, metrics: dict, step: int | None = None) -> None:
        ts = int(time.time() * 1000)
        for k, v in metrics.items():
            if v is None:
                continue
            self.client.log_metric(run_id, k, float(v), timestamp=ts, step=step or 0)

    def log_artifact(self, run_id: str, path: Path, subdir: str | None = None) -> None:
        self.client.log_artifact(run_id, str(path), subdir)

    def finish(self, run_id: str, status: str) -> None:
        self.client.set_tag(run_id, "status", status)
        self.client.set_terminated(run_id, status="FINISHED" if status == "COMPLETED" else "FAILED")

    def run_info(self, run_id: str):
        return self.client.get_run(run_id)
