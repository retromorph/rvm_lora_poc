"""Before/after integrity snapshot of a reference run that a later namespace must not touch.

    python -m rvm_lora.integrity snapshot <out.json> [--namespace v2]

Records the SHA256 of every file under artifacts/<ns>/, checkpoints/<ns>/, reports/<ns>/ and a digest of every
MLflow run (status, end time, tags, params, metrics). `compare` reports changed/missing reference items; new
MLflow runs are listed separately (adding runs is expected, altering historical ones is not).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .config import REPO_ROOT, sha256_file
from .tracking import write_json


def file_hashes(namespace: str) -> dict[str, str]:
    out = {}
    for top in ("artifacts", "checkpoints", "reports"):
        base = REPO_ROOT / top / namespace
        for p in sorted(base.rglob("*")) if base.exists() else []:
            if p.is_file():
                out[str(p.relative_to(REPO_ROOT))] = sha256_file(p)
    return out


def mlflow_digests(uri: str) -> dict[str, dict]:
    from mlflow.tracking import MlflowClient

    c = MlflowClient(uri)
    out = {}
    for e in c.search_experiments():
        for r in c.search_runs([e.experiment_id], max_results=50000):
            body = {"status": r.info.status, "end_time": r.info.end_time, "tags": dict(sorted(r.data.tags.items())),
                    "params": dict(sorted(r.data.params.items())), "metrics": dict(sorted(r.data.metrics.items()))}
            out[r.info.run_id] = {"experiment": e.name, "status": r.info.status,
                                  "digest": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
    return out


def snapshot(namespace: str, uri: str) -> dict:
    return {"namespace": namespace, "files": file_hashes(namespace), "mlflow_uri": uri, "mlflow": mlflow_digests(uri)}


def compare(before: dict, after: dict) -> dict:
    fb, fa = before["files"], after["files"]
    mb, ma = before["mlflow"], after["mlflow"]
    changed_files = sorted(k for k in fb if fa.get(k) != fb[k])
    changed_runs = sorted(k for k in mb if ma.get(k, {}).get("digest") != mb[k]["digest"])
    return {"reference_namespace": before["namespace"], "files_checked": len(fb), "files_changed_or_missing": changed_files,
            "files_added": sorted(set(fa) - set(fb)), "mlflow_runs_checked": len(mb),
            "mlflow_runs_changed_or_missing": changed_runs, "mlflow_runs_added": len(set(ma) - set(mb)),
            "untouched": not changed_files and not changed_runs and not (set(fa) - set(fb))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["snapshot"])
    ap.add_argument("out")
    ap.add_argument("--namespace", default="v2")
    ap.add_argument("--keep", action="store_true", help="do not overwrite an existing snapshot (resume)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.keep and out.exists():
        print(f"keeping existing snapshot {out}")
        return 0
    write_json(out, snapshot(a.namespace, os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
