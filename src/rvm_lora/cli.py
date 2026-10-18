"""Command-line entry point: python -m rvm_lora.cli <command> [--config configs/preregistered.yaml]."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import DATASET_REVISION, MODEL_REVISION, load_config, sha256_file
from .pipeline import ART, REPORTS, ROOT, git_dirty, git_sha, log, now
from .tracking import write_json


def _tracker():
    from .tracking import Tracker

    return Tracker()


# ------------------------------------------------------------------------------------------ env-check
def env_check(cfg, args) -> int:
    import importlib.metadata as md

    info = {"timestamp": now(), "git_sha": git_sha(), "git_dirty": git_dirty(), "config_sha256": cfg.sha256,
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock") if (ROOT / "uv.lock").exists() else None,
            "python": sys.version.split()[0], "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
            "packages": {}}
    for p in ("torch", "transformers", "peft", "datasets", "accelerate", "mlflow", "numpy", "scipy", "pandas",
              "matplotlib", "safetensors", "huggingface-hub"):
        try:
            info["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            info["packages"][p] = None
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                     "--format=csv,noheader"], text=True).strip()
        info["nvidia_smi"] = q
    except Exception as e:
        info["nvidia_smi"] = f"unavailable: {e!r}"
    import torch

    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2**20
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        try:  # real BF16 forward/backward smoke
            lin = torch.nn.Linear(64, 64).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = lin(torch.randn(8, 64, device="cuda")).sum()
            y.backward()
            info["bf16_forward_backward_ok"] = bool(torch.isfinite(lin.weight.grad).all())
        except Exception as e:
            info["bf16_forward_backward_ok"] = False
            info["bf16_error"] = repr(e)
    du = shutil.disk_usage(ROOT)
    info["disk_free_gb"] = du.free / 2**30
    info["cpu_count"] = os.cpu_count()
    try:
        import urllib.request

        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        with urllib.request.urlopen(uri + "/health", timeout=5) as r:
            info["mlflow_health"] = r.read().decode()[:50]
        info["mlflow_uri"] = uri
    except Exception as e:
        info["mlflow_health"] = f"unreachable: {e!r}"
    write_json(REPORTS / "environment.json", info)
    print(json.dumps(info, indent=2))
    return 0


# ------------------------------------------------------------------------------------------ gate 0
GATE0_CHECKS = [
    # (check name, pytest node-id substrings that must all pass)
    ("gaussian_evidence_matches_direct", ["test_evidence.py::test_gram_evidence_matches_direct",
                                          "test_evidence.py::test_gradient_matches_finite_difference"]),
    ("singleton_analytic_solution", ["test_evidence.py::test_singleton_analytic_vs_numeric"]),
    ("singleton_ranking_equivalence_q2_over_h", ["test_evidence.py::test_singleton_ranking_equals_q2_over_h"]),
    ("conditional_group_evidence_equals_direct_difference",
     ["test_evidence.py::test_conditional_fixed_increment_equals_difference",
      "test_evidence.py::test_degenerate_cases"]),
    ("selector_behaviour_and_null", ["test_nulls.py"]),
    ("logit_direction_jacobian_vs_jvp_fd", ["test_features.py"]),
    ("exact_adapter_placement_and_param_counts", ["test_ranks.py"]),
    ("splits_disjoint_stable_no_validation", ["test_splits.py"]),
    ("gate_logic_and_reproducibility", ["test_gates.py"]),
    ("baselines_oplora_ilp_and_eva_budget", ["test_baselines.py"]),
    ("mlflow_idempotent_run_keys", ["test_tracking.py"]),
    ("cuda_smoke_two_steps_frozen_head", ["test_end_to_end_smoke.py"]),
]


def gate0(cfg, args) -> int:
    import xml.etree.ElementTree as ET

    from .gates import evaluate_gate0

    outdir = ART / "gate0"
    outdir.mkdir(parents=True, exist_ok=True)
    junit = outdir / "pytest_junit.xml"
    env = {**os.environ, "RVM_GATE0_OUT": str(outdir / "measurements"), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-m", "not post_hoc", f"--junitxml={junit}", "tests"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    (outdir / "pytest_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    tree = ET.parse(junit)
    cases = {}
    for tc in tree.iter("testcase"):
        nid = f"{tc.get('file') or tc.get('classname', '').replace('.', '/') + '.py'}::{tc.get('name')}"
        failed = any(ch.tag in ("failure", "error") for ch in tc)
        skipped = any(ch.tag == "skipped" for ch in tc)
        cases[nid] = "FAIL" if failed else ("SKIPPED" if skipped else "PASS")
    meas = {}
    mdir = outdir / "measurements"
    if mdir.exists():
        for f in sorted(mdir.glob("*.json")):
            meas[f.stem] = json.loads(f.read_text())
    checks = []
    for name, pats in GATE0_CHECKS:
        matched = {k: v for k, v in cases.items() if any(p in k.replace("tests/", "") for p in pats)}
        if not matched:
            st = "BLOCKED"
        elif any(v == "FAIL" for v in matched.values()):
            st = "FAIL"
        elif any(v == "SKIPPED" for v in matched.values()):
            st = "BLOCKED"
        else:
            st = "PASS"
        stems = {p.split("::")[0].replace(".py", "") for p in pats}
        tests = {p.split("::")[1] for p in pats if "::" in p}
        measured = {k: v for k, v in meas.items() if k.split("__")[0] in stems
                    and (not tests or k.split("__")[1] in tests)}
        checks.append({"name": name, "status": st, "tests": matched, "measured": measured,
                       "source": "artifacts/gate0/pytest_junit.xml"})
    g0 = evaluate_gate0(checks)
    g0.update(pytest_returncode=proc.returncode, n_tests=len(cases),
              n_passed=sum(v == "PASS" for v in cases.values()), wall_s=time.perf_counter() - t0,
              git_sha=git_sha(), evaluated_at=now(), measurements=meas)
    write_json(outdir / "gate0.json", g0)
    print(proc.stdout[-3000:])
    log(f"Gate 0: {g0['status']} ({g0['n_passed']}/{g0['n_tests']} tests passed)")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            from .tracking import run_key

            tr = _tracker()
            key = run_key(cfg.sha256, "gate0", "gate0_suite", "none", 0, git_sha())
            rid, _ = tr.start("gate0", key, {"gate": "gate0", "method": "gate0_suite", "task": "none", "seed": 0,
                                             "git_sha": git_sha(), "protocol_sha256": cfg.sha256}, {})
            tr.log_metrics(rid, {"n_tests": g0["n_tests"], "n_passed": g0["n_passed"]})
            tr.log_artifact(rid, outdir / "gate0.json")
            tr.log_artifact(rid, junit)
            tr.finish(rid, "COMPLETED" if g0["status"] == "PASS" else "FAILED")
            g0["mlflow_run_id"] = rid
            write_json(outdir / "gate0.json", g0)
        except Exception as e:
            log(f"MLflow logging of Gate 0 failed: {e!r}")
    return 0 if g0["status"] == "PASS" else 1


# ------------------------------------------------------------------------------------------ dry run
def dry_run(cfg) -> int:
    """Print the full plan without touching data or labels."""
    unit = cfg.rank_unit_params
    print(f"# RVM-LoRA plan (dry run; no data or labels read)\nconfig: {cfg.path} sha256={cfg.sha256}")
    print(f"model {cfg['backbone']}@{MODEL_REVISION}; dataset {cfg['dataset_repository']}@{DATASET_REVISION}")
    print("prerequisites: uv sync --frozen; MLflow at $MLFLOW_TRACKING_URI (127.0.0.1:5000 on the GPU host); "
          "1 CUDA GPU; SSH tunnel for UI: ssh -N -L 15000:127.0.0.1:5000 $REMOTE_HOST")
    print("stages: env-check -> prepare (split IDs) -> gate0 (pytest + CUDA smoke) -> freeze -> warm-head(qnli,42) -> "
          "probe(qnli,42; scores frozen) -> oracle-a (20 runs) -> eval-gate1 -> [if PASS] train-b (24 runs) -> "
          "eval-gate2 -> report")
    print("\nGate 1 architectures (QNLI seed 42, audit evaluation):")
    for a in cfg.architectures():
        assert a.parameters == unit * a.rank_units
        print(f"  {a.arch_id:18s} {a.family:20s} params={a.parameters:6d} ranks={a.ranks}")
    sb = cfg["stage_b"]
    print(f"\nGate 2 (only if Gate 1 PASS): {len(sb['methods'])} methods x {sb['tasks']} x seeds {sb['seeds']} = "
          f"{len(sb['methods']) * len(sb['tasks']) * len(sb['seeds'])} runs; B24={sb['budget_parameters']} "
          f"B48={sb['comparison_parameters']}")
    for m in sb["methods"]:
        print(f"  {m}")
    print("\nexpected artifacts: reports/{protocol_freeze.json,gates.json,run_index.csv,architecture_scores.csv,"
          "FINAL_REPORT.md,figures/,review_bundle.zip}; artifacts/{gate0,gate1,gate2,probes,runs}/")
    print(f"maximum full adapter trainings: {cfg['maximum_full_adapter_training_runs']}")
    return 0


# ------------------------------------------------------------------------------------------ all / status
def run_all(cfg, args) -> int:
    from . import pipeline as P

    if args.dry_run:
        return dry_run(cfg)
    P.prepare(cfg)  # frozen split IDs + tokenisation only; no outcome is computed
    g0p = ART / "gate0" / "gate0.json"
    if not g0p.exists() or json.loads(g0p.read_text())["status"] != "PASS":
        if gate0(cfg, args) != 0:
            log("Gate 0 did not pass: stopping before any held-out outcome is computed")
            P.write_gates(cfg)
            _report(cfg)
            return 2
    P.write_freeze(cfg)
    tr = _tracker()
    task, seed = cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]
    P.warm_head(cfg, task, seed, tr)
    P.probe(cfg, task, seed, gate1=True, tracker=tr)
    P.oracle_a(cfg, tr)
    g1 = P.eval_gate1(cfg)
    log(f"Gate 1: {g1['status']}")
    if g1["status"] == "PASS":
        P.train_b(cfg, tr)
    P.eval_gate2(cfg)
    P.write_gates(cfg)
    _report(cfg)
    return 0


def _report(cfg) -> None:
    from .report import build_report

    build_report(cfg)


def status(cfg, args) -> int:
    from .report import run_index

    rows = run_index(cfg)
    from collections import Counter

    print(Counter((r["gate"], r["status"]) for r in rows))
    for r in rows:
        print(f"{r['gate']:6s} {r['method']:20s} {r['task']:5s} {r['seed']:5} {r['arch_id']:18s} {r['status']:22s} "
              f"{r.get('mlflow_run_id') or ''}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rvm_lora")
    ap.add_argument("command", choices=["env-check", "gate0", "prepare", "freeze", "warm-head", "probe", "oracle-a",
                                        "eval-gate1", "train-b", "eval-gate2", "gates", "report", "all", "status"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "preregistered.yaml"))
    ap.add_argument("--task", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    from . import pipeline as P

    c = args.command
    if c == "env-check":
        return env_check(cfg, args)
    if c == "gate0":
        return gate0(cfg, args)
    if c == "all":
        return run_all(cfg, args)
    if c == "status":
        return status(cfg, args)
    if c == "prepare":
        P.prepare(cfg)
    elif c == "freeze":
        print(json.dumps(P.write_freeze(cfg), indent=2))
    elif c == "warm-head":
        P.warm_head(cfg, args.task or cfg["stage_a"]["task"], args.seed or cfg["stage_a"]["run_seed"], _tracker())
    elif c == "probe":
        t, s = args.task or cfg["stage_a"]["task"], args.seed or cfg["stage_a"]["run_seed"]
        P.probe(cfg, t, s, gate1=(t, s) == (cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]), tracker=_tracker())
    elif c == "oracle-a":
        P.oracle_a(cfg, _tracker())
    elif c == "eval-gate1":
        print(json.dumps(P.eval_gate1(cfg)["criteria"], indent=2, default=str))
    elif c == "train-b":
        P.train_b(cfg, _tracker())
    elif c == "eval-gate2":
        print(json.dumps(P.eval_gate2(cfg), indent=2, default=str)[:5000])
    elif c == "gates":
        print(json.dumps(P.write_gates(cfg)["overall_decision"]))
    elif c == "report":
        P.write_gates(cfg)
        _report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
