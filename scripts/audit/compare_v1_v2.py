"""Post-run comparison of the original (v1) and corrected (v2) runs, plus (if present) the EXPLORATORY SVD-init runs.

Writes <out>/v1_v2_comparison.json. Read-only. The v2 gate verdict comes from reports/v2/gates.json (mechanical
evaluator); this script only cross-checks and tabulates it.
argv: <repo> <out json> [<svdinit_runs.json>]
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SVD = Path(sys.argv[3]) if len(sys.argv) > 3 else None
sys.path.insert(0, str(REPO / "src"))
from rvm_lora.config import load_config  # noqa: E402
from rvm_lora.gates import evaluate_gate1  # noqa: E402

cfg = load_config(REPO / "configs/preregistered.yaml")
res = {"label": "v1 vs v2 comparison (audit, read-only)"}


def runs(root):
    out = {}
    for p in glob.glob(str(root / "runs/gate1/*/result.json")):
        r = json.loads(Path(p).read_text())
        ev = np.load(Path(p).parent / "eval_audit.npz")
        out[r["arch_id"]] = {"r": r, "logits": ev["logits"],
                             "curve": json.loads((Path(p).parent / "train_curve.json").read_text())}
    return out


v1, v2 = runs(REPO / "artifacts"), runs(REPO / "artifacts" / "v2")
per = []
for a in cfg.architectures():
    x, y = v1[a.arch_id], v2[a.arch_id]
    per.append({"arch_id": a.arch_id, "v1_audit_loss": x["r"]["metrics"]["audit_loss"],
                "v2_audit_loss": y["r"]["metrics"]["audit_loss"],
                "logits_bitwise_equal": bool(np.array_equal(x["logits"], y["logits"])),
                "max_abs_logit_diff": float(np.abs(x["logits"] - y["logits"]).max()),
                "train_curve_bitwise_equal": x["curve"] == y["curve"],
                "v1_mlflow_run_id": x["r"]["mlflow_run_id"], "v2_mlflow_run_id": y["r"]["mlflow_run_id"],
                "v1_head_digest": x["r"]["head_digest"], "v2_head_digest": y["r"]["head_digest"],
                "v2_head_checkpoint_sha256": y["r"].get("head_checkpoint_sha256"), "v2_git_sha": y["r"].get("git_sha"),
                "v1_train_wall_s": x["r"]["metrics"]["train_wall_s"], "v2_train_wall_s": y["r"]["metrics"]["train_wall_s"]})
res["per_architecture"] = per
res["all_logits_bitwise_equal"] = all(p["logits_bitwise_equal"] for p in per)
res["all_train_curves_bitwise_equal"] = all(p["train_curve_bitwise_equal"] for p in per)
res["distinct_mlflow_ids"] = all(p["v1_mlflow_run_id"] != p["v2_mlflow_run_id"] for p in per)
res["head_only_logits_bitwise_equal"] = bool(np.array_equal(np.load(REPO / "artifacts/gate1/head_only_audit.npz")["logits"],
                                                            np.load(REPO / "artifacts/v2/gate1/head_only_audit.npz")["logits"]))
for name, rel in [("selection", "probes/qnli_s42/selection.json"), ("probe", "probes/qnli_s42/probe_result.json")]:
    a = json.loads((REPO / "artifacts" / rel).read_text())
    b = json.loads((REPO / "artifacts/v2" / rel).read_text())
    if name == "probe":
        res["probe_hashes_equal"] = a["hashes"] == b["hashes"]
        res["probe_head_sha"] = [a["head_checkpoint_sha256"], b["head_checkpoint_sha256"]]
        res["selection_cost_s"] = {"v1": {k: a["timings"][k] for k in ("group_rvm_selection_s", "conditional_newton_selection_s",
                                                                       "selection_job_wall_s")},
                                   "v2": {k: b["timings"][k] for k in ("group_rvm_selection_s", "conditional_newton_selection_s",
                                                                       "selection_job_wall_s")}}
    else:
        res["selection"] = {k: {"v1": a[k]["ranks"], "v2": b[k]["ranks"]} for k in ("group_rvm", "conditional_newton", "rvm_no_logdet")}
        res["selection"]["newton_thresholded"] = {"v1": a["conditional_newton"]["diagnostic_thresholded_ranks"],
                                                  "v2": b["conditional_newton"]["diagnostic_thresholded_ranks"]}
        res["selection"]["rvm_threshold"] = {"v1": a["group_rvm"]["threshold"], "v2": b["group_rvm"]["threshold"]}
        res["selection"]["fresh_null"] = {"v1": a["group_rvm_fresh_null"]["activations"],
                                          "v2": b["group_rvm_fresh_null"]["activations"]}
        res["selection"]["rvm_trace_v2"] = [{k: s.get(k) for k in ("iter", "best_group", "gain_per_param", "action")}
                                            for s in b["group_rvm"]["trace"]]
        res["arch_scores_changed"] = [
            {"arch_id": x["arch_id"], "rvm_v1": x["group_rvm"], "rvm_v2": y["group_rvm"]}
            for x, y in zip(a["architecture_scores"], b["architecture_scores"]) if x["group_rvm"] != y["group_rvm"]]
        res["nonrvm_scores_identical"] = all(
            x[k] == y[k] for x, y in zip(a["architecture_scores"], b["architecture_scores"])
            for k in ("gradient_norm", "independent_fisher", "conditional_newton"))
g1, g2 = json.loads((REPO / "reports/gates.json").read_text()), json.loads((REPO / "reports/v2/gates.json").read_text())
res["verdicts"] = {"v1": {g: g1[g]["status"] for g in ("gate0", "gate1", "gate2")} | {"overall": g1["overall_decision"]},
                   "v2": {g: g2[g]["status"] for g in ("gate0", "gate1", "gate2")} | {"overall": g2["overall_decision"]}}
res["v2_gate1_criteria"] = [{k: c.get(k) for k in ("criterion", "value", "cutoff", "comparator", "status", "best_nonbayes")}
                            for c in g2["gate1"]["criteria"]]

# EXPLORATORY: the same Gate-1 statistics against the SVD-init utilities (never a gate)
if SVD and SVD.exists():
    sv = {r["arch_id"]: r for r in json.loads(SVD.read_text())["runs"]}
    g1v2 = json.loads((REPO / "artifacts/v2/gate1/gate1.json").read_text())
    rows = []
    for r in g1v2["rows"]:
        rows.append({**{k: r[k] for k in ("arch_id", "parameters", "gradient_norm", "independent_fisher",
                                          "conditional_newton", "group_rvm")}, "U": sv[r["arch_id"]]["U_svdinit"]})
    pairs = [(a.arch_id, f"L{a.layers[0]:02d}_qv_r2", f"L{a.layers[1]:02d}_qv_r2")
             for a in cfg.architectures() if a.family == "two_layer_rank2"]
    ex = evaluate_gate1(rows, pairs, {"fresh_activations": g2["gate1"]["criteria"][4]["value"]}, cfg["stage_a"])
    res["EXPLORATORY_svdinit"] = {
        "label": "EXPLORATORY: same frozen thresholds applied to SVD-aligned-init utilities; not a gate, not a protocol result",
        "U_svdinit": {k: v["U_svdinit"] for k, v in sv.items()},
        "U_random_init_v2": {r["arch_id"]: r["U"] for r in g1v2["rows"]},
        "criteria": [{k: c.get(k) for k in ("criterion", "value", "cutoff", "status", "best_nonbayes")} for c in ex["criteria"]],
        "concordance": {s: ex["concordance"][s]["concordance"] for s in ex["concordance"]},
        "top3_mean_U": {s: ex["top3"][s]["mean_U"] for s in ex["top3"]},
        "spearman": ex["spearman"], "interactions": ex["interactions"],
        "all_zero_initial_update": all(v["zero_initial_update"] for v in sv.values()),
        "mean_U_svdinit": float(np.mean([v["U_svdinit"] for v in sv.values()])),
        "mean_U_random_v2": float(np.mean([r["U"] for r in g1v2["rows"]])),
        "mlflow_run_ids": {k: v["mlflow_run_id"] for k, v in sv.items()}}
OUT.write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: v for k, v in res.items() if k not in ("per_architecture",)}, indent=1, default=str)[:12000])
