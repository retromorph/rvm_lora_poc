"""Phase 1 audit: independent recomputation of the recorded Gate-1 numbers from raw artifacts.

Read-only with respect to the original run: reads artifacts/, reports/, checkpoints/ and MLflow; writes only
to the output path given on the command line. The gate statistics are recomputed with code written for
this audit (not rvm_lora.gates) and then compared with reports/gates.json.
"""

import glob
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
out = {"label": "Phase-1 independent recomputation (audit, read-only)"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


cfg = yaml.safe_load(open(ROOT / "configs/preregistered.yaml"))
sa = cfg["stage_a"]
freeze = json.loads((ROOT / "reports/protocol_freeze.json").read_text())
gates = json.loads((ROOT / "reports/gates.json").read_text())

# ------------------------------------------------------------------ hashes and provenance
prov = {}
prov["config_sha256_now"] = sha(ROOT / "configs/preregistered.yaml")
prov["config_sha256_freeze"] = freeze["config_sha256"]
prov["prereg_sha256_now"] = sha(ROOT / "docs/PREREGISTRATION.md")
prov["prereg_sha256_freeze"] = freeze["preregistration_md_sha256"]
prov["clarif_sha256_now"] = sha(ROOT / "docs/PROTOCOL_CLARIFICATIONS.md")
prov["clarif_sha256_freeze"] = freeze["clarifications_md_sha256"]
prov["methods_sha256_now"] = sha(ROOT / "docs/METHODS.md")
prov["methods_sha256_freeze"] = freeze["methods_md_sha256"]
prov["uv_lock_now"] = sha(ROOT / "uv.lock")
prov["uv_lock_freeze"] = freeze["uv_lock_sha256"]
man = json.loads((ROOT / "reports/splits/qnli_split_manifest.json").read_text())
prov["qnli_manifest_file_sha_now"] = sha(ROOT / "reports/splits/qnli_split_manifest.json")
prov["qnli_manifest_file_sha_freeze"] = freeze["split_manifests"]["qnli"]["manifest_file_sha256"]
prov["qnli_split_ids_recomputed_match"] = {k: ids_sha(v) == man["sha256"][k] == freeze["split_manifests"]["qnli"][k]
                                           for k, v in man["ids"].items()}
ids = man["ids"]
sets = {k: set(v) for k, v in ids.items()}
prov["qnli_disjoint"] = all(not (sets[a] & sets[b]) for a, b in itertools.combinations(
    ["head", "directions", "evidence", "other", "audit"], 2))
prov["qnli_adapter_train_eq_dir_evid_other"] = ids["adapter_train"] == ids["directions"] + ids["evidence"] + ids["other"]
prov["qnli_sizes"] = {k: len(v) for k, v in ids.items()}
# encoded arrays align with manifests
for part in ["head", "directions", "evidence", "adapter_train", "audit"]:
    z = np.load(ROOT / "data" / f"qnli_{part}.npz")
    prov[f"encoded_{part}_idx_matches_manifest"] = list(map(int, z["idx"])) == ids[part]
    prov[f"encoded_{part}_label_mean"] = float(z["labels"].mean())
hmeta = json.loads((ROOT / "artifacts/heads/qnli_s42.json").read_text())
prov["head_ckpt_sha_file"] = sha(ROOT / "checkpoints/head_qnli_seed42.safetensors")
prov["head_ckpt_sha_meta"] = hmeta["head_checkpoint_sha256"]
prov["head_ids_sha_meta_matches_manifest"] = hmeta["head_ids_sha256"] == man["sha256"]["head"]
pdir = ROOT / "artifacts/probes/qnli_s42"
pr = json.loads((pdir / "probe_result.json").read_text())
prov["probe_head_sha"] = pr["head_checkpoint_sha256"]
prov["features_file_sha_now"] = sha(pdir / "features.npz")
prov["features_file_sha_recorded"] = pr["hashes"]["features_file_sha256"]
prov["atoms_sha_now"] = sha(pdir / "atoms.pt")
prov["atoms_sha_recorded"] = pr["hashes"]["atoms_sha256"]
prov["selection_sha_now"] = sha(pdir / "selection.json")
prov["selection_sha_recorded"] = pr["selection_file_sha256"]
ps = json.loads((ROOT / "artifacts/gate1/precomputed_scores.json").read_text())
prov["precomputed_scores_selection_sha"] = ps["selection_file_sha256"]
prov["precomputed_scores_sha_now"] = sha(ROOT / "artifacts/gate1/precomputed_scores.json")
prov["precomputed_scores_sha_recorded"] = pr.get("precomputed_scores_sha256")
prov["probe_dir_ids"] = {"directions": pr["hashes"]["directions_ids_sha256"] == man["sha256"]["directions"],
                         "evidence": pr["hashes"]["evidence_ids_sha256"] == man["sha256"]["evidence"]}
z = np.load(pdir / "features.npz")
prov["Phi_sha_recomputed_matches"] = hashlib.sha256(np.ascontiguousarray(z["Phi"]).tobytes()).hexdigest() == \
    pr["hashes"]["Phi_sha256"]
# timeline: freeze < scores frozen < first audit read
head_audit = json.loads((ROOT / "artifacts/gate1/head_only_audit.json").read_text())
prov["timeline"] = {"protocol_freeze": freeze["frozen_at"], "head_warm_start": hmeta["timestamp"],
                    "scores_frozen_at": ps["frozen_at"], "head_only_audit_read": head_audit["timestamp"]}
prov["scores_frozen_before_first_audit_read"] = ps["frozen_at"] < head_audit["timestamp"]
out["provenance"] = prov

# ------------------------------------------------------------------ audit losses from raw logits
audit = np.load(ROOT / "data/qnli_audit.npz")
y = audit["labels"].astype(np.int64)


def ce_acc(logits, labels):
    logits = logits.astype(np.float64)
    lse = np.logaddexp(logits[:, 0], logits[:, 1])
    loss = lse - logits[np.arange(len(labels)), labels]
    return float(loss.mean()), float(((logits[:, 1] > logits[:, 0]).astype(int) == labels).mean()), loss


h = np.load(ROOT / "artifacts/gate1/head_only_audit.npz")
assert list(h["idx"]) == list(audit["idx"])
hl, ha, hper = ce_acc(h["logits"], y)
out["head_only"] = {"recomputed_loss": hl, "recorded_loss": head_audit["audit_loss"],
                    "recomputed_acc": ha, "recorded_acc": head_audit["audit_accuracy"],
                    "per_example_matches_saved": bool(np.allclose(hper, h["loss"], rtol=0, atol=1e-12)),
                    "frac_pred_class1": float((h["logits"][:, 1] > h["logits"][:, 0]).mean()),
                    "audit_label_mean": float(y.mean()),
                    "logit_diff_mean": float((h["logits"][:, 1] - h["logits"][:, 0]).mean()),
                    "logit_diff_std": float((h["logits"][:, 1] - h["logits"][:, 0]).std()),
                    "constant_predictor_ce_at_audit_base_rate": float(-(y.mean() * np.log(y.mean()) +
                                                                        (1 - y.mean()) * np.log(1 - y.mean())))}

runs = {}
for p in sorted(glob.glob(str(ROOT / "artifacts/runs/gate1/*/result.json"))):
    r = json.loads(Path(p).read_text())
    ev = np.load(Path(p).parent / "eval_audit.npz")
    assert list(ev["idx"]) == list(audit["idx"]), "eval idx misaligned"
    L, A, per = ce_acc(ev["logits"], y)
    runs[r["arch_id"]] = {"recomputed_loss": L, "recorded_loss": r["metrics"]["audit_loss"],
                          "recomputed_acc": A, "recorded_acc": r["metrics"]["audit_accuracy"],
                          "per_example_matches_saved": bool(np.allclose(per, ev["loss"], rtol=0, atol=1e-12)),
                          "eval_file_sha_ok": sha(Path(p).parent / "eval_audit.npz") == r["eval_file_sha256"],
                          "head_digest": r["head_digest"], "status": r["status"], "attempt": r["attempt"],
                          "mlflow_run_id": r["mlflow_run_id"], "ranks": r["ranks"],
                          "adapter_params": r["metrics"]["adapter_params"], "steps_curve": None,
                          "microbatch": r["microbatch"], "accumulation": r["accumulation"],
                          "U": hl - L}
    runs[r["arch_id"]]["steps_curve"] = len(json.loads((Path(p).parent / "train_curve.json").read_text()))
out["runs"] = runs
out["n_runs"] = len(runs)
out["all_head_digests_identical"] = len({v["head_digest"] for v in runs.values()}) == 1
out["max_abs_loss_discrepancy"] = max(abs(v["recomputed_loss"] - v["recorded_loss"]) for v in runs.values())

# ------------------------------------------------------------------ architecture definitions
unit = cfg["model"]["rank_unit_parameters"]


def mname(l, p):
    return f"roberta.encoder.layer.{l}.attention.self.{p}"


arch = {}
a2 = sa["architectures"]["single_layer_rank2"]
for l in a2["layers"]:
    arch[f"L{l:02d}_qv_r2"] = {mname(l, "query"): 2, mname(l, "value"): 2}
for l in sa["architectures"]["single_layer_rank4"]["layers"]:
    arch[f"L{l:02d}_qv_r4"] = {mname(l, "query"): 4, mname(l, "value"): 4}
for a, b in sa["architectures"]["two_layer_rank2"]["pairs"]:
    arch[f"L{a:02d}+L{b:02d}_qv_r2"] = {mname(a, "query"): 2, mname(a, "value"): 2, mname(b, "query"): 2,
                                        mname(b, "value"): 2}
out["ranks_match_yaml"] = all(runs[k]["ranks"] == v for k, v in arch.items()) and len(arch) == 20
out["params_match"] = all(runs[k]["adapter_params"] == unit * sum(v.values()) for k, v in arch.items())

# ------------------------------------------------------------------ independent gate statistics
scores = {s["arch_id"]: s for s in ps["arch_scores"]}
S = ["gradient_norm", "independent_fisher", "conditional_newton", "group_rvm"]
NB = S[:3]
U = {k: runs[k]["U"] for k in arch}
cost = {k: unit * sum(v.values()) for k, v in arch.items()}
d = sa["meaningful_utility_diff_nats"]


def conc(score):
    ni = nc = 0
    for c in sorted(set(cost.values())):
        ks = sorted(k for k in arch if cost[k] == c)
        for a, b in itertools.combinations(ks, 2):
            du = U[a] - U[b]
            if abs(du) <= d:
                continue
            ni += 1
            ds = scores[a][score] - scores[b][score]
            nc += int(ds != 0 and np.sign(ds) == np.sign(du))
    return nc, ni


def top3(score):
    ch = []
    for c in sorted(set(cost.values())):
        ks = sorted((k for k in arch if cost[k] == c), key=lambda k: (-scores[k][score], k))
        ch += ks[:3]
    return ch, float(np.mean([U[k] for k in ch]))


rec = {}
for s in S:
    nc, ni = conc(s)
    ch, mu = top3(s)
    rec[s] = {"concordant": nc, "informative": ni, "concordance": nc / ni, "top3": ch, "top3_mean_U": mu}
best_nb_c = max(NB, key=lambda s: rec[s]["concordance"])
best_nb_t = max(NB, key=lambda s: rec[s]["top3_mean_U"])
inter = []
for a, b in sa["architectures"]["two_layer_rank2"]["pairs"]:
    pid, i, j = f"L{a:02d}+L{b:02d}_qv_r2", f"L{a:02d}_qv_r2", f"L{b:02d}_qv_r2"
    io = U[pid] - U[i] - U[j]
    ie = scores[pid]["group_rvm"] - scores[i]["group_rvm"] - scores[j]["group_rvm"]
    inter.append({"pair": pid, "I_oracle": io, "I_evidence": ie, "informative": abs(io) > d,
                  "correct": bool(ie != 0 and np.sign(ie) == np.sign(io))})
g1 = gates["gate1"]
crit = {c["criterion"]: c for c in g1["criteria"]}
out["gate1_recomputed"] = {
    "per_score": rec, "best_nonbayes_concordance": best_nb_c, "best_nonbayes_top3": best_nb_t,
    "rvm_concordance": rec["group_rvm"]["concordance"],
    "margin_concordance": rec["group_rvm"]["concordance"] - rec[best_nb_c]["concordance"],
    "margin_top3": rec["group_rvm"]["top3_mean_U"] - rec[best_nb_t]["top3_mean_U"],
    "interactions": inter, "n_informative_interactions": sum(i["informative"] for i in inter),
    "n_correct_informative": sum(i["informative"] and i["correct"] for i in inter),
    "head_only_accuracy": ha, "head_only_ce": hl,
}
r = out["gate1_recomputed"]
out["gate1_matches_recorded"] = {
    "informative_pairs": rec["group_rvm"]["informative"] == crit["informative_cost_matched_pairs"]["value"],
    "rvm_concordance": abs(r["rvm_concordance"] - crit["rvm_pairwise_concordance"]["value"]) < 1e-15,
    "margin_concordance": abs(r["margin_concordance"] - crit["rvm_concordance_minus_best_nonbayes"]["value"]) < 1e-12,
    "margin_top3": abs(r["margin_top3"] - crit["rvm_six_choice_top3_utility_margin_nats"]["value"]) < 1e-12,
    "interactions_correct": r["n_correct_informative"] == crit["nonadditivity_sign_correct"]["value"],
    "fresh_null": crit["fresh_null_activations"]["value"],
    "recorded_gate1_status": g1["status"], "recorded_overall": gates["overall_decision"],
}
# decision recomputed from the YAML thresholds
th = sa
c1 = (rec["group_rvm"]["informative"] >= th["min_informative_cost_matched_pairs"]
      and r["rvm_concordance"] >= th["min_rvm_concordance"]
      and r["margin_concordance"] >= th["min_concordance_margin_over_best_nonbayes"])
c2 = r["margin_top3"] >= th["min_six_choice_mean_utility_margin_nats"]
c3 = crit["fresh_null_activations"]["value"] <= th["max_fresh_null_activations"]
c4 = (r["n_correct_informative"] >= th["conditional_interactions"]["min_correct_if_rule_applicable"]
      if r["n_informative_interactions"] >= th["conditional_interactions"]["min_informative_for_rule"] else True)
out["gate1_pass_recomputed"] = bool(c1 and c2 and c3 and c4)
out["gate1_conditions_recomputed"] = {"c1_concordance": c1, "c2_top3": c2, "c3_null": c3, "c4_interactions": c4}

# ------------------------------------------------------------------ MLflow cross-check
try:
    from mlflow.tracking import MlflowClient

    cl = MlflowClient("http://127.0.0.1:5000")
    ml = {}
    for name in ["rvm-lora/gate0", "rvm-lora/gate1", "rvm-lora/gate2"]:
        e = cl.get_experiment_by_name(name)
        if e is None:
            ml[name] = None
            continue
        rs = cl.search_runs([e.experiment_id], max_results=1000)
        ml[name] = [{"run_id": x.info.run_id, "status": x.info.status, "name": x.info.run_name,
                     "tags": {k: v for k, v in x.data.tags.items() if not k.startswith("mlflow.")},
                     "metrics": x.data.metrics} for x in rs]
    out["mlflow_counts"] = {k: (len(v) if v is not None else None) for k, v in ml.items()}
    chk = []
    g1runs = {x["run_id"]: x for x in (ml["rvm-lora/gate1"] or [])}
    for k, v in runs.items():
        m = g1runs.get(v["mlflow_run_id"])
        if m is None:
            chk.append({"arch": k, "found": False})
            continue
        chk.append({"arch": k, "found": True, "status": m["status"], "tag_status": m["tags"].get("status"),
                    "audit_loss_equal": m["metrics"].get("audit_loss") == v["recorded_loss"],
                    "head_sha_tag": m["tags"].get("head_checkpoint_sha256"),
                    "git_sha": m["tags"].get("git_sha"), "protocol_sha256": m["tags"].get("protocol_sha256"),
                    "data_split_sha256": m["tags"].get("data_split_sha256"), "arch_tag": m["tags"].get("arch_id")})
    out["mlflow_gate1_runs"] = chk
    out["mlflow_summary"] = {
        "all_found": all(c["found"] for c in chk),
        "all_finished": all(c.get("status") == "FINISHED" for c in chk),
        "all_audit_loss_equal": all(c.get("audit_loss_equal") for c in chk),
        "head_sha_tags": sorted({c.get("head_sha_tag") for c in chk}),
        "git_shas": sorted({c.get("git_sha") for c in chk}),
        "protocol_shas": sorted({c.get("protocol_sha256") for c in chk}),
        "split_shas": sorted({c.get("data_split_sha256") for c in chk}),
        "arch_tags_match": all(c.get("arch_tag") == c["arch"] for c in chk),
    }
    out["mlflow_all_gate1_run_names_status"] = sorted((x["name"], x["status"], x["tags"].get("status"))
                                                     for x in (ml["rvm-lora/gate1"] or []))
    out["mlflow_gate0_runs"] = sorted((x["name"], x["status"], x["tags"].get("git_sha"), x["metrics"].get("n_passed"),
                                       x["metrics"].get("n_tests")) for x in (ml["rvm-lora/gate0"] or []))
except Exception as e:  # noqa: BLE001
    out["mlflow_error"] = repr(e)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: out[k] for k in ["gate1_matches_recorded", "gate1_pass_recomputed", "all_head_digests_identical",
                                      "max_abs_loss_discrepancy", "ranks_match_yaml", "params_match"]}, indent=1))
print(json.dumps(out["gate1_recomputed"], indent=1, default=str)[:4000])
print(json.dumps(out["provenance"], indent=1, default=str))
print(json.dumps(out.get("mlflow_summary", out.get("mlflow_error")), indent=1))
print(json.dumps(out["head_only"], indent=1))
