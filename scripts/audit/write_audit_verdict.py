"""Machine-readable correctness-audit verdict: reports/v2/audit_verdict.json (values read from logged files).

argv: <repo>
"""

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
g1, g2 = json.loads((REPO / "reports/gates.json").read_text()), json.loads((REPO / "reports/v2/gates.json").read_text())
fr1, fr2 = (json.loads((REPO / p).read_text()) for p in ("reports/protocol_freeze.json", "reports/v2/protocol_freeze.json"))
cmp_ = json.loads((REPO / "reports/audit/v1_v2_comparison.json").read_text())
p4 = json.loads((REPO / "reports/audit/phase4_evidence.json").read_text())
crit = lambda g: {c["criterion"]: {"value": c["value"], "cutoff": c["cutoff"], "status": c["status"]}  # noqa: E731
                  for c in g["gate1"]["criteria"]}
out = {
    "audit_document": "reports/CORRECTNESS_AUDIT.md",
    "repair_status": "PROTOCOL_PRESERVING_REPAIR_APPLIED_AND_RERUN",
    "frozen_protocol_files_unchanged": all(fr1[k] == fr2[k] for k in ("config_sha256", "preregistration_md_sha256",
                                                                       "clarifications_md_sha256", "methods_md_sha256")),
    "issues": [
        {"id": "A1", "classification": "CONFIRMED_IMPLEMENTATION_BUG", "material": True, "repaired": True,
         "summary": "RVM precision optimiser returned local optima of the Type-II evidence",
         "files": ["src/rvm_lora/evidence.py (optimize_evidence)", "src/rvm_lora/selectors.py (RVMObjective, architecture_scores)"],
         "v1_underestimated_architectures": {r["arch_id"]: r["dense_minus_recorded"] for r in p4["gate1_arch_evidence"]
                                             if r["dense_minus_recorded"] > 1e-6},
         "repair_commit": "d1aa145", "regression_tests": ["tests/test_evidence.py::test_optimizer_reaches_global_maximum_on_real_subproblems",
                                                          "tests/test_evidence.py::test_optimizer_matches_dense_search_scale_disparate"]},
        {"id": "A2", "classification": "CONFIRMED_IMPLEMENTATION_BUG", "material": False, "repaired": True,
         "summary": "stage caches reused without head-hash / code-version provenance (did not affect v1)",
         "files": ["src/rvm_lora/pipeline.py", "src/rvm_lora/cli.py", "src/rvm_lora/tracking.py"],
         "repair_commit": "d1aa145", "regression_tests": ["tests/test_cache.py"]},
        {"id": "H", "classification": "INTENTIONAL_PROTOCOL_DESIGN", "material": None, "repaired": False,
         "summary": "prescribed frozen-backbone 200-step head warm start yields a near-chance head (implementation verified)"},
        {"id": "L", "classification": "INTENTIONAL_PROTOCOL_DESIGN", "material": None, "repaired": False,
         "summary": "gradient-SVD evidence atoms vs random zero-update LoRA initial subspace (mapping verified correct)"},
        {"id": "N", "classification": "INTENTIONAL_PROTOCOL_DESIGN", "material": None, "repaired": False,
         "summary": "N(0,I) null pseudo-responses ignore the head's calibration offset"},
        {"id": "U2", "classification": "UNVERIFIED", "material": None, "repaired": False,
         "summary": "Gate-2 training paths never executed (Gate 2 not run in v1 or v2)"},
    ],
    "original_v1": {"decision": g1["overall_decision"], "gates": {g: g1[g]["status"] for g in ("gate0", "gate1", "gate2")},
                    "gate1_criteria": crit(g1), "freeze_git_sha": fr1["git_sha"], "gates_file": "reports/gates.json"},
    "corrected_v2": {"decision": g2["overall_decision"], "gates": {g: g2[g]["status"] for g in ("gate0", "gate1", "gate2")},
                     "gate1_criteria": crit(g2), "freeze_git_sha": fr2["git_sha"], "gates_file": "reports/v2/gates.json",
                     "full_adapter_trainings": 20, "max_allowed": 44},
    "decision_changed": g1["overall_decision"] != g2["overall_decision"],
    "rerun_bitwise_identical": {"trainings": cmp_["all_logits_bitwise_equal"] and cmp_["all_train_curves_bitwise_equal"],
                                "head_only_logits": cmp_["head_only_logits_bitwise_equal"],
                                "probe_features": cmp_["probe_hashes_equal"], "non_bayes_scores": cmp_["nonrvm_scores_identical"]},
    "changed_rvm_scores": cmp_["arch_scores_changed"],
    "exploratory_svd_init": {"label": "EXPLORATORY_NOT_A_GATE",
                             "concordance": cmp_.get("EXPLORATORY_svdinit", {}).get("concordance"),
                             "top3_mean_U": cmp_.get("EXPLORATORY_svdinit", {}).get("top3_mean_U")},
    "answers": {
        "original_experiments_implemented_incorrectly": "PARTLY: one material defect (A1) in the RVM evidence optimiser",
        "valid_protocol_equivalent_rerun_completed": True,
        "rerun_caveat": "post-hoc corrected re-evaluation after v1 audit outcomes were known",
        "corrected_results_change_decision": False,
    },
}
(REPO / "reports/v2/audit_verdict.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("repair_status", "frozen_protocol_files_unchanged", "decision_changed",
                                      "rerun_bitwise_identical")}, indent=1))
