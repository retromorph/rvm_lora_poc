"""Deterministic, mechanical gate evaluation from logged metrics and the frozen YAML thresholds.

Status precedence when combining criteria: FAIL > BLOCKED > INCONCLUSIVE > PASS. `NOT_EVALUABLE`
(interaction rule with < 3 informative effects) does not block. The report writer may describe but
never override these outputs.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

PASS, FAIL, INCONCLUSIVE, BLOCKED = "PASS", "FAIL", "INCONCLUSIVE", "BLOCKED"
NOT_EVALUABLE, NOT_RUN = "NOT_EVALUABLE", "NOT_RUN_GATE_FAILURE"
SCORES = ["gradient_norm", "independent_fisher", "conditional_newton", "group_rvm"]
NONBAYES = ["gradient_norm", "independent_fisher", "conditional_newton"]
B24_COMPETITORS = ["uniform_r1", "eva", "oplora", "conditional_newton"]


def combine(statuses: list[str]) -> str:
    s = [x for x in statuses if x != NOT_EVALUABLE]
    for level in (FAIL, BLOCKED, INCONCLUSIVE, NOT_RUN):
        if level in s:
            return level
    return PASS


def criterion(name: str, value, cutoff, comparator: str, status: str, source: str, **extra) -> dict:
    return {"criterion": name, "value": value, "cutoff": cutoff, "comparator": comparator, "status": status,
            "source": source, **extra}


# ============================================================================ Gate 0
def evaluate_gate0(checks: list[dict]) -> dict:
    """checks: [{name, status, measured, source}] from the test suite and CUDA/MLflow smoke tests."""
    if not checks:
        return {"status": BLOCKED, "criteria": [], "note": "no Gate-0 evidence found"}
    return {"status": combine([c["status"] for c in checks]), "criteria": checks}


# ============================================================================ Gate 1
def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def pairwise_concordance(rows: list[dict], score: str, min_diff: float) -> dict:
    """Within-cost-stratum pairs with |dU| > min_diff; score ties count as discordant (conservative)."""
    n_inf = conc = 0
    per_stratum = {}
    for cost in sorted({r["parameters"] for r in rows}):
        stratum = sorted([r for r in rows if r["parameters"] == cost], key=lambda r: r["arch_id"])
        ni = nc = 0
        for a, b in itertools.combinations(stratum, 2):
            du = a["U"] - b["U"]
            if not abs(du) > min_diff:
                continue
            ni += 1
            ds = a[score] - b[score]
            nc += int(ds != 0 and _sign(ds) == _sign(du))
        per_stratum[str(cost)] = {"informative_pairs": ni, "concordant": nc,
                                  "concordance": (nc / ni) if ni else None}
        n_inf += ni
        conc += nc
    return {"informative_pairs": n_inf, "concordant": conc, "concordance": (conc / n_inf) if n_inf else None,
            "per_stratum": per_stratum}


def top_k_mean_utility(rows: list[dict], score: str, k: int) -> dict:
    """Per stratum: top-k by score, ties broken by architecture ID (ascending); mean realised U."""
    chosen = []
    per = {}
    for cost in sorted({r["parameters"] for r in rows}):
        stratum = sorted([r for r in rows if r["parameters"] == cost], key=lambda r: (-r[score], r["arch_id"]))
        top = stratum[:k]
        per[str(cost)] = [r["arch_id"] for r in top]
        chosen.extend(top)
    return {"chosen": per, "mean_U": float(np.mean([r["U"] for r in chosen])), "n": len(chosen)}


def spearman(rows: list[dict], score: str) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for cost in sorted({r["parameters"] for r in rows}):
        st = [r for r in rows if r["parameters"] == cost]
        rho = spearmanr([r[score] for r in st], [r["U"] for r in st]).statistic if len(st) > 2 else None
        out[str(cost)] = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
    return out


def interactions(rows: list[dict], pairs: list[tuple[str, str, str]], min_diff: float) -> list[dict]:
    by = {r["arch_id"]: r for r in rows}
    out = []
    for pair_id, a, b in pairs:
        io = by[pair_id]["U"] - by[a]["U"] - by[b]["U"]
        ie = by[pair_id]["group_rvm"] - by[a]["group_rvm"] - by[b]["group_rvm"]
        informative = abs(io) > min_diff
        out.append({"pair": pair_id, "I_oracle": io, "I_evidence": ie, "informative": informative,
                    "correct_sign": bool(ie != 0 and _sign(ie) == _sign(io))})
    return out


def evaluate_gate1(rows: list[dict], pairs: list[tuple[str, str, str]], null: dict | None, cfg_stage_a: dict,
                   missing_reason: str | None = None) -> dict:
    """rows: one dict per architecture with arch_id, parameters, the four scores and realised U (or None)."""
    th = cfg_stage_a
    d = th["meaningful_utility_diff_nats"]
    src = "reports/architecture_scores.csv"
    if missing_reason or any(r.get("U") is None for r in rows) or len(rows) != th["architecture_count"]:
        st = BLOCKED if missing_reason == "infrastructure" else INCONCLUSIVE
        crit = [criterion("all_20_architectures_measured", sum(r.get("U") is not None for r in rows),
                          th["architecture_count"], "==", st, src)]
        return {"status": st, "criteria": crit}
    conc = {s: pairwise_concordance(rows, s, d) for s in SCORES}
    top = {s: top_k_mean_utility(rows, s, th["top_k_per_cost_stratum"]) for s in SCORES}
    rho = {s: spearman(rows, s) for s in SCORES}
    n_inf = conc["group_rvm"]["informative_pairs"]
    best_nb = max(NONBAYES, key=lambda s: (conc[s]["concordance"] if conc[s]["concordance"] is not None else -1))
    best_top = max(NONBAYES, key=lambda s: top[s]["mean_U"])
    rvm_c = conc["group_rvm"]["concordance"]
    nb_c = conc[best_nb]["concordance"]
    enough = n_inf >= th["min_informative_cost_matched_pairs"]
    crit = []
    crit.append(criterion("informative_cost_matched_pairs", n_inf, th["min_informative_cost_matched_pairs"], ">=",
                          PASS if enough else INCONCLUSIVE, src))
    crit.append(criterion("rvm_pairwise_concordance", rvm_c, th["min_rvm_concordance"], ">=",
                          INCONCLUSIVE if not enough else (PASS if rvm_c >= th["min_rvm_concordance"] else FAIL), src))
    margin = (rvm_c - nb_c) if (rvm_c is not None and nb_c is not None) else None
    crit.append(criterion("rvm_concordance_minus_best_nonbayes", margin,
                          th["min_concordance_margin_over_best_nonbayes"], ">=",
                          INCONCLUSIVE if not enough or margin is None else
                          (PASS if margin >= th["min_concordance_margin_over_best_nonbayes"] else FAIL), src,
                          best_nonbayes=best_nb))
    tm = top["group_rvm"]["mean_U"] - top[best_top]["mean_U"]
    crit.append(criterion("rvm_six_choice_top3_utility_margin_nats", tm,
                          th["min_six_choice_mean_utility_margin_nats"], ">=",
                          PASS if tm >= th["min_six_choice_mean_utility_margin_nats"] else FAIL, src,
                          best_nonbayes=best_top))
    if null is None:
        crit.append(criterion("fresh_null_activations", None, th["max_fresh_null_activations"], "<=", BLOCKED,
                              "artifacts/gate1/nulls.json"))
    else:
        a = null["fresh_activations"]
        crit.append(criterion("fresh_null_activations", a, th["max_fresh_null_activations"], "<=",
                              PASS if a <= th["max_fresh_null_activations"] else FAIL, "artifacts/gate1/nulls.json"))
    inter = interactions(rows, pairs, d)
    n_inf_int = sum(i["informative"] for i in inter)
    n_corr = sum(i["informative"] and i["correct_sign"] for i in inter)
    ci = th["conditional_interactions"]
    if n_inf_int >= ci["min_informative_for_rule"]:
        st = PASS if n_corr >= ci["min_correct_if_rule_applicable"] else FAIL
    else:
        st = NOT_EVALUABLE
    crit.append(criterion("nonadditivity_sign_correct", n_corr, ci["min_correct_if_rule_applicable"], ">=", st, src,
                          informative_effects=n_inf_int))
    return {"status": combine([c["status"] for c in crit]), "criteria": crit, "concordance": conc,
            "top3": top, "spearman": rho, "interactions": inter, "best_nonbayes_concordance": best_nb,
            "best_nonbayes_top3": best_top}


# ============================================================================ Gate 2
def paired_bootstrap(loss_rvm: dict, loss_base: dict, tasks: list[str], seeds: list[int], n_boot: int, seed: int,
                     lower_pct: float) -> dict:
    """Stratified paired-example bootstrap of mean_{task,seed}(L_base - L_rvm).

    For each resample, each task draws one index vector (with replacement) that is reused for both
    training seeds of that task; tasks are resampled independently (QNLI first, then SST-2).
    """
    rng = np.random.default_rng(seed)
    n = {t: len(loss_rvm[(t, seeds[0])]) for t in tasks}
    for t in tasks:
        for s in seeds:
            if len(loss_rvm[(t, s)]) != n[t] or len(loss_base[(t, s)]) != n[t]:
                raise ValueError("per-example arrays must align on the same validation IDs")
    diffs = {k: np.asarray(loss_base[k]) - np.asarray(loss_rvm[k]) for k in loss_rvm}
    stats = np.empty(n_boot)
    for bi in range(n_boot):
        vals = []
        for t in tasks:
            idx = rng.integers(0, n[t], n[t])
            for s in seeds:
                vals.append(diffs[(t, s)][idx].mean())
        stats[bi] = np.mean(vals)
    point = float(np.mean([diffs[k].mean() for k in diffs]))
    return {"point": point, "lower": float(np.percentile(stats, lower_pct)),
            "upper": float(np.percentile(stats, 100 - lower_pct)), "n_boot": n_boot, "seed": seed}


def evaluate_gate2(runs: dict, cfg_stage_b: dict, fidelity: dict, structures: dict, gate1_status: str,
                   per_example: dict | None = None) -> dict:
    """runs[(method, task, seed)] = {status, loss, accuracy, adapter_params, selection_wall_s, train_wall_s}.

    structures[(task, seed)] = {"group_rvm": ranks, "conditional_newton": ranks}; fidelity[method] = bool.
    per_example[(method, task, seed)] = per-example loss array aligned on validation idx.
    """
    if gate1_status != PASS:
        return {"status": NOT_RUN, "criteria": [], "note": f"Gate 1 status {gate1_status}; Gate 2 not run"}
    return gate2_criteria(runs, cfg_stage_b, fidelity, structures, per_example)


EXPLORATORY_OVERRIDE = "exploratory_override"


def evaluate_gate2_exploratory(runs: dict, cfg_stage_b: dict, fidelity: dict, structures: dict, gate1_status: str,
                               per_example: dict | None = None) -> dict:
    """Counterfactual Gate 2 for an explicitly authorised exploratory execution despite a non-PASS Gate 1.

    The criteria are the unchanged `gate2_criteria`; the result is what Gate 2 *would* have concluded if
    evaluated on its own. It carries no `status` key, so it can never be passed to `assemble` or read as the
    preregistered Gate-2 status, which stays `NOT_RUN_GATE_FAILURE` whenever Gate 1 did not pass.
    """
    core = gate2_criteria(runs, cfg_stage_b, fidelity, structures, per_example)
    standalone = core.pop("status")
    for c in core["criteria"]:
        c["execution_type"] = EXPLORATORY_OVERRIDE
    return {"execution_type": EXPLORATORY_OVERRIDE, "gate1_status": gate1_status,
            "preregistered_gate2_status": NOT_RUN if gate1_status != PASS else "NOT_EVALUATED_HERE",
            "exploratory_gate2_standalone_result": standalone,
            "label": "COUNTERFACTUAL: frozen Gate-2 criteria evaluated as if Gate 1 had passed; does not count "
                     "toward, and cannot change, the preregistered Go/No-Go decision", **core}


def gate2_criteria(runs: dict, cfg_stage_b: dict, fidelity: dict, structures: dict,
                   per_example: dict | None = None) -> dict:
    """The frozen Gate-2 criteria (Routes A/B, mandatory conditions), independent of the Gate-1 status."""
    sb = cfg_stage_b
    tasks, seeds, methods = sb["tasks"], sb["seeds"], sb["methods"]
    keys = [(m, t, s) for m in methods for t in tasks for s in seeds]
    missing = [k for k in keys if runs.get(k, {}).get("status") != "COMPLETED"]
    crit = []
    if missing:
        st = BLOCKED if any(runs.get(k, {}).get("status") == BLOCKED for k in missing) else INCONCLUSIVE
        crit.append(criterion("all_24_runs_completed", len(keys) - len(missing), len(keys), "==", st,
                              "reports/run_index.csv", missing=[list(map(str, k)) for k in missing]))
        return {"status": st, "criteria": crit}
    L = {m: float(np.mean([runs[(m, t, s)]["loss"] for t in tasks for s in seeds])) for m in methods}
    best = min(B24_COMPETITORS, key=lambda m: (L[m], B24_COMPETITORS.index(m)))
    ra, rb = sb["route_a"], sb["route_b"]
    adv = L[best] - L["group_rvm"]
    pair_diffs = {f"{t}/{s}": runs[(best, t, s)]["loss"] - runs[("group_rvm", t, s)]["loss"]
                  for t in tasks for s in seeds}
    wins = sum(v > 0 for v in pair_diffs.values())
    boot = None
    if per_example is not None:
        boot = paired_bootstrap({(t, s): per_example[("group_rvm", t, s)] for t in tasks for s in seeds},
                                {(t, s): per_example[(best, t, s)] for t in tasks for s in seeds},
                                tasks, seeds, sb["bootstrap_resamples"], sb["bootstrap_seed"],
                                sb["bootstrap_lower_percentile"])
    src = "artifacts/gate2/per_run_metrics.json"
    a1 = adv >= ra["min_loss_advantage_nats"]
    a2 = wins >= ra["min_winning_task_seed_pairs"]
    a3 = (boot is not None) and boot["lower"] > 0
    crit += [
        criterion("routeA_aggregate_loss_advantage_vs_best_B24", adv, ra["min_loss_advantage_nats"], ">=",
                  PASS if a1 else FAIL, src, best_baseline=best),
        criterion("routeA_task_seed_wins", wins, ra["min_winning_task_seed_pairs"], ">=", PASS if a2 else FAIL, src,
                  pair_differences=pair_diffs),
        criterion("routeA_bootstrap_90pct_lower_bound", boot["lower"] if boot else None, 0.0, ">",
                  (PASS if a3 else FAIL) if boot else INCONCLUSIVE, "artifacts/gate2/bootstrap.json", bootstrap=boot),
    ]
    ex = L["group_rvm"] - L["uniform_r2"]
    nadv = L["conditional_newton"] - L["group_rvm"]
    task_ex = {t: float(np.mean([runs[("group_rvm", t, s)]["loss"] - runs[("uniform_r2", t, s)]["loss"]
                                 for s in seeds])) for t in tasks}
    b1 = ex <= rb["max_rvm_excess_vs_uniform_r2_nats"]
    b2 = nadv >= rb["min_rvm_advantage_vs_conditional_newton_nats"]
    b3 = all(v <= rb["max_taskwise_excess_vs_uniform_r2_nats"] for v in task_ex.values())
    crit += [
        criterion("routeB_rvm_excess_vs_uniform_r2", ex, rb["max_rvm_excess_vs_uniform_r2_nats"], "<=",
                  PASS if b1 else FAIL, src),
        criterion("routeB_rvm_advantage_vs_conditional_newton", nadv, rb["min_rvm_advantage_vs_conditional_newton_nats"],
                  ">=", PASS if b2 else FAIL, src),
        criterion("routeB_taskwise_excess_vs_uniform_r2", max(task_ex.values()),
                  rb["max_taskwise_excess_vs_uniform_r2_nats"], "<=", PASS if b3 else FAIL, src, per_task=task_ex),
    ]
    differs = {f"{t}/{s}": structures[(t, s)]["group_rvm"] != structures[(t, s)]["conditional_newton"]
               for t in tasks for s in seeds}
    any_diff = any(differs.values())
    spec_a = any_diff and nadv > 0
    spec_b = any_diff and nadv >= rb["min_rvm_advantage_vs_conditional_newton_nats"]
    crit.append(criterion("mandatory_rvm_specificity", {"structures_differ": differs, "newton_minus_rvm_loss": nadv},
                          "differs in >=1 config and Newton-RVM > 0 (A) / >= 0.005 (B)", "rule",
                          PASS if (spec_a or spec_b) else FAIL, "artifacts/gate2/structures.json",
                          route_a_ok=spec_a, route_b_ok=spec_b))
    sel = [runs[("group_rvm", t, s)]["selection_wall_s"] for t in tasks for s in seeds]
    tr = [runs[("uniform_r1", t, s)]["train_wall_s"] for t in tasks for s in seeds]
    ratio = float(np.mean(sel) / np.median(tr))
    ov = ratio <= sb["mandatory"]["max_calibration_to_uniform_train_walltime"]
    crit.append(criterion("mandatory_overhead_ratio", ratio, sb["mandatory"]["max_calibration_to_uniform_train_walltime"],
                          "<=", PASS if ov else FAIL, src, mean_rvm_selection_s=float(np.mean(sel)),
                          median_uniform_r1_train_s=float(np.median(tr))))
    budget_ok = all(runs[(m, t, s)]["adapter_params"] <= sb["budget_parameters"]
                    for m in methods if m != "uniform_r2" for t in tasks for s in seeds) and all(
        runs[("uniform_r2", t, s)]["adapter_params"] == sb["comparison_parameters"] for t in tasks for s in seeds)
    faithful = all(fidelity.get(m, False) for m in methods)
    integ = budget_ok and faithful
    crit.append(criterion("mandatory_baseline_integrity", {"budget_ok": budget_ok, "fidelity": fidelity}, "all true",
                          "rule", PASS if integ else INCONCLUSIVE, "artifacts/gate2/fidelity.json"))
    route_a = a1 and a2 and a3 and spec_a
    route_b = b1 and b2 and b3 and spec_b
    if not integ:
        status = INCONCLUSIVE
    elif (route_a or route_b) and ov:
        status = PASS
    else:
        status = FAIL
    return {"status": status, "criteria": crit, "L": L, "best_b24_baseline": best, "route_a": route_a,
            "route_b": route_b, "bootstrap": boot}


# ============================================================================ overall
def overall_decision(g0: str, g1: str, g2: str) -> str:
    if g0 == PASS and g1 == PASS and g2 == PASS:
        return "GO_TO_3B_REPLICATION"
    if g0 != PASS:  # implementation not validated: the scientific question was not tested
        return "BLOCKED" if g0 == BLOCKED else "INCONCLUSIVE"
    statuses = [g0, g1, g2]
    if FAIL in statuses:
        return "NO_GO"
    if BLOCKED in statuses:
        return "BLOCKED"
    return "INCONCLUSIVE"


def assemble(g0: dict, g1: dict, g2: dict, meta: dict[str, Any]) -> dict:
    return {"gate0": g0, "gate1": g1, "gate2": g2,
            "overall_decision": overall_decision(g0["status"], g1["status"], g2["status"]), "meta": meta}
