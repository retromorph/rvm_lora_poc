"""EXPLORATORY post-hoc diagnostics for the failure analysis (CPU only, logged artifacts only).

Nothing here feeds any gate; all quantities are labelled exploratory in the report. No new training,
no new held-out evaluation: inputs are the frozen QNLI/42 probe features and the logged Gate-1 results.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from .config import Config
from .pipeline import ART, REPORTS, probe_dir
from .selectors import Dictionary, architecture_scores
from .tracking import write_json


def gate1_diagnostics(cfg: Config) -> dict:
    sa = cfg["stage_a"]
    g1doc = json.loads((ART / "gate1" / "gate1.json").read_text())
    rows = {r["arch_id"]: r for r in g1doc["rows"]}
    head = json.loads((ART / "gate1" / "head_only_audit.json").read_text())
    pdir = probe_dir(sa["task"], sa["run_seed"])
    z = np.load(pdir / "features.npz")
    Phi, t = z["Phi"], z["t"]
    D = Dictionary(Phi.T @ Phi, Phi.T @ t, cfg.modules, cfg.n_dir, cfg.rank_unit_params)
    lam = json.loads((ART / "gate1" / "precomputed_scores.json").read_text())["lambda"]
    out: dict = {"label": "EXPLORATORY (post hoc, not part of any gate)",
                 "head_only_audit": {"loss": head["audit_loss"], "accuracy": head["audit_accuracy"],
                                     "ln2": float(np.log(2))}}
    # 1) scale: mean column norm of each architecture's columns vs realised utility, within strata
    per = []
    for a in cfg.architectures():
        ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
        cols, _, _ = D.columns(ranks)
        colnorm = np.sqrt(np.diag(D.K)[cols])
        whitened = (D.b[cols] ** 2 / np.diag(D.K)[cols])  # q^2/h per column (scale-free)
        per.append({"arch_id": a.arch_id, "parameters": a.parameters, "U": rows[a.arch_id]["U"],
                    "mean_column_norm": float(colnorm.mean()), "sum_whitened_q2_over_h": float(whitened.sum())})
    strat = {}
    for cost in sorted({p["parameters"] for p in per}):
        st = [p for p in per if p["parameters"] == cost]
        strat[str(cost)] = {k: float(spearmanr([p[k] for p in st], [p["U"] for p in st]).statistic)
                            for k in ("mean_column_norm", "sum_whitened_q2_over_h")}
    out["scale_vs_utility_spearman"] = strat
    out["per_architecture"] = per
    # 2) non-additivity signs of the non-Bayesian scores (grad norm and Fisher are exactly additive)
    inter = []
    for it in json.loads((ART / "gate1" / "gate1.json").read_text())["gate1"]["interactions"]:
        pid = it["pair"]
        i, j = pid.split("_")[0].split("+")
        s_ij, s_i, s_j = (rows[pid], rows[f"{i}_qv_r2"], rows[f"{j}_qv_r2"])
        inter.append({**it, "I_newton": s_ij["conditional_newton"] - s_i["conditional_newton"] - s_j["conditional_newton"],
                      "I_gradient_norm": s_ij["gradient_norm"] - s_i["gradient_norm"] - s_j["gradient_norm"]})
    out["interactions_all_scores"] = inter
    # 3) ARD exclusions in the full-architecture evidence fits
    excl = {}
    for a in cfg.architectures():
        ranks = tuple(a.ranks.get(m, 0) for m in cfg.modules)
        s = architecture_scores(D, ranks, lam, cfg.log_alpha_bounds)
        active = [m for m, r in zip(cfg.modules, ranks) if r > 0]
        excl[a.arch_id] = [m.split(".")[3] + ("Q" if m.endswith("query") else "V")
                           for m, bnd in zip(active, s["rvm_boundary"]) if bnd]
    out["rvm_modules_at_exclusion_bound"] = excl
    n_q = sum(x.endswith("Q") for v in excl.values() for x in v)
    n_v = sum(x.endswith("V") for v in excl.values() for x in v)
    out["exclusion_counts"] = {"query": n_q, "value": n_v}
    # 4) per-module whitened singleton strength of the first atom vs its raw gradient magnitude
    mods = []
    for m, name in enumerate(cfg.modules):
        c = m * cfg.n_dir
        mods.append({"module": name, "b0": float(D.b[c]), "K00": float(D.K[c, c]),
                     "q2_over_h_first_atom": float(D.b[c] ** 2 / D.K[c, c])})
    out["first_atom_per_module"] = mods
    write_json(REPORTS / "exploratory_diagnostics.json", out)
    return out
