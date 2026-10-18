"""Report automation: run index (all planned runs), architecture scores, figures, FINAL_REPORT.md, bundle.

The report only *describes* `reports/gates.json`; every status/decision string is copied from it.
Decision quantities are printed at full precision (repr); rounded values are for presentation only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .config import RUN_NAMESPACE, Config, sha256_file
from .pipeline import ART, REPORTS, ROOT, git_sha, now, probe_dir
from .tracking import mirror_dir, read_mirror, run_key

FIG = REPORTS / "figures"
REL = str(REPORTS.relative_to(ROOT))  # "reports" (v1) or "reports/<namespace>"
ARTREL = str(ART.relative_to(ROOT))
# Reference categorical palette (dataviz skill, light mode), fixed slot order.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCORE_LABEL = {"gradient_norm": "Gradient norm", "independent_fisher": "Independent Fisher",
               "conditional_newton": "Conditional Newton", "group_rvm": "Group RVM"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) >= 10 ** (-nd) or x == 0 else f"{x:.3e}"
    return str(x)


def _full(x):
    return "—" if x is None else repr(float(x)) if isinstance(x, (float, np.floating)) else str(x)


# =============================================================================== run index
def run_index(cfg: Config) -> list[dict]:
    g1 = _load(ART / "gate1" / "gate1.json")
    g1s = g1["gate1"]["status"] if g1 else None
    rows = []
    sa = cfg["stage_a"]
    for a in cfg.architectures():
        key = run_key(cfg.sha256, "gate1", "fixed_architecture", sa["task"], sa["run_seed"], a.arch_id)
        m = read_mirror(ROOT, "gate1", key)
        rows.append(_row("gate1", "fixed_architecture", sa["task"], sa["run_seed"], a.arch_id, key, m,
                         "audit", "PLANNED_NOT_STARTED"))
    sb = cfg["stage_b"]
    for t in sb["tasks"]:
        for s in sb["seeds"]:
            for meth in sb["methods"]:
                key = run_key(cfg.sha256, "gate2", meth, t, s, meth)
                m = read_mirror(ROOT, "gate2", key)
                if m is None:
                    default = ("NOT_RUN_GATE_FAILURE" if g1s in ("FAIL", "INCONCLUSIVE") else
                               "BLOCKED" if g1s == "BLOCKED" else "PLANNED_NOT_STARTED")
                else:
                    default = "PLANNED_NOT_STARTED"
                rows.append(_row("gate2", meth, t, s, meth, key, m, "official_validation", default))
    return rows


def _row(gate, method, task, seed, arch, key, m, prefix, default):
    mt = (m or {}).get("metrics", {})
    return {"gate": gate, "method": method, "task": task, "seed": seed, "arch_id": arch, "run_key": key,
            "status": (m or {}).get("status", default), "mlflow_run_id": (m or {}).get("mlflow_run_id"),
            "attempt": (m or {}).get("attempt"), "adapter_params": mt.get("adapter_params"),
            "rank_sum": mt.get("rank_sum"), f"eval_loss": mt.get(f"{prefix}_loss"),
            "eval_accuracy": mt.get(f"{prefix}_accuracy"), "eval_split": prefix if m else None,
            "train_wall_s": mt.get("train_wall_s"), "selection_wall_s": mt.get("selection_wall_s"),
            "total_wall_s": mt.get("total_wall_s"), "peak_gpu_memory_mb": mt.get("peak_gpu_memory_mb"),
            "microbatch": (m or {}).get("microbatch"), "precision_train": (m or {}).get("precision_train"),
            "ranks_json": json.dumps((m or {}).get("ranks"), sort_keys=True) if m and m.get("ranks") is not None else None,
            "error": (m or {}).get("error"),
            "mirror": str(mirror_dir(ROOT, gate, key).relative_to(ROOT) / "result.json") if m else None}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


# =============================================================================== figures
def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(colors=INK2, labelsize=8)


def fig_gate1(rows: list[dict], g1: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    ok = [r for r in rows if r.get("U") is not None]
    if not ok:
        return out
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), facecolor=SURF)
    for j, cost in enumerate(sorted({r["parameters"] for r in ok})):
        st = sorted([r for r in ok if r["parameters"] == cost], key=lambda r: r["arch_id"])
        for i, score in enumerate(SCORE_LABEL):
            ax = axes[j, i]
            _style(ax)
            x = np.array([r[score] for r in st])
            y = np.array([r["U"] for r in st])
            color = C1 if cost == 6144 else C2
            ax.scatter(x, y, s=46, color=color, edgecolor=SURF, linewidth=1.5, zorder=3,
                       marker="o" if cost == 6144 else "s")
            top = set(g1["top3"][score]["chosen"][str(cost)])
            for r in st:
                if r["arch_id"] in top:
                    ax.annotate(r["arch_id"], (r[score], r["U"]), fontsize=7, color=INK2, xytext=(4, 3),
                                textcoords="offset points")
            rho = g1["spearman"][score][str(cost)]
            c = g1["concordance"][score]["per_stratum"][str(cost)]
            ax.set_title(f"{SCORE_LABEL[score]} · {cost:,} params\nSpearman {_fmt(rho, 3)} · concordance "
                         f"{_fmt(c['concordance'], 3)} ({c['informative_pairs']} pairs)", fontsize=9, color=INK)
            if i == 0:
                ax.set_ylabel("Realised audit utility U (nats)", fontsize=8, color=INK2)
            ax.set_xlabel("Precomputed score", fontsize=8, color=INK2)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            ax.xaxis.get_offset_text().set_fontsize(7)
            from matplotlib.ticker import MaxNLocator

            ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("Gate 1: precomputed structural scores vs realised audit utility (QNLI, seed 42); "
                 "labelled points = that score's top 3", fontsize=11, color=INK)
    fig.tight_layout()
    p = FIG / "gate1_scatter.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p.relative_to(ROOT)))

    # utility per architecture (bar), stratum colours
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURF)
    _style(ax)
    ax.grid(False, axis="x")
    order = sorted(ok, key=lambda r: (r["parameters"], r["arch_id"]))
    xs = np.arange(len(order))
    ax.bar(xs, [r["U"] for r in order], width=0.7, color=[C1 if r["parameters"] == 6144 else C2 for r in order],
           edgecolor=SURF, linewidth=2)
    ax.axhline(0, color="#9a9994", linewidth=1)
    ax.set_xticks(xs, [r["arch_id"] for r in order], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("U = audit CE(head only) − audit CE(arch)  [nats]", fontsize=8, color=INK2)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=C1, label="6,144 params"), Patch(color=C2, label="12,288 params")],
              fontsize=8, frameon=False)
    ax.set_title("Gate 1: realised audit utility of the 20 fixed architectures", fontsize=10, color=INK)
    fig.tight_layout()
    p = FIG / "gate1_utility.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p.relative_to(ROOT)))
    return out


def fig_null(sel: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURF)
    _style(ax)
    g = np.array(sel["group_rvm"]["null_max_gains"])
    ax.hist(g, bins=25, color=C1, edgecolor=SURF, linewidth=1)
    thr = sel["group_rvm"]["threshold"]
    ax.axvline(thr, color=INK, linewidth=1.5, linestyle="--")
    ax.annotate(f"95th pct threshold = {thr:.3e}", (thr, ax.get_ylim()[1] * 0.9), fontsize=8, color=INK,
                xytext=(5, 0), textcoords="offset points")
    tr = sel["group_rvm"]["trace"]
    if tr:
        real = tr[0]["gain_per_param"]
        ax.set_title(f"RVM null calibration (seed 1931, n=100). Real-data first-step gain/param = {real:.3e};\n"
                     f"fresh nulls (seed 1932) activating: {sel['group_rvm_fresh_null']['activations']}/100",
                     fontsize=9, color=INK)
    ax.set_xlabel("max initial log-evidence gain per adapter parameter under N(0, I)", fontsize=8, color=INK2)
    ax.set_ylabel("count", fontsize=8, color=INK2)
    fig.tight_layout()
    p = FIG / "gate1_null_calibration.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return [str(p.relative_to(ROOT))]


def fig_allocations(cfg: Config) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    seq = ListedColormap(["#f4f3f0", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281",
                          "#0d366b"])
    panels = []
    for pdir in sorted((ART / "probes").glob("*_s*")):
        sel = _load(pdir / "selection.json")
        op = _load(pdir / "oplora.json")
        if not sel:
            continue
        rows = {"Group RVM": sel["group_rvm"]["ranks"], "Cond. Newton (B24)": sel["conditional_newton"]["ranks"],
                "RVM no-logdet (ablation)": sel["rvm_no_logdet"]["ranks"]}
        if op:
            rows["OPLoRA stage 1"] = op["ranks"]
        panels.append((pdir.name, rows))
    if not panels:
        return []
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 1.6 + 1.3 * len(panels) * 1.0 + 0.6 * len(panels)),
                             facecolor=SURF, squeeze=False)
    for ax, (name, rows) in zip(axes[:, 0], panels):
        M = np.array([[rows[k].get(m, 0) for m in cfg.modules] for k in rows])
        im = ax.imshow(M, cmap=seq, vmin=0, vmax=8, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] >= 4 else INK)
        ax.set_yticks(range(len(rows)), list(rows), fontsize=8)
        ax.set_xticks(range(24), [f"L{m.split('.')[3]}{m[-5].upper() if m.endswith('query') else 'V'}"
                                  for m in cfg.modules], fontsize=7)
        ax.set_title(f"Selected ranks per module · {name} (units: {', '.join(f'{k}={sum(v.values())}' for k, v in rows.items())})",
                     fontsize=9, color=INK)
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8, label="rank")
    p = FIG / "allocations.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return [str(p.relative_to(ROOT))]


def fig_gate2(cfg: Config, idx_rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = [r for r in idx_rows if r["gate"] == "gate2" and r["status"] == "COMPLETED"]
    if not done:
        return []
    out = []
    methods = cfg["stage_b"]["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF)
    for ax, task in zip(axes, cfg["stage_b"]["tasks"]):
        _style(ax)
        for k, meth in enumerate(methods):
            rr = [r for r in done if r["method"] == meth and r["task"] == task]
            if not rr:
                continue
            x = [r["adapter_params"] for r in rr]
            y = [r["eval_loss"] for r in rr]
            ax.scatter(x, y, s=40, color=C1 if meth == "group_rvm" else INK2 if meth.startswith("uniform") else C2,
                       marker="o^sDPX"[k % 6], edgecolor=SURF, linewidth=1.2, zorder=3, label=meth)
        ax.set_title(f"{task.upper()}: official validation CE vs exact LoRA parameters (2 seeds)", fontsize=9, color=INK)
        ax.set_xlabel("trainable LoRA parameters", fontsize=8, color=INK2)
        ax.set_ylabel("official validation CE (nats)", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = FIG / "gate2_quality_vs_params.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p.relative_to(ROOT)))
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURF)
    _style(ax)
    sel = {m: np.mean([r["selection_wall_s"] or 0 for r in done if r["method"] == m]) for m in methods}
    trn = {m: np.mean([r["train_wall_s"] or 0 for r in done if r["method"] == m]) for m in methods}
    xs = np.arange(len(methods))
    ax.bar(xs - 0.2, [sel[m] for m in methods], width=0.38, color=C1, label="selection/calibration s", edgecolor=SURF)
    ax.bar(xs + 0.2, [trn[m] for m in methods], width=0.38, color=C2, label="adapter training s", edgecolor=SURF)
    ax.set_xticks(xs, methods, fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Gate 2: mean wall time per run", fontsize=9, color=INK)
    fig.tight_layout()
    p = FIG / "overhead.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p.relative_to(ROOT)))
    return out


# =============================================================================== markdown
def _gate_table(crit: list[dict]) -> str:
    lines = ["| Criterion | Comparator | Cutoff | Actual (full precision) | Status |", "|---|---|---|---|---|"]
    for c in crit:
        v = c["value"]
        vs = _full(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else json.dumps(v, default=str)
        lines.append(f"| {c['criterion']} | {c['comparator']} | {c['cutoff']} | {vs} | **{c['status']}** |")
    return "\n".join(lines)


def _gate0_section(g0: dict) -> str:
    rows = ["| Required check | Tests | Measured | Status |", "|---|---|---|---|"]
    for c in g0.get("criteria", []):
        meas = "; ".join(f"{k.split('__')[-1]}: " + ", ".join(f"{a}={_fmt(b, 6) if isinstance(b, float) else b}"
                                                               for a, b in v.items()
                                                               if not isinstance(b, dict) and not (isinstance(b, list) and len(b) > 6))
                         for k, v in c.get("measured", {}).items()) or "pass/fail assertions"
        rows.append(f"| {c['name']} | {len(c.get('tests', {}))} | {meas} | **{c['status']}** |")
    return "\n".join(rows)


def build_report(cfg: Config) -> dict:
    REPORTS.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    gates = _load(REPORTS / "gates.json")
    if gates is None:
        from .pipeline import write_gates

        gates = write_gates(cfg)
    g0, g1, g2 = gates["gate0"], gates["gate1"], gates["gate2"]
    idx = run_index(cfg)
    write_csv(REPORTS / "run_index.csv", idx)
    g1doc = _load(ART / "gate1" / "gate1.json")
    rows = g1doc["rows"] if g1doc else []
    sa = cfg["stage_a"]
    sel = _load(probe_dir(sa["task"], sa["run_seed"]) / "selection.json")
    pr = _load(probe_dir(sa["task"], sa["run_seed"]) / "probe_result.json")
    figs = []
    if rows:
        arch_rows = []
        for r in rows:
            arch_rows.append({**{k: r[k] for k in ("arch_id", "family", "parameters")},
                              "rank_pattern": json.dumps(r["ranks"], sort_keys=True),
                              **{k: r[k] for k in SCORE_LABEL}, "rvm_log_alpha": json.dumps(r["rvm_log_alpha"]),
                              "audit_loss": r["audit_loss"], "audit_accuracy": r["audit_accuracy"], "U": r["U"],
                              "status": r["status"], "mlflow_run_id": r["mlflow_run_id"],
                              "eligible_for_pairwise": r["U"] is not None})
        write_csv(REPORTS / "architecture_scores.csv", arch_rows)
        if g1.get("concordance"):
            summ = []
            for s in SCORE_LABEL:
                summ.append({"score": s, "concordance": g1["concordance"][s]["concordance"],
                             "informative_pairs": g1["concordance"][s]["informative_pairs"],
                             "concordant": g1["concordance"][s]["concordant"],
                             "spearman_6144": g1["spearman"][s]["6144"], "spearman_12288": g1["spearman"][s]["12288"],
                             "top3_mean_U": g1["top3"][s]["mean_U"],
                             "top3_6144": " ".join(g1["top3"][s]["chosen"]["6144"]),
                             "top3_12288": " ".join(g1["top3"][s]["chosen"]["12288"])})
            write_csv(REPORTS / "gate1_summary.csv", summ)
            figs += fig_gate1(rows, g1)
    if sel:
        figs += fig_null(sel)
    figs += fig_allocations(cfg)
    figs += fig_gate2(cfg, idx)
    if rows and all(r["U"] is not None for r in rows):
        from .diagnostics import gate1_diagnostics

        gate1_diagnostics(cfg)
    md = render_markdown(cfg, gates, idx, rows, sel, pr, figs)
    (REPORTS / "FINAL_REPORT.md").write_text(md)
    bundle = build_bundle(cfg)
    return {"figures": figs, "bundle": bundle}


def render_markdown(cfg, gates, idx, rows, sel, pr, figs) -> str:
    g0, g1, g2 = gates["gate0"], gates["gate1"], gates["gate2"]
    freeze = _load(REPORTS / "protocol_freeze.json") or {}
    env = _load(REPORTS / "environment.json") or {}
    from .pipeline import head_meta_path

    sa = cfg["stage_a"]
    hm = _load(head_meta_path(sa["task"], sa["run_seed"])) or {}
    narrative = _load(REPORTS / "narrative.json") or {}
    L = []
    A = L.append
    A("# RVM-LoRA pilot: final Go/No-Go assessment" +
      (f" — corrected run `{RUN_NAMESPACE}`\n" if RUN_NAMESPACE else "\n"))
    A(f"**Date:** {now()} (UTC)  ")
    A(f"**Overall decision (machine-evaluated, `{REL}/gates.json`):** **{gates['overall_decision']}**  ")
    A(f"**Gate 0:** {g0['status']} · **Gate 1:** {g1['status']} · **Gate 2:** {g2['status']}  ")
    A(f"**Git SHA (report build):** `{git_sha()}` · **Protocol freeze git SHA:** `{freeze.get('git_sha')}`  ")
    A(f"**Protocol YAML SHA256:** `{cfg.sha256}` · **uv.lock SHA256:** `{freeze.get('uv_lock_sha256')}`  ")
    A("**MLflow tracking URI:** remote-local `http://127.0.0.1:5000` (SQLite `mlruns/mlflow.db`, artifacts "
      "`mlruns/artifacts`), UI only through `ssh -N -L 15000:127.0.0.1:5000 <host>`  ")
    A("**MLflow experiments:** `rvm-lora/gate0`, `rvm-lora/gate1`, `rvm-lora/gate2` (run IDs in "
      f"`{REL}/run_index.csv` and below)\n")
    A("## Executive judgment\n")
    A(narrative.get("executive", "_(narrative not yet written)_") + "\n")
    # ------------------------------------------------------------------ integrity
    A("## Experimental integrity and environment\n")
    A("| Evidence item | Value / hash | Verified? |\n|---|---|---|")
    A(f"| Hardware GPU / VRAM / driver | {env.get('nvidia_smi')} | yes (`nvidia-smi`, torch) |")
    pk = env.get("packages", {})
    A(f"| OS / Python / torch / CUDA / Transformers / PEFT / MLflow | Ubuntu 26.04, kernel 7.0.0-15 / {env.get('python')} / "
      f"{pk.get('torch')} / CUDA runtime {env.get('torch_cuda')} / {pk.get('transformers')} / {pk.get('peft')} / "
      f"{pk.get('mlflow')}; BF16 supported={env.get('bf16_supported')}, BF16 fwd/bwd smoke={env.get('bf16_forward_backward_ok')} | yes |")
    A(f"| Model ID and exact revision | `{cfg['backbone']}@{freeze.get('model', {}).get('revision')}` | pinned in code |")
    sm = freeze.get("split_manifests", {})
    for t in cfg.tasks:
        s = sm.get(t, {})
        A(f"| {t.upper()} training/audit split hash | manifest `{s.get('manifest_file_sha256', '')[:16]}…`, "
          f"adapter_train `{s.get('adapter_train', '')[:16]}…`, audit `{s.get('audit', '')[:16]}…` | yes (tests) |")
    heads = []
    for p in sorted((ART / "heads").glob("*.json")):
        h = _load(p)
        heads.append(f"{h['task']}/{h['seed']}: `{h['head_checkpoint_sha256'][:16]}…`")
    A(f"| Classifier-head checkpoint hashes per task/seed | {'; '.join(heads) or '—'} | yes (hash re-checked on load) |")
    A(f"| Protocol freeze timestamp / git / YAML / MD | {freeze.get('frozen_at')} / `{freeze.get('git_sha')}` / "
      f"`{freeze.get('config_sha256', '')[:16]}…` / prereg `{freeze.get('preregistration_md_sha256', '')[:16]}…`, "
      f"clarifications `{freeze.get('clarifications_md_sha256', '')[:16]}…` | yes |")
    A(f"| Pilot run manager log and run index | `{LOG.relative_to(ROOT)}` (compact copy in bundle), `{REL}/run_index.csv` | yes |")
    A("| Evidence/calibration/official validation isolation | split tests (disjoint, train-only); audit read only "
      "after freeze and score precomputation; official validation read only in Gate 2 | "
      f"{'yes' if g0['status'] == 'PASS' else 'see Gate 0'} |")
    A("| Parameter-count assertions, LoRA-only gradients | asserted per run (`assert_exact_layout`, head digest "
      "before/after training) | yes |")
    A("| Baseline fidelity and source commits | PEFT 0.21.2 EVA (native); OPLoRA = DPLoRA stage 1 "
      "`cjpark-ai/DPLoRA@6666d8bdb37ebd153a8916b0464c8a3878976aae` (MIT), exact ILP port | see C6 in "
      "`docs/PROTOCOL_CLARIFICATIONS.md` |\n")
    # ------------------------------------------------------------------ gate 0
    A("## Gate 0: numerical and pipeline correctness\n")
    A(_gate0_section(g0))
    A(f"\n**Gate 0 determination:** {g0['status']} ({g0.get('n_passed')}/{g0.get('n_tests')} tests; "
      f"`{ARTREL}/gate0/gate0.json`, MLflow run `{g0.get('mlflow_run_id')}`). "
      + narrative.get("gate0", "") + "\n")
    # ------------------------------------------------------------------ gate 1
    A("## Gate 1: QNLI architecture oracle (20 full runs)\n")
    A("Each `U(A) = audit_CE(head only) − audit_CE(A)` uses the same frozen warm-started head (QNLI, seed 42) "
      "and the final 600-step checkpoint. All four scores were precomputed and logged "
      f"(`{ARTREL}/gate1/precomputed_scores.json`, MLflow probe run) before the first audit label was read"
      + (" in this run namespace (the original run's audit outcomes were already known; see "
         "`reports/CORRECTNESS_AUDIT.md`).\n" if RUN_NAMESPACE else ".\n"))
    if g1.get("head_only_audit_loss") is not None:
        A(f"Head-only audit CE: `{_full(g1['head_only_audit_loss'])}` nats.\n")
    if rows:
        A("| Architecture | Params | Grad norm | Indep. Fisher | Cond. Newton | Group RVM (log ev.) | Audit CE | "
          "Audit acc. | U (nats) | Status | MLflow run |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for r in sorted(rows, key=lambda r: (r["parameters"], r["arch_id"])):
            A(f"| {r['arch_id']} | {r['parameters']:,} | {_fmt(r['gradient_norm'], 3)} | "
              f"{_fmt(r['independent_fisher'], 3)} | {_fmt(r['conditional_newton'], 3)} | {_fmt(r['group_rvm'], 3)} | "
              f"{_fmt(r['audit_loss'], 5)} | {_fmt(r['audit_accuracy'], 4)} | {_fmt(r['U'], 5)} | {r['status']} | "
              f"`{r['mlflow_run_id']}` |")
        A(f"\nFull-precision values: `{REL}/architecture_scores.csv`.\n")
    if g1.get("concordance"):
        A("| Structural score | Pairwise concordance (concordant/informative) | Spearman 6,144 | Spearman 12,288 | "
          "Six-choice top-3 mean U | Margin vs best cheap (top-3) |")
        A("|---|---:|---:|---:|---:|---:|")
        best_top = g1["best_nonbayes_top3"]
        for s in SCORE_LABEL:
            c = g1["concordance"][s]
            m = g1["top3"][s]["mean_U"] - g1["top3"][best_top]["mean_U"]
            A(f"| {SCORE_LABEL[s]} | {_fmt(c['concordance'], 4)} ({c['concordant']}/{c['informative_pairs']}) | "
              f"{_fmt(g1['spearman'][s]['6144'], 3)} | {_fmt(g1['spearman'][s]['12288'], 3)} | "
              f"{_fmt(g1['top3'][s]['mean_U'], 5)} | {_fmt(m, 5)} |")
        A("\nTop-3 choices per stratum: " + "; ".join(
            f"{SCORE_LABEL[s]}: 6,144 → {', '.join(g1['top3'][s]['chosen']['6144'])}; 12,288 → "
            f"{', '.join(g1['top3'][s]['chosen']['12288'])}" for s in SCORE_LABEL) + "\n")
        A("**Interaction non-additivity (not raw evidence increments):**\n")
        A("| Layer pair | Realised U(i,j)−U(i)−U(j), nats | Predicted E(i,j)−E(i)−E(j), log evidence | Informative? | "
          "Correct sign? |\n|---|---:|---:|---|---|")
        for it in g1["interactions"]:
            A(f"| {it['pair']} | {_full(it['I_oracle'])} | {_full(it['I_evidence'])} | {it['informative']} | "
              f"{it['correct_sign'] if it['informative'] else 'n/a'} |")
    rn = g1.get("rvm_null", {})
    if rn:
        A(f"\n**Stopping/null:** threshold (95th percentile of 100 calibration nulls, seed 1931) = "
          f"`{_full(rn.get('threshold'))}` log-evidence nats per adapter parameter; fresh-null activations "
          f"(seed 1932) = **{rn.get('fresh_activations')}/100**; on the real QNLI features RVM selected "
          f"{rn.get('real_data_rank_units')} of 24 rank units (stop: {rn.get('real_data_stop')}), "
          f"ranks `{json.dumps(rn.get('real_data_ranks'))}`.\n")
    A("**Gate-1 mandatory criteria** (thresholds from `configs/preregistered.yaml`):\n")
    A(_gate_table(g1.get("criteria", [])))
    A(f"\n**Gate 1: {g1['status']}.** " + narrative.get("gate1", "") + "\n")
    for f in figs:
        if "gate1" in f:
            A(f"![{Path(f).stem}]({Path(f).relative_to('reports')})\n")
    # ------------------------------------------------------------------ gate 2
    A("## Gate 2: matched-budget comparisons\n")
    if g2["status"] == "NOT_RUN_GATE_FAILURE":
        A(f"**Gate 2: NOT_RUN_GATE_FAILURE.** {g2.get('note', '')}. Per the frozen conditional progression, none "
          "of the 24 Gate-2 trainings was started and the official GLUE validation splits were never evaluated. "
          f"All 24 rows are listed with this status in `{REL}/run_index.csv`; no Gate-2 number is reported or "
          "imputed.\n")
        A("| Method | Mean QNLI CE | Mean SST-2 CE | Mean accuracy | Actual LoRA params | Mean selection s | "
          "Mean adapter-train s | Peak GB |\n|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in cfg["stage_b"]["methods"]:
            A(f"| {m} | NOT_RUN_GATE_FAILURE | NOT_RUN_GATE_FAILURE | — | — | — | — | — |")
    else:
        A(_gate2_tables(cfg, idx, g2))
    A("")
    # ------------------------------------------------------------------ ablations / failure analysis
    if RUN_NAMESPACE:
        A(f"## Before/after comparison (original v1 → corrected `{RUN_NAMESPACE}`)\n")
        A(_before_after_section(cfg, gates))
        A(narrative.get("before_after", "") + "\n")
    A("## Failure analysis / critical ablations\n")
    A(_ablation_section(cfg))
    A(narrative.get("failure_analysis", "") + "\n")
    A("## Resource accounting\n")
    A(_resource_section(cfg, idx, hm, pr))
    A(narrative.get("resources", "") + "\n")
    A("## Deviations, missing evidence, and limitations\n")
    A(narrative.get("deviations", "_(see docs/PROTOCOL_CLARIFICATIONS.md)_") + "\n")
    A("## Final decision and next experiment\n")
    A(f"**Machine-evaluated decision:** `{gates['overall_decision']}` (from `{REL}/gates.json`).  ")
    A(narrative.get("decision", "") + "\n")
    A("## Reproduction and evidence index\n")
    A(_repro_section(cfg, freeze))
    A("\n## Appendix: all planned main runs\n")
    A("| Gate | Method | Task | Seed | Architecture | Status | Params | Eval CE | MLflow run |")
    A("|---|---|---|---|---|---|---:|---:|---|")
    for r in idx:
        A(f"| {r['gate']} | {r['method']} | {r['task']} | {r['seed']} | {r['arch_id']} | {r['status']} | "
          f"{_fmt(r['adapter_params'])} | {_fmt(r['eval_loss'], 5)} | {r['mlflow_run_id'] or '—'} |")
    return "\n".join(L) + "\n"


def _gate2_tables(cfg, idx, g2) -> str:
    out = []
    done = [r for r in idx if r["gate"] == "gate2"]
    out.append("| Method | Mean QNLI CE (seeds) | Mean SST-2 CE (seeds) | Mean accuracy | Actual LoRA params "
               "(per run) | Mean selection s | Mean adapter-train s | Peak GB |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in cfg["stage_b"]["methods"]:
        rr = [r for r in done if r["method"] == m]
        ok = [r for r in rr if r["status"] == "COMPLETED"]

        def mean(key, task=None):
            v = [r[key] for r in ok if (task is None or r["task"] == task) and r[key] is not None]
            return float(np.mean(v)) if v else None
        st = "" if len(ok) == len(rr) else f" ({len(ok)}/{len(rr)} completed)"
        out.append(f"| {m}{st} | {_fmt(mean('eval_loss', 'qnli'), 5)} | {_fmt(mean('eval_loss', 'sst2'), 5)} | "
                   f"{_fmt(mean('eval_accuracy'), 4)} | {', '.join(str(r['adapter_params']) for r in ok)} | "
                   f"{_fmt(mean('selection_wall_s'), 2)} | {_fmt(mean('train_wall_s'), 1)} | "
                   f"{_fmt((mean('peak_gpu_memory_mb') or 0) / 1024, 2)} |")
    out.append("\nPer task × seed official validation CE (full precision):\n")
    out.append("| Method | " + " | ".join(f"{t}/{s}" for t in cfg["stage_b"]["tasks"] for s in cfg["stage_b"]["seeds"])
               + " |")
    out.append("|---|" + "---:|" * 4)
    for m in cfg["stage_b"]["methods"]:
        cells = []
        for t in cfg["stage_b"]["tasks"]:
            for s in cfg["stage_b"]["seeds"]:
                r = [x for x in done if x["method"] == m and x["task"] == t and x["seed"] == s][0]
                cells.append(_full(r["eval_loss"]) if r["status"] == "COMPLETED" else r["status"])
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    if g2.get("criteria"):
        out.append(f"\n**Primary B24 competitor (post-hoc best):** {g2.get('best_b24_baseline')}. "
                   f"Mean losses: `{json.dumps(g2.get('L'))}`\n")
        out.append(_gate_table(g2["criteria"]))
    out.append(f"\n**Gate 2: {g2['status']}** (Route A: {g2.get('route_a')}, Route B: {g2.get('route_b')}).")
    return "\n".join(out)


def _before_after_section(cfg, gates) -> str:
    """Original (v1) vs corrected run, read from both artifact trees; never typed by hand."""
    v1g = _load(ROOT / "reports" / "gates.json") or {}
    v1d = _load(ROOT / "artifacts" / "gate1" / "gate1.json") or {}
    v2d = _load(ART / "gate1" / "gate1.json") or {}
    v1h = _load(ROOT / "artifacts" / "gate1" / "head_only_audit.json") or {}
    v2h = _load(ART / "gate1" / "head_only_audit.json") or {}
    v1hm = _load(ROOT / "artifacts" / "heads" / "qnli_s42.json") or {}
    from .pipeline import head_meta_path

    v2hm = _load(head_meta_path(cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"])) or {}
    out = ["| Quantity | Original v1 | Corrected | Equal? |", "|---|---|---|---|"]

    def row(name, a, b):
        out.append(f"| {name} | {_full(a) if isinstance(a, float) else a} | {_full(b) if isinstance(b, float) else b} | "
                   f"{'yes' if a == b else 'no'} |")

    row("Overall decision", v1g.get("overall_decision"), gates.get("overall_decision"))
    for g in ("gate0", "gate1", "gate2"):
        row(f"{g} status", v1g.get(g, {}).get("status"), gates.get(g, {}).get("status"))
    row("Head checkpoint SHA256", v1hm.get("head_checkpoint_sha256"), v2hm.get("head_checkpoint_sha256"))
    row("Head train loss first20/last20", f"{v1hm.get('head_train_loss_first20')}/{v1hm.get('head_train_loss_last20')}",
        f"{v2hm.get('head_train_loss_first20')}/{v2hm.get('head_train_loss_last20')}")
    row("Head-only audit CE", v1h.get("audit_loss"), v2h.get("audit_loss"))
    row("Head-only audit accuracy", v1h.get("audit_accuracy"), v2h.get("audit_accuracy"))
    c1 = {c["criterion"]: c for c in v1g.get("gate1", {}).get("criteria", [])}
    c2 = {c["criterion"]: c for c in gates.get("gate1", {}).get("criteria", [])}
    for k in c1.keys() | c2.keys():
        row(f"Gate-1 `{k}` value", c1.get(k, {}).get("value"), c2.get(k, {}).get("value"))
        row(f"Gate-1 `{k}` status", c1.get(k, {}).get("status"), c2.get(k, {}).get("status"))
    g1a, g1b = v1g.get("gate1", {}), gates.get("gate1", {})
    for s in SCORE_LABEL:
        ca, cb = g1a.get("concordance", {}).get(s, {}), g1b.get("concordance", {}).get(s, {})
        row(f"{SCORE_LABEL[s]} concordance", f"{ca.get('concordance')} ({ca.get('concordant')}/{ca.get('informative_pairs')})",
            f"{cb.get('concordance')} ({cb.get('concordant')}/{cb.get('informative_pairs')})")
        ta, tb = g1a.get("top3", {}).get(s, {}), g1b.get("top3", {}).get(s, {})
        row(f"{SCORE_LABEL[s]} top-3 mean U", ta.get("mean_U"), tb.get("mean_U"))
        row(f"{SCORE_LABEL[s]} top-3 choices", json.dumps(ta.get("chosen")), json.dumps(tb.get("chosen")))
    ia = {i["pair"]: i for i in g1a.get("interactions", [])}
    ib = {i["pair"]: i for i in g1b.get("interactions", [])}
    for p in ia.keys() | ib.keys():
        row(f"{p} I_oracle", ia.get(p, {}).get("I_oracle"), ib.get(p, {}).get("I_oracle"))
        row(f"{p} I_evidence (sign correct?)", f"{ia.get(p, {}).get('I_evidence')} ({ia.get(p, {}).get('correct_sign')})",
            f"{ib.get(p, {}).get('I_evidence')} ({ib.get(p, {}).get('correct_sign')})")
    out.append("\n| Architecture | U v1 | U corrected | RVM evidence v1 | RVM evidence corrected | Newton v1 | Newton corrected |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    ra = {r["arch_id"]: r for r in v1d.get("rows", [])}
    rb = {r["arch_id"]: r for r in v2d.get("rows", [])}
    for a in cfg.architectures():
        x, y = ra.get(a.arch_id, {}), rb.get(a.arch_id, {})
        out.append(f"| {a.arch_id} | {_full(x.get('U'))} | {_full(y.get('U'))} | {_full(x.get('group_rvm'))} | "
                   f"{_full(y.get('group_rvm'))} | {_full(x.get('conditional_newton'))} | {_full(y.get('conditional_newton'))} |")
    sa = cfg["stage_a"]
    s1 = _load(ROOT / "artifacts" / "probes" / f"{sa['task']}_s{sa['run_seed']}" / "selection.json") or {}
    s2 = _load(probe_dir(sa["task"], sa["run_seed"]) / "selection.json") or {}
    out.append("\n| Selector (QNLI/42, real features) | v1 | corrected |\n|---|---|---|")
    for name, k in (("Group RVM ranks", "group_rvm"), ("Conditional Newton (B24) ranks", "conditional_newton"),
                    ("No-logdet ablation ranks", "rvm_no_logdet")):
        out.append(f"| {name} | `{json.dumps(s1.get(k, {}).get('ranks'), sort_keys=True)}` | "
                   f"`{json.dumps(s2.get(k, {}).get('ranks'), sort_keys=True)}` |")
    cn1, cn2 = s1.get("conditional_newton", {}), s2.get("conditional_newton", {})
    out.append(f"| Newton with own null threshold (diagnostic) | `{json.dumps(cn1.get('diagnostic_thresholded_ranks'), sort_keys=True)}` | "
               f"`{json.dumps(cn2.get('diagnostic_thresholded_ranks'), sort_keys=True)}` |")
    out.append(f"| RVM null threshold (nats/param) | {_full(s1.get('group_rvm', {}).get('threshold'))} | "
               f"{_full(s2.get('group_rvm', {}).get('threshold'))} |")
    out.append(f"| RVM fresh-null activations | {s1.get('group_rvm_fresh_null', {}).get('activations')} | "
               f"{s2.get('group_rvm_fresh_null', {}).get('activations')} |")
    out.append(f"| Thresholded Newton == RVM | {cn1.get('diagnostic_thresholded_ranks') == s1.get('group_rvm', {}).get('ranks')} | "
               f"{cn2.get('diagnostic_thresholded_ranks') == s2.get('group_rvm', {}).get('ranks')} |")
    return "\n".join(out) + "\n"


def _ablation_section(cfg) -> str:
    out = ["| Probe | Selector | Rank units | Structure (module: rank) | Stop |", "|---|---|---:|---|---|"]
    for pdir in sorted((ART / "probes").glob("*_s*")):
        sel = _load(pdir / "selection.json")
        op = _load(pdir / "oplora.json")
        if not sel:
            continue

        def short(r):
            return ", ".join(f"L{k.split('.')[3]}{'Q' if k.endswith('query') else 'V'}:{v}" for k, v in sorted(
                r.items(), key=lambda kv: cfg.modules.index(kv[0])))
        for name, key in (("Group RVM", "group_rvm"), ("Conditional Newton (B24)", "conditional_newton"),
                          ("RVM no-logdet (offline ablation)", "rvm_no_logdet")):
            s = sel[key]
            out.append(f"| {pdir.name} | {name} | {s['rank_units']} | {short(s['ranks'])} | {s['stop_reason']} |")
        cn = sel["conditional_newton"]
        out.append(f"| {pdir.name} | Newton with own null threshold (diagnostic) | {cn['diagnostic_thresholded_units']} | "
                   f"{short(cn['diagnostic_thresholded_ranks'])} | threshold {_full(cn['diagnostic_null_threshold'])} |")
        if op:
            out.append(f"| {pdir.name} | OPLoRA stage 1 | {sum(op['ranks'].values())} | {short(op['ranks'])} | ILP |")
        same_nl = sel["rvm_no_logdet"]["ranks"] == sel["conditional_newton"]["ranks"]
        same_rn = sel["group_rvm"]["ranks"] == sel["conditional_newton"]["ranks"]
        out.append(f"| {pdir.name} | identity checks | — | RVM == Newton: {same_rn}; no-logdet == Newton: {same_nl} | "
                   f"λ = {cn['lambda']} |")
    return "\n".join(out) + "\n"


def _resource_section(cfg, idx, hm, pr) -> str:
    done = [r for r in idx if r["status"] == "COMPLETED"]
    tw = sum(r["train_wall_s"] or 0 for r in done)
    peak = max([r["peak_gpu_memory_mb"] or 0 for r in done] or [0])
    out = [f"- Completed main adapter trainings: {len(done)} (Gate 1: {sum(r['gate'] == 'gate1' for r in done)}, "
           f"Gate 2: {sum(r['gate'] == 'gate2' for r in done)}); summed 600-step training wall time "
           f"{tw:.1f} s = {tw / 3600:.3f} GPU-hours (sequential, one GPU job at a time).",
           f"- Peak CUDA allocation across runs: {peak:.0f} MB; training precision BF16 autocast, evaluation FP32 "
           f"(TF32 disabled), selection linear algebra float64 on CPU.",
           f"- QNLI/42 head warm start: {_fmt(hm.get('head_wall_s'), 1)} s (excluded from the overhead ratio).",
           f"- OOM fallbacks: {sum(1 for r in done if r.get('microbatch') == 4)} runs used microbatch 4 × 8."]
    if pr:
        t = pr["timings"]
        out.append(f"- QNLI/42 probe: direction pass {t['direction_pass_s']:.2f} s, feature pass "
                   f"{t['feature_pass_s']:.2f} s, OPLoRA importance pass {t['oplora_importance_pass_s']:.2f} s, "
                   f"selection subprocess {t['selection_job_wall_s']:.2f} s (start-up {t['selection_job_startup_s']:.2f} s); "
                   f"RVM selection cost as charged {t['group_rvm_selection_s']:.2f} s, Newton "
                   f"{t['conditional_newton_selection_s']:.2f} s, OPLoRA {t['oplora_selection_s']:.2f} s.")
    return "\n".join(out) + "\n"


def _repro_section(cfg, freeze) -> str:
    return "\n".join([
        f"- `configs/preregistered.yaml` SHA256 `{cfg.sha256}`; `docs/PREREGISTRATION.md` SHA256 "
        f"`{freeze.get('preregistration_md_sha256')}`; freeze record `{REL}/protocol_freeze.json`.",
        f"- MLflow run IDs: `{REL}/run_index.csv` (column `mlflow_run_id`) and the tables above.",
        f"- `{REL}/gates.json` (mechanical evaluator output), `{REL}/run_index.csv`, "
        f"`{REL}/architecture_scores.csv`, `{REL}/gate1_summary.csv`, `{REL}/figures/`.",
        "- Exact shell commands (fresh checkout on the GPU host):\n\n```bash\n"
        "bash scripts/bootstrap_remote.sh            # uv, Python 3.11, uv sync --frozen, dry run\n"
        "bash scripts/start_mlflow.sh                # MLflow on 127.0.0.1:5000 in tmux 'rvm_mlflow'\n"
        "export MLFLOW_TRACKING_URI=http://127.0.0.1:5000\n"
        "uv run --frozen python -m rvm_lora.cli env-check\n"
        "uv run --frozen pytest -q -m 'not post_hoc'\n"
        (f"export RVM_RUN_NAMESPACE={RUN_NAMESPACE}            # isolated, versioned outputs\n" if RUN_NAMESPACE else "")
        + "tmux new-session -d -s rvm_pilot 'bash scripts/run_remote.sh'   # full frozen gate chain\n"
        "uv run --frozen python -m rvm_lora.cli status\n"
        "uv run --frozen python -m rvm_lora.cli report   # gates.json, CSVs, figures, FINAL_REPORT.md, bundle\n"
        "uv run --frozen pytest -q -m post_hoc           # gates.json reproducible from logged artifacts\n```",
        f"- Review bundle: `{REL}/review_bundle.zip`; its SHA256 is in `{REL}/review_bundle.sha256` (the bundle "
        "contains this report, so the hash cannot be embedded here). Retrieve from the GPU host with "
        f"`scp <user>@<host>:<REMOTE_PROJECT_DIR>/{REL}/review_bundle.zip .`",
    ])


# =============================================================================== bundle
BUNDLE_FILES = ["reports/FINAL_REPORT.md", "reports/gates.json", "reports/run_index.csv",
                "reports/architecture_scores.csv", "reports/gate1_summary.csv", "reports/protocol_freeze.json",
                "reports/exploratory_diagnostics.json", "reports/narrative.json",
                "reports/environment.json", "reports/splits/qnli_split_manifest.json",
                "reports/splits/sst2_split_manifest.json", "configs/preregistered.yaml", "configs/baseline_fidelity.json",
                "docs/PREREGISTRATION.md", "docs/METHODS.md", "docs/PROTOCOL_CLARIFICATIONS.md", "pyproject.toml",
                "uv.lock", "README.md", "AGENTS.md", "reports/CORRECTNESS_AUDIT.md"]
SHARED_REPORT_FILES = ("reports/splits/", "reports/CORRECTNESS_AUDIT.md")
LOG = ROOT / "logs" / (f"pilot_{RUN_NAMESPACE}.log" if RUN_NAMESPACE else "pilot.log")


def _bundle_path(f: str) -> Path:
    """Namespaced report outputs live in reports/<ns>/; split manifests and the audit are shared."""
    if f.startswith("reports/") and not f.startswith(SHARED_REPORT_FILES):
        return REPORTS / f[len("reports/"):]
    return ROOT / f


def build_bundle(cfg) -> str:
    """Portable review archive: no weights, datasets, DB, credentials or per-example raw outputs."""
    zp = REPORTS / "review_bundle.zip"
    files = [_bundle_path(f) for f in BUNDLE_FILES if _bundle_path(f).exists()]
    files += sorted((ROOT / "reports" / "audit").glob("*.json")) if RUN_NAMESPACE else []
    files += sorted(FIG.glob("*.png"))
    for sub in ("gate0", "gate1", "gate2"):
        for p in sorted((ART / sub).glob("*.json")):
            files.append(p)
    files += [p for p in sorted((ART / "heads").glob("*.json"))]
    for pdir in sorted((ART / "probes").glob("*_s*")):
        files += [pdir / f for f in ("probe_result.json", "selection.json", "oplora.json") if (pdir / f).exists()]
    for gate in ("gate1", "gate2"):
        for p in sorted((ART / "runs" / gate).glob("*/result.json")):
            files.append(p)
    if (ART / "gate0" / "pytest_junit.xml").exists():
        files.append(ART / "gate0" / "pytest_junit.xml")
    src = sorted((ROOT / "src" / "rvm_lora").glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + \
        sorted((ROOT / "scripts").glob("*.sh"))
    files += src
    manifest = {}
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = str(f.relative_to(ROOT))
            z.write(f, rel)
            manifest[rel] = sha256_file(f)
        log = LOG
        if log.exists():
            lines = [ln for ln in log.read_text(errors="replace").splitlines()
                     if not any(s in ln for s in ("it/s]", "s/it]", "Loading weights", "Generating"))]
            z.writestr(f"logs/{log.stem}_compact.log", "\n".join(lines[-4000:]))
        z.writestr("BUNDLE_MANIFEST.json", json.dumps({"git_sha": git_sha(), "created": now(),
                                                       "files_sha256": manifest}, indent=2))
        z.writestr("REPRODUCE.md", _repro_section(cfg, _load(REPORTS / "protocol_freeze.json") or {}))
    sha = sha256_file(zp)
    (REPORTS / "review_bundle.sha256").write_text(f"{sha}  {REL}/review_bundle.zip\n")
    return sha


def _unused():
    return hashlib
