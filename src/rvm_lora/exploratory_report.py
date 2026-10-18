"""Report for the explicitly authorised exploratory Gate-2 execution (Gate-1 override).

Everything numeric is read from logged artifacts of this namespace; every gate quantity is the unchanged
`gates.gate2_criteria` output, labelled counterfactual. The preregistered decision is copied from the
reference namespace's `gates.json` and is never recomputed from exploratory results. Interpretation lives in
`narrative.json` (written after the results were seen) and is rendered in separately marked sections.
"""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from . import gates as G
from . import integrity
from .config import RUN_NAMESPACE, Config, sha256_file
from .pipeline import ART, REPORTS, ROOT, git_sha, head_meta_path, now, probe_dir
from .report import GRID, INK, INK2, LOG, SURF, _full, _gate_table, _load, _style, write_csv
from .tracking import mirror_dir, read_mirror, run_key, write_json

FIG = REPORTS / "figures"
REL = str(REPORTS.relative_to(ROOT))
ARTREL = str(ART.relative_to(ROOT))
LABEL = {"uniform_r1": "Uniform r1", "eva": "EVA", "oplora": "OPLoRA", "conditional_newton": "Cond. Newton",
         "group_rvm": "Group RVM", "uniform_r2": "Uniform r2 (B48)"}
# fixed entity -> categorical slot (reference palette, light mode; validated 6 slots, adjacent pairs)
COLOR = {"group_rvm": "#2a78d6", "conditional_newton": "#eb6834", "oplora": "#1baf7a", "eva": "#eda100",
         "uniform_r1": "#e87ba4", "uniform_r2": "#008300"}
MARKER = {"group_rvm": "o", "conditional_newton": "s", "oplora": "D", "eva": "^", "uniform_r1": "v", "uniform_r2": "P"}
SEQ = ["#f4f3f0", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
COMPETITORS = ["uniform_r1", "eva", "oplora", "conditional_newton", "uniform_r2"]
LN2 = float(np.log(2.0))


def _f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def _short(ranks: dict | None, modules: list[str]) -> str:
    if not ranks:
        return "∅ (no module)"
    if len(ranks) == len(modules) and len(set(ranks.values())) == 1:
        return f"all 24 modules × r{next(iter(ranks.values()))}"
    return ", ".join(f"L{k.split('.')[3]}{'Q' if k.endswith('query') else 'V'}:{v}"
                     for k, v in sorted(ranks.items(), key=lambda kv: modules.index(kv[0])) if v)


# =============================================================================== collection
def collect_runs(cfg: Config) -> list[dict]:
    sb = cfg["stage_b"]
    rows = []
    for t in sb["tasks"]:
        for s in sb["seeds"]:
            for m in sb["methods"]:
                key = run_key(cfg.sha256, "gate2", m, t, s, m)
                r = read_mirror(ROOT, "gate2", key) or {}
                mt = r.get("metrics", {})
                ranks = r.get("ranks")
                empty = str(r.get("note", "")).startswith("empty selection")
                rows.append({
                    "method": m, "task": t, "seed": s, "status": r.get("status", "NOT_STARTED"),
                    "official_validation_loss": mt.get("official_validation_loss"),
                    "official_validation_accuracy": mt.get("official_validation_accuracy"),
                    "adapter_params": mt.get("adapter_params"), "rank_units": mt.get("rank_sum"),
                    "modules_adapted": sum(1 for v in (ranks or {}).values() if v),
                    "budget_parameters": sb["comparison_parameters"] if m == "uniform_r2" else sb["budget_parameters"],
                    "selection_wall_s": mt.get("selection_wall_s"), "train_wall_s": mt.get("train_wall_s"),
                    "total_wall_s": mt.get("total_wall_s"), "peak_gpu_memory_mb": mt.get("peak_gpu_memory_mb"),
                    "train_loss_last20": mt.get("train_loss"), "train_loss_first20": r.get("train_loss_first"),
                    "microbatch": r.get("microbatch"), "accumulation": r.get("accumulation"),
                    "oom_events": len(r.get("oom_events") or []), "precision_train": r.get("precision_train"),
                    "initialisation": ("none (head-only model, empty selection)" if empty else
                                       "EVA native (data-driven A, B=0)" if m == "eva" else
                                       "shared zero-update (Kaiming-uniform A prefix, B=0)"),
                    "empty_selection": empty, "ranks_json": json.dumps(ranks, sort_keys=True) if ranks is not None else None,
                    "mlflow_run_id": r.get("mlflow_run_id"), "attempt": r.get("attempt"), "run_key": key,
                    "head_checkpoint_sha256": r.get("head_checkpoint_sha256"), "git_sha": r.get("git_sha"),
                    "run_namespace": r.get("run_namespace"), "eval_file_sha256": r.get("eval_file_sha256"),
                    "started": r.get("started"), "finished": r.get("finished"), "error": r.get("error"),
                    "mirror": str(mirror_dir(ROOT, "gate2", key).relative_to(ROOT) / "result.json") if r else None})
    return rows


def per_example(cfg: Config) -> dict:
    """Per-example validation losses; asserts that all methods are evaluated on identical validation IDs."""
    sb = cfg["stage_b"]
    out, ids = {}, {}
    for t in sb["tasks"]:
        for s in sb["seeds"]:
            for m in sb["methods"]:
                p = mirror_dir(ROOT, "gate2", run_key(cfg.sha256, "gate2", m, t, s, m)) / "eval_validation.npz"
                if not p.exists():
                    continue
                z = np.load(p)
                out[(m, t, s)] = z["loss"]
                if t in ids and not np.array_equal(ids[t], z["idx"]):
                    raise AssertionError(f"validation IDs differ for {m}/{t}/{s}")
                ids.setdefault(t, z["idx"])
    return out


def comparisons(cfg: Config, rows: list[dict], per: dict) -> list[dict]:
    """Descriptive RVM-vs-method contrasts (same bootstrap procedure as the gate; only the vs-best-B24 one is a criterion)."""
    sb = cfg["stage_b"]
    tasks, seeds = sb["tasks"], sb["seeds"]
    by = {(r["method"], r["task"], r["seed"]): r for r in rows}
    out = []
    for c in COMPETITORS:
        keys = [(t, s) for t in tasks for s in seeds]
        if not all(by[(m, t, s)]["status"] == "COMPLETED" for m in (c, "group_rvm") for t, s in keys):
            out.append({"competitor": c, "complete": False})
            continue
        d = {f"{t}/{s}": by[(c, t, s)]["official_validation_loss"] - by[("group_rvm", t, s)]["official_validation_loss"]
             for t, s in keys}
        da = {f"{t}/{s}": by[("group_rvm", t, s)]["official_validation_accuracy"] -
              by[(c, t, s)]["official_validation_accuracy"] for t, s in keys}
        boot = G.paired_bootstrap({k: per[("group_rvm",) + k] for k in keys}, {k: per[(c,) + k] for k in keys},
                                  tasks, seeds, sb["bootstrap_resamples"], sb["bootstrap_seed"],
                                  sb["bootstrap_lower_percentile"])
        out.append({"competitor": c, "complete": True, "budget": "B48" if c == "uniform_r2" else "B24",
                    "L_competitor": float(np.mean([by[(c, t, s)]["official_validation_loss"] for t, s in keys])),
                    "L_rvm": float(np.mean([by[("group_rvm", t, s)]["official_validation_loss"] for t, s in keys])),
                    "loss_advantage_rvm": float(np.mean(list(d.values()))), "per_task_seed_loss_advantage": d,
                    "per_task_loss_advantage": {t: float(np.mean([d[f"{t}/{s}"] for s in seeds])) for t in tasks},
                    "rvm_wins": int(sum(v > 0 for v in d.values())),
                    "accuracy_advantage_rvm": float(np.mean(list(da.values()))), "per_task_seed_accuracy_advantage": da,
                    "bootstrap_90pct": boot, "role": "descriptive"})
    return out


def allocations(cfg: Config, rows: list[dict]) -> dict:
    sb = cfg["stage_b"]
    out = {}
    for t in sb["tasks"]:
        for s in sb["seeds"]:
            sel = _load(probe_dir(t, s) / "selection.json") or {}
            op = _load(probe_dir(t, s) / "oplora.json") or {}
            eva = [r for r in rows if r["method"] == "eva" and r["task"] == t and r["seed"] == s][0]
            out[f"{t}/{s}"] = {"group_rvm": sel.get("group_rvm", {}).get("ranks"),
                               "conditional_newton": sel.get("conditional_newton", {}).get("ranks"),
                               "oplora": op.get("ranks"),
                               "eva": json.loads(eva["ranks_json"]) if eva["ranks_json"] else None,
                               "rvm_no_logdet (offline ablation, not trained)": sel.get("rvm_no_logdet", {}).get("ranks"),
                               "newton_own_null_threshold (diagnostic, not trained)":
                                   sel.get("conditional_newton", {}).get("diagnostic_thresholded_ranks")}
    return out


def aux_runs(cfg: Config) -> list[dict]:
    sb = cfg["stage_b"]
    out = []
    g0 = _load(ART / "gate0" / "gate0.json") or {}
    out.append({"stage": "gate0", "kind": "test_suite", "method": "gate0_suite", "task": "none", "seed": 0,
                "status": g0.get("status"), "mlflow_run_id": g0.get("mlflow_run_id"), "wall_s": g0.get("wall_s"),
                "detail": f"{g0.get('n_passed')}/{g0.get('n_tests')} tests, git {g0.get('git_sha')}"})
    for t in sb["tasks"]:
        for s in sb["seeds"]:
            h = _load(head_meta_path(t, s)) or {}
            out.append({"stage": "gate2", "kind": "head_warm_start", "method": "head_warm_start", "task": t, "seed": s,
                        "status": "COMPLETED" if h else "NOT_STARTED", "mlflow_run_id": h.get("mlflow_run_id"),
                        "wall_s": h.get("total_wall_s"),
                        "detail": f"head sha256 {h.get('head_checkpoint_sha256')}; train loss first20 "
                                  f"{h.get('head_train_loss_first20')} last20 {h.get('head_train_loss_last20')}"})
            p = _load(probe_dir(t, s) / "probe_result.json") or {}
            tm = p.get("timings", {})
            out.append({"stage": "gate2", "kind": "probe_selection", "method": "probe_selection", "task": t, "seed": s,
                        "status": "COMPLETED" if p else "NOT_STARTED", "mlflow_run_id": p.get("mlflow_run_id"),
                        "wall_s": (tm.get("direction_pass_s", 0) + tm.get("feature_pass_s", 0) +
                                   tm.get("oplora_importance_pass_s", 0) + tm.get("selection_job_wall_s", 0)) if p else None,
                        "detail": f"selection.json sha256 {p.get('selection_file_sha256')}"})
    return out


COMPUTE_CODE = ["selectors.py", "evidence.py", "probe.py", "selection_job.py", "train.py", "adapters.py",
                "baselines.py", "modeling.py", "data.py", "config.py", "pipeline.py", "gates.py", "tracking.py"]


def code_identity(base: str | None, shas: list[str]) -> dict:
    """Which computation modules are byte-identical between the reference freeze commit and the training commit."""
    if not base or len(shas) != 1:
        return {"base": base, "training": shas, "files": {}}
    files = {}
    for f in COMPUTE_CODE:
        rc = subprocess.run(["git", "diff", "--quiet", base, shas[0], "--", f"src/rvm_lora/{f}"], cwd=ROOT).returncode
        files[f] = {0: "identical", 1: "changed"}.get(rc, "unknown")
    stat = subprocess.run(["git", "diff", "--stat", base, shas[0], "--", "src/rvm_lora"], cwd=ROOT, text=True,
                          capture_output=True).stdout.strip()
    return {"base": base, "training": shas[0], "files": files, "diff_stat": stat}


def split_identity(cfg: Config) -> dict:
    """Split IDs regenerated from the pinned GLUE train split and the frozen split seed must equal the shared
    manifests, the encoded data files used here, and the IDs recorded by this namespace's heads and probes."""
    from . import data as D
    from .pipeline import data_file

    out = {}
    for t in cfg.tasks:
        man = D.load_manifest(ROOT, t)
        idx = np.asarray(D.load_glue(t, "train")["idx"], dtype=np.int64)
        regen = D.make_split(idx, cfg)
        chk = {"regenerated_equals_manifest": {k: regen[k] == man["ids"][k] for k in man["ids"]}}
        chk["encoded_files_equal_manifest"] = {
            p: D.ids_sha256(list(D.load_encoded(data_file(t, p)).idx)) == man["sha256"][p]
            for p in ("head", "directions", "evidence", "adapter_train", "audit")}
        val = D.load_encoded(data_file(t, "validation"))
        chk["validation_rows"] = len(val)
        chk["validation_ids_sha256"] = D.ids_sha256(list(val.idx))
        for s in cfg["stage_b"]["seeds"]:
            h = _load(head_meta_path(t, s)) or {}
            p = (_load(probe_dir(t, s) / "probe_result.json") or {}).get("hashes", {})
            chk[f"seed{s}_head_ids_equal_manifest"] = h.get("head_ids_sha256") == man["sha256"]["head"]
            chk[f"seed{s}_probe_direction_ids_equal_manifest"] = p.get("directions_ids_sha256") == man["sha256"]["directions"]
            chk[f"seed{s}_probe_evidence_ids_equal_manifest"] = p.get("evidence_ids_sha256") == man["sha256"]["evidence"]
        flat = [v for k, v in chk.items() if isinstance(v, bool)] + [v for k in ("regenerated_equals_manifest",
                                                                                  "encoded_files_equal_manifest")
                                                                     for v in chk[k].values()]
        out[t] = {**chk, "all_equal": all(flat), "manifest_file_sha256": sha256_file(D.manifest_path(ROOT, t))}
    return out


def log_span() -> dict:
    """First start and last completion of the run script, from the namespace log."""
    starts, ends = [], []
    if LOG.exists():
        for ln in LOG.read_text(errors="replace").splitlines():
            if ln.startswith("[run_gate2_exploratory] start "):
                starts.append(ln.split()[2])
            elif ln.startswith("[run_gate2_exploratory] experiments done "):
                ends.append(ln.split()[-1])
    return {"first_start": starts[0] if starts else None, "last_done": ends[-1] if ends else None,
            "script_starts": len(starts)}


# =============================================================================== figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_quality(cfg, rows) -> str:
    plt = _plt()
    sb = cfg["stage_b"]
    methods = sb["methods"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=SURF)
    for j, t in enumerate(sb["tasks"]):
        for i, (key, ylab) in enumerate((("official_validation_loss", "official validation CE (nats)"),
                                         ("official_validation_accuracy", "official validation accuracy"))):
            ax = axes[i, j]
            _style(ax)
            ax.grid(False, axis="x")
            for k, m in enumerate(methods):
                rr = [r for r in rows if r["method"] == m and r["task"] == t and r["status"] == "COMPLETED"]
                for r in rr:
                    dx = -0.12 if r["seed"] == sb["seeds"][0] else 0.12
                    ax.scatter(k + dx, r[key], s=56, color=COLOR[m], marker="o" if dx < 0 else "s",
                               edgecolor=SURF, linewidth=2, zorder=3)
                if rr:
                    mu = float(np.mean([r[key] for r in rr]))
                    ax.plot([k - 0.28, k + 0.28], [mu, mu], color=INK, linewidth=2, zorder=4, solid_capstyle="round")
            if key == "official_validation_loss":
                ax.axhline(LN2, color="#9a9994", linewidth=1, linestyle="--", zorder=1)
                ax.annotate("ln 2 (uniform predictor)", (len(methods) - 0.5, LN2), fontsize=7, color=INK2,
                            ha="right", va="bottom")
            else:
                ax.axhline(0.5, color="#9a9994", linewidth=1, linestyle="--", zorder=1)
            ax.set_xticks(range(len(methods)), [LABEL[m] for m in methods], fontsize=8, rotation=15)
            ax.set_ylabel(ylab, fontsize=8, color=INK2)
            ax.set_title(f"{t.upper()}: {'cross-entropy (lower is better)' if i == 0 else 'accuracy'}",
                         fontsize=9, color=INK)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", color=INK2, label=f"seed {sb['seeds'][0]}"),
               Line2D([], [], marker="s", linestyle="", color=INK2, label=f"seed {sb['seeds'][1]}"),
               Line2D([], [], color=INK, linewidth=2, label="two-seed mean")]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=3)
    fig.suptitle("Exploratory Gate 2: official-validation quality per method (each point = one 600-step run)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG / "gate2_quality_by_method.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(ROOT))


def fig_params(cfg, rows) -> str:
    plt = _plt()
    sb = cfg["stage_b"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURF)
    for ax, t in zip(axes, sb["tasks"]):
        _style(ax)
        for m in sb["methods"]:
            rr = [r for r in rows if r["method"] == m and r["task"] == t and r["status"] == "COMPLETED"]
            if not rr:
                continue
            ax.scatter([r["adapter_params"] for r in rr], [r["official_validation_loss"] for r in rr], s=64,
                       color=COLOR[m], marker=MARKER[m], edgecolor=SURF, linewidth=1, zorder=3, label=LABEL[m])
        ax.legend(fontsize=7.5, frameon=False, loc="best")
        ax.set_xticks([0, sb["budget_parameters"], sb["comparison_parameters"]],
                      ["0", f"{sb['budget_parameters']:,}\n(B24)", f"{sb['comparison_parameters']:,}\n(B48)"])
        ax.set_xlim(-4000, sb["comparison_parameters"] + 12000)
        ax.set_xlabel("exact trainable LoRA parameters", fontsize=8, color=INK2)
        ax.set_ylabel("official validation CE (nats)", fontsize=8, color=INK2)
        ax.set_title(f"{t.upper()}: CE vs exact adapter parameters (both seeds)", fontsize=9, color=INK)
    fig.tight_layout()
    p = FIG / "gate2_quality_vs_params.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(ROOT))


def fig_alloc(cfg, alloc) -> str:
    plt = _plt()
    from matplotlib.colors import ListedColormap

    names = [("group_rvm", "Group RVM"), ("conditional_newton", "Cond. Newton"), ("oplora", "OPLoRA"),
             ("eva", "EVA (native)"), ("rvm_no_logdet (offline ablation, not trained)", "no-logdet (ablation)")]
    fig, axes = plt.subplots(len(alloc), 1, figsize=(12, 2.5 * len(alloc) + 0.6), facecolor=SURF, squeeze=False,
                             gridspec_kw={"hspace": 0.75})
    im = None
    for ax, (cfgkey, a) in zip(axes[:, 0], alloc.items()):
        M = np.array([[(a.get(k) or {}).get(m, 0) for m in cfg.modules] for k, _ in names])
        im = ax.imshow(M, cmap=ListedColormap(SEQ), vmin=0, vmax=8, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] >= 4 else INK)
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=SURF, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_yticks(range(len(names)), [n for _, n in names], fontsize=8)
        ax.set_xticks(range(24), [f"L{m.split('.')[3]}{'Q' if m.endswith('query') else 'V'}" for m in cfg.modules],
                      fontsize=7)
        units = " · ".join(f"{n.split(' (')[0]} {sum((a.get(k) or {}).values())}" for k, n in names)
        ax.set_title(f"{cfgkey} · rank units: {units}", fontsize=9, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.5, fraction=0.025, pad=0.02, label="rank")
    p = FIG / "allocations.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return str(p.relative_to(ROOT))


def fig_diff(comps) -> str:
    plt = _plt()
    cc = [c for c in comps if c.get("complete")]
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURF)
    _style(ax)
    ax.grid(False, axis="y")
    for i, c in enumerate(cc):
        y = len(cc) - 1 - i
        b = c["bootstrap_90pct"]
        ax.plot([b["lower"], b["upper"]], [y, y], color="#2a78d6", linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([c["loss_advantage_rvm"]], [y], s=70, color="#2a78d6", edgecolor=SURF, linewidth=2, zorder=3)
        v = list(c["per_task_seed_loss_advantage"].values())
        ax.scatter(v, [y - 0.22] * len(v), s=22, facecolor="none", edgecolor=INK2, linewidth=1, zorder=3)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(range(len(cc)), [f"vs {LABEL[c['competitor']]}" for c in reversed(cc)], fontsize=8)
    ax.set_xlabel("L(competitor) − L(Group RVM), official validation CE in nats  (> 0: RVM better)", fontsize=8,
                  color=INK2)
    ax.set_title("Group RVM vs each method: 4-run mean (dot), descriptive 90% paired bootstrap (line), "
                 "task/seed differences (open)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    p = FIG / "rvm_paired_differences.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(ROOT))


def fig_walltime(cfg, rows, ratio_cut) -> str:
    plt = _plt()
    methods = cfg["stage_b"]["methods"]
    ok = [r for r in rows if r["status"] == "COMPLETED"]
    sel = [float(np.mean([r["selection_wall_s"] or 0 for r in ok if r["method"] == m])) for m in methods]
    trn = [float(np.mean([r["train_wall_s"] or 0 for r in ok if r["method"] == m])) for m in methods]
    u1 = float(np.median([r["train_wall_s"] for r in ok if r["method"] == "uniform_r1"]))
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURF)
    _style(ax)
    ax.grid(False, axis="x")
    xs = np.arange(len(methods))
    ax.bar(xs - 0.2, sel, width=0.38, color="#2a78d6", label="selection / calibration (s)", edgecolor=SURF, linewidth=2)
    ax.bar(xs + 0.2, trn, width=0.38, color="#eb6834", label="600-step adapter training (s)", edgecolor=SURF,
           linewidth=2)
    ax.axhline(ratio_cut * u1, color=INK, linewidth=1, linestyle="--", label=f"overhead budget: {ratio_cut:.0%} of "
               f"median uniform-r1 training ({ratio_cut * u1:.1f} s)")
    for x, v in zip(xs, sel):
        ax.annotate(f"{v:.1f}", (x - 0.2, v), fontsize=7, color=INK2, ha="center", va="bottom")
    for x, v in zip(xs, trn):
        ax.annotate(f"{v:.1f}", (x + 0.2, v), fontsize=7, color=INK2, ha="center", va="bottom")
    ax.set_ylim(0, 1.12 * max(sel + trn))
    ax.set_xticks(xs, [LABEL[m] for m in methods], fontsize=8)
    ax.set_ylabel("mean wall time per run (s)", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("Exploratory Gate 2: measured selection and training wall time (mean of 4 task/seed runs)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    p = FIG / "wall_time.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(ROOT))


# =============================================================================== MLflow export
def mlflow_export(rows: list[dict], aux: list[dict]) -> dict:
    from mlflow.tracking import MlflowClient

    c = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    ids = [r["mlflow_run_id"] for r in rows + aux if r.get("mlflow_run_id")]
    out = {}
    for rid in ids:
        r = c.get_run(rid)
        hist = {k: [(m.step, m.value) for m in c.get_metric_history(rid, k)] for k in r.data.metrics
                if k == "train_loss"}
        out[rid] = {"experiment_id": r.info.experiment_id,
                    "experiment": c.get_experiment(r.info.experiment_id).name, "run_name": r.info.run_name,
                    "status": r.info.status, "start_time": r.info.start_time, "end_time": r.info.end_time,
                    "tags": dict(sorted(r.data.tags.items())), "params": dict(sorted(r.data.params.items())),
                    "metrics": dict(sorted(r.data.metrics.items())), "metric_history": hist,
                    "artifacts": [a.path for a in c.list_artifacts(rid)]}
    return out


# =============================================================================== build
def build_exploratory_report(cfg: Config, override: dict) -> dict:
    if not RUN_NAMESPACE:
        raise RuntimeError("the exploratory report is built only inside its own run namespace")
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sb = cfg["stage_b"]
    ref = ROOT / override["reference_artifacts"]
    ref_rep = ROOT / "reports" / ref.relative_to(ROOT / "artifacts")
    g2doc = _load(ART / "gate2" / "gate2_exploratory.json")
    if g2doc is None:
        raise RuntimeError("run `eval-gate2 --exploratory-override-gate1 ...` first")
    g2x = g2doc["gate2"]
    pre = _load(ref_rep / "gates.json")
    rows = collect_runs(cfg)
    per = per_example(cfg)
    comps = comparisons(cfg, rows, per)
    alloc = allocations(cfg, rows)
    aux = aux_runs(cfg)
    write_csv(REPORTS / "results.csv", rows)
    write_csv(REPORTS / "run_index.csv",
              [{**a, "rank_allocation": None, "official_validation_loss": None} for a in aux] +
              [{"stage": "gate2", "kind": "adapter_run" if not r["empty_selection"] else "head_only_eval",
                "method": r["method"], "task": r["task"], "seed": r["seed"], "status": r["status"],
                "mlflow_run_id": r["mlflow_run_id"], "wall_s": r["total_wall_s"],
                "detail": f"attempt {r['attempt']}; params {r['adapter_params']}; mirror {r['mirror']}",
                "rank_allocation": r["ranks_json"], "official_validation_loss": r["official_validation_loss"]}
               for r in rows])
    # integrity of the reference (v2) run: compare against the snapshot taken before the exploratory run started
    before = _load(ART / "integrity" / "v2_before.json")
    after = integrity.snapshot(before["namespace"], os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    write_json(ART / "integrity" / "v2_after.json", after)
    integ = integrity.compare(before, after)
    write_json(ART / "integrity" / "v2_compare.json", integ)
    refchk = _load(ART / "gate2" / "reference_check.json") or {}
    freeze = _load(REPORTS / "protocol_freeze.json") or {}
    env = _load(REPORTS / "environment.json") or {}
    g0 = _load(ART / "gate0" / "gate0.json") or {}
    shas = sorted({r["git_sha"] for r in rows if r["git_sha"]})
    ref_freeze = _load(ref_rep / "protocol_freeze.json") or {}
    ident = code_identity(ref_freeze.get("git_sha"), shas)
    crit = {c["criterion"]: c for c in g2x["criteria"]}
    ov = crit.get("mandatory_overhead_ratio", {})
    standalone = {
        "label": g2x["label"], "generated_at": now(), "report_git_sha": git_sha(),
        "preregistered_overall_decision": pre["overall_decision"],
        "preregistered_gate_statuses": {g: pre[g]["status"] for g in ("gate0", "gate1", "gate2")},
        "preregistered_decision_source": str((ref_rep / "gates.json").relative_to(ROOT)),
        "preregistered_decision_recomputed_from_statuses": G.overall_decision(
            pre["gate0"]["status"], pre["gate1"]["status"], pre["gate2"]["status"]),
        "exploratory_gate2_standalone_result": g2x["exploratory_gate2_standalone_result"],
        "execution_type": g2x["execution_type"], "gate1_status": g2x["gate1_status"],
        "preregistered_gate2_status": g2x["preregistered_gate2_status"], "override": g2x.get("override"),
        "route_a": g2x.get("route_a"), "route_b": g2x.get("route_b"), "best_b24_baseline": g2x.get("best_b24_baseline"),
        "L_mean_loss_per_method": g2x.get("L"), "criteria": g2x["criteria"], "bootstrap_vs_best_b24": g2x.get("bootstrap"),
        "thresholds_source": "configs/preregistered.yaml (stage_b), unchanged", "config_sha256": cfg.sha256,
        "descriptive_comparisons": comps, "allocations": alloc,
        "overhead": {"ratio": ov.get("value"), "cutoff": ov.get("cutoff"), "status": ov.get("status"),
                     "mean_rvm_selection_s": ov.get("mean_rvm_selection_s"),
                     "median_uniform_r1_train_s": ov.get("median_uniform_r1_train_s")},
        "training_code_git_shas": shas, "code_identity_vs_reference_freeze": ident, "log_span": log_span(),
        "split_identity": split_identity(cfg),
        "reference_check_all_equal": [c.get("all_equal") for c in refchk.get("checks", [])],
        "reference_integrity": integ, "gate0_in_namespace": {"status": g0.get("status"), "n_passed": g0.get("n_passed"),
                                                            "n_tests": g0.get("n_tests"), "git_sha": g0.get("git_sha")},
        "split_manifests_equal_to_original": freeze.get("original_run", {}).get("split_manifests_equal"),
        "uv_lock_equal_to_original": freeze.get("original_run", {}).get("uv_lock_equal"),
        "environment": {k: env.get(k) for k in ("nvidia_smi", "gpu_name", "torch_cuda", "python", "packages",
                                                "uv_lock_sha256", "bf16_supported")},
        "mlflow": {"tracking_uri": "http://127.0.0.1:5000 (GPU host, loopback only)", "experiment": "rvm-lora/gate2",
                   "run_ids": {f"{r['method']}|{r['task']}|{r['seed']}": r["mlflow_run_id"] for r in rows},
                   "export": f"{REL}/mlflow_export.json"},
    }
    write_json(REPORTS / "standalone_gate2.json", standalone)
    write_json(REPORTS / "mlflow_export.json", mlflow_export(rows, aux))
    figs = [fig_quality(cfg, rows), fig_params(cfg, rows), fig_diff(comps), fig_alloc(cfg, alloc),
            fig_walltime(cfg, rows, sb["mandatory"]["max_calibration_to_uniform_train_walltime"])]
    md = render(cfg, standalone, g2x, rows, comps, alloc, aux, refchk, integ, freeze, env, g0, ref, figs)
    (REPORTS / "FINAL_REPORT.md").write_text(md)
    sha = build_bundle(cfg, ref, ref_rep)
    return {"figures": figs, "bundle_sha256": sha, "standalone": standalone["exploratory_gate2_standalone_result"]}


# =============================================================================== markdown
def render(cfg, st, g2x, rows, comps, alloc, aux, refchk, integ, freeze, env, g0, ref, figs) -> str:
    sb = cfg["stage_b"]
    tasks, seeds, methods = sb["tasks"], sb["seeds"], sb["methods"]
    nar = _load(REPORTS / "narrative.json") or {}
    by = {(r["method"], r["task"], r["seed"]): r for r in rows}
    ok = [r for r in rows if r["status"] == "COMPLETED"]
    L = []
    A = L.append

    def nv(k):
        return nar.get(k, "_(interpretation not yet written)_")

    A("# RVM-LoRA pilot: exploratory Gate 2 under an explicit Gate-1 override\n")
    A(f"**Date:** {now()} (UTC) · **Namespace:** `{RUN_NAMESPACE}` · **Execution type:** "
      f"`{st['execution_type']}`  ")
    A(f"**Report build git SHA:** `{git_sha()}` · **Training/selection code git SHA(s):** "
      + ", ".join(f"`{s}`" for s in st["training_code_git_shas"]) + "  ")
    A(f"**Protocol YAML SHA256:** `{cfg.sha256}` (unchanged; thresholds from `configs/preregistered.yaml`)\n")
    A("| Decision quantity | Value | Source |\n|---|---|---|")
    A(f"| `preregistered_overall_decision` | **{st['preregistered_overall_decision']}** | "
      f"`{st['preregistered_decision_source']}` (copied, not recomputed from exploratory results) |")
    A(f"| Preregistered Gate 0 / Gate 1 / Gate 2 | {st['preregistered_gate_statuses']['gate0']} / "
      f"{st['preregistered_gate_statuses']['gate1']} / {st['preregistered_gate_statuses']['gate2']} | same file |")
    A(f"| `exploratory_gate2_standalone_result` (counterfactual) | **{st['exploratory_gate2_standalone_result']}** | "
      f"`{REL}/standalone_gate2.json`, `{ARTREL}/gate2/gate2_exploratory.json` |")
    A(f"| Counterfactual Route A / Route B | {st['route_a']} / {st['route_b']} | same |\n")
    A("> **Scope.** The user explicitly authorised running the frozen Gate-2 experiment despite the failed Gate 1, "
      "for this exploratory execution only. The 24 configurations, thresholds and code paths are the preregistered "
      "ones. The Gate-2 \"result\" below is a counterfactual: it says what the frozen Gate-2 criteria give when "
      "evaluated on their own. It does not count toward the preregistered Go/No-Go decision and cannot change it. "
      "Gate 1 remains FAIL and the project decision remains NO_GO.\n")
    A("## Executive summary (interpretation)\n")
    A(nv("executive") + "\n")

    # ------------------------------------------------------------------ provenance
    A("## Provenance, isolation and reproducibility\n")
    pk = env.get("packages", {})
    A("| Item | Value | Verified |\n|---|---|---|")
    A(f"| Hardware | {env.get('nvidia_smi')} | `nvidia-smi` |")
    A(f"| Software | Python {env.get('python')}, torch {pk.get('torch')} (CUDA {env.get('torch_cuda')}), "
      f"transformers {pk.get('transformers')}, PEFT {pk.get('peft')}, MLflow {pk.get('mlflow')}, "
      f"numpy {pk.get('numpy')}, scipy {pk.get('scipy')}; `uv sync --frozen`, uv.lock `{env.get('uv_lock_sha256')}` "
      f"(equal to original freeze: {st['uv_lock_equal_to_original']}) | `{REL}/environment.json` |")
    A(f"| Model / dataset revisions | `{cfg['backbone']}@{freeze.get('model', {}).get('revision')}`, "
      f"`{cfg['dataset_repository']}@{freeze.get('dataset', {}).get('revision')}` | pinned in code, logged as MLflow tags |")
    ci = st["code_identity_vs_reference_freeze"]
    same = [f for f, v in ci["files"].items() if v == "identical"]
    diff = [f for f, v in ci["files"].items() if v != "identical"]
    A(f"| Corrected implementation | training commit contains `4c54b1174b7501f7843407eaf042e8398ead6c2b` "
      f"(`git merge-base --is-ancestor`, checked by `scripts/run_gate2_exploratory.sh` before any step). Versus the "
      f"v2 freeze commit `{(ci['base'] or '')[:7]}`: identical: {', '.join(f'`{f}`' for f in same) or '—'}; changed: "
      f"{', '.join(f'`{f}`' for f in diff) or 'none'} (override, MLflow tags and Gate-2 refactor only; see "
      f"`git diff {(ci['base'] or '')[:7]} {(ci['training'] or '')[:7]}`) | `git diff --quiet` per file |")
    A(f"| Protocol files | YAML, PREREGISTRATION, METHODS, PROTOCOL_CLARIFICATIONS hashes equal to the original "
      f"freeze (else `freeze` refuses) | `{REL}/protocol_freeze.json` |")
    si = st["split_identity"]
    A(f"| Split identities | manifests reused from `reports/splits/` (equal to the original freeze: "
      f"{st['split_manifests_equal_to_original']}). Re-generated from the pinned GLUE train IDs with split seed "
      f"{cfg['split_seed']} and compared with the manifests, the encoded head/directions/evidence/adapter_train/audit "
      f"files, and the IDs recorded by every head and probe here: "
      + "; ".join(f"{t}: all equal = **{v['all_equal']}**, validation rows {v['validation_rows']}" for t, v in si.items())
      + ". Validation IDs are identical across all 24 runs per task (asserted). | `standalone_gate2.json` → "
      "`split_identity` |")
    A(f"| Gate-0 suite on this commit | {g0.get('status')} ({g0.get('n_passed')}/{g0.get('n_tests')} tests, incl. "
      f"CUDA smoke and the new override regression tests), MLflow `{g0.get('mlflow_run_id')}` | "
      f"`{ARTREL}/gate0/gate0.json` |")
    A(f"| Gate-1 record carried by the override | `{st['override']['gate1_record']}` status "
      f"**{st['gate1_status']}**, SHA256 `{st['override']['gate1_record_sha256']}` (read-only) | MLflow tags |")
    A(f"| v2 run untouched | {integ['files_checked']} files under `artifacts/v2`, `checkpoints/v2`, `reports/v2`: "
      f"{len(integ['files_changed_or_missing'])} changed/missing, {len(integ['files_added'])} added; "
      f"{integ['mlflow_runs_checked']} pre-existing MLflow runs: {len(integ['mlflow_runs_changed_or_missing'])} "
      f"changed; {integ['mlflow_runs_added']} runs added (this execution) → untouched: **{integ['untouched']}** | "
      f"`{ARTREL}/integrity/v2_compare.json` |")
    A(f"| MLflow | experiment `rvm-lora/gate2`; every run tagged `stage=gate2`, "
      f"`execution_type=exploratory_override`, `gate1_status={st['gate1_status']}`, "
      f"`protocol_version={st['override']['protocol_version']}`, `run_namespace={RUN_NAMESPACE}`, `git_sha`, "
      f"`head_checkpoint_sha256`, `model_revision`, `dataset_revision`, `rank_allocation`; offline copy "
      f"`{REL}/mlflow_export.json` | tags |\n")
    A("**QNLI/42 reproduction of the corrected v2 head and probes.** The exploratory namespace rebuilt the QNLI/42 "
      "classifier head and probes from scratch. Nothing was copied from v2. Every item below must equal the v2 "
      "value, otherwise `train-b` stops before any training:\n")
    A("| Item | v2 reference | exploratory | Equal |\n|---|---|---|---|")
    for c in refchk.get("checks", []):
        if not c.get("reference"):
            continue
        for k, v in c["checks"].items():
            fmt = (lambda x: f"`{json.dumps(x, sort_keys=True)[:90]}`" if isinstance(x, (dict, list)) else f"`{x}`")
            A(f"| {c['task']}/{c['seed']} {k} | {fmt(v['reference'])} | {fmt(v['exploratory'])} | {v['equal']} |")
    others = [f"{c['task']}/{c['seed']}" for c in refchk.get("checks", []) if not c.get("reference")]
    A(f"\nThe other configurations ({', '.join(others)}) have no v2 counterpart: Gate 2 never ran in v2. Their heads "
      "and probes were created here for the first time with the same code.\n")

    # ------------------------------------------------------------------ A
    A("## A. Main results (observed)\n")
    A("Official GLUE validation (QNLI 5,463 rows, SST-2 872 rows); final 600-step checkpoint; binary "
      "cross-entropy in nats per example (float64). Seed-level values appear in brackets as `[seed 42; seed 2025]`.\n")
    A("| Method | Budget | QNLI CE mean [42; 2025] | QNLI acc mean [42; 2025] | SST-2 CE mean [42; 2025] | "
      "SST-2 acc mean [42; 2025] | 4-run mean CE | 4-run mean acc |")
    A("|---|---|---|---|---|---|---:|---:|")
    for m in methods:
        cells = []
        for t in tasks:
            for key, nd in (("official_validation_loss", 5), ("official_validation_accuracy", 4)):
                v = [by[(m, t, s)][key] for s in seeds]
                cells.append("—" if None in v else f"{np.mean(v):.{nd}f} [{v[0]:.{nd}f}; {v[1]:.{nd}f}]")
        all_l = [by[(m, t, s)]["official_validation_loss"] for t in tasks for s in seeds]
        all_a = [by[(m, t, s)]["official_validation_accuracy"] for t in tasks for s in seeds]
        A(f"| {LABEL[m]} | {'B48' if m == 'uniform_r2' else 'B24'} | " + " | ".join(cells) +
          f" | {_f(np.mean(all_l) if None not in all_l else None, 5)} | "
          f"{_f(np.mean(all_a) if None not in all_a else None, 4)} |")
    rank = sorted([m for m in methods if (st["L_mean_loss_per_method"] or {}).get(m) is not None],
                  key=lambda m: st["L_mean_loss_per_method"][m])
    A("\n**Ranking by 4-run mean CE (lower is better):** " +
      " < ".join(f"{LABEL[m]} ({st['L_mean_loss_per_method'][m]:.5f})" for m in rank) + "\n")
    A("**Per-run measurements** (full-precision CE in `results.csv`; selection = method-specific calibration, "
      "PROTOCOL_CLARIFICATIONS C5):\n")
    A("| Method | Task | Seed | CE | Acc. | Exact params | Rank units | Modules | Selection s | Train s | "
      "Total s | Peak GPU MB | Microbatch | MLflow run |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for m in methods:
        for t in tasks:
            for s in seeds:
                r = by[(m, t, s)]
                A(f"| {LABEL[m]} | {t} | {s} | {_f(r['official_validation_loss'], 5)} | "
                  f"{_f(r['official_validation_accuracy'], 4)} | {r['adapter_params'] if r['adapter_params'] is not None else '—'} | "
                  f"{r['rank_units'] if r['rank_units'] is not None else '—'} | {r['modules_adapted']} | "
                  f"{_f(r['selection_wall_s'], 2)} | {_f(r['train_wall_s'], 1)} | {_f(r['total_wall_s'], 1)} | "
                  f"{_f(r['peak_gpu_memory_mb'], 0)} | {r['microbatch'] or '—'}×{r['accumulation'] or '—'} | "
                  f"`{r['mlflow_run_id']}` |")
    A("\n**Rank allocations actually trained** (exact JSON in `results.csv` column `ranks_json` and MLflow tag "
      "`rank_allocation`):\n")
    A("| Method | Task/seed | Rank units | Allocation |\n|---|---|---:|---|")
    for m in methods:
        for t in tasks:
            for s in seeds:
                r = by[(m, t, s)]
                A(f"| {LABEL[m]} | {t}/{s} | {r['rank_units']} | "
                  f"{_short(json.loads(r['ranks_json']) if r['ranks_json'] else None, cfg.modules)} |")
    for f in figs[:2]:
        A(f"\n![{Path(f).stem}]({Path(f).relative_to(REL)})")
    A("\n" + nv("a_main") + "\n")

    # ------------------------------------------------------------------ B
    A("## B. Matched-budget comparisons (observed)\n")
    A("Each row compares Group RVM (B24, ≤36,864 parameters) with one method under the same budget, head, data "
      "order, seeds and evaluation. The difference is `L(method) − L(RVM)`, so positive values mean RVM has lower "
      "CE. The 90% interval uses the frozen Gate-2 procedure: stratified paired-example bootstrap, 2,000 "
      "resamples, seed 991, the same indices for both seeds within a task. For the post-hoc strongest B24 "
      "competitor it is a Gate-2 criterion. For every other row it is **descriptive only**.\n")
    A("| RVM vs | Budget | L(method) | L(RVM) | L(method) − L(RVM) | QNLI mean diff | SST-2 mean diff | RVM wins "
      "(of 4) | 90% paired bootstrap [5%, 95%] | Acc(RVM) − Acc(method) |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for c in comps:
        if not c.get("complete"):
            A(f"| {LABEL[c['competitor']]} | — | incomplete | | | | | | | |")
            continue
        b = c["bootstrap_90pct"]
        tag = " **(gate criterion: strongest B24)**" if c["competitor"] == st["best_b24_baseline"] else ""
        A(f"| {LABEL[c['competitor']]}{tag} | {c['budget']} | {c['L_competitor']:.5f} | {c['L_rvm']:.5f} | "
          f"{c['loss_advantage_rvm']:+.5f} | {c['per_task_loss_advantage']['qnli']:+.5f} | "
          f"{c['per_task_loss_advantage']['sst2']:+.5f} | {c['rvm_wins']} | [{b['lower']:+.5f}, {b['upper']:+.5f}] | "
          f"{c['accuracy_advantage_rvm']:+.4f} |")
    A("\nPer task/seed CE differences `L(method) − L(RVM)` (full precision in `standalone_gate2.json`):\n")
    A("| RVM vs | " + " | ".join(f"{t}/{s}" for t in tasks for s in seeds) + " |")
    A("|---|" + "---:|" * (len(tasks) * len(seeds)))
    for c in comps:
        if c.get("complete"):
            A(f"| {LABEL[c['competitor']]} | " + " | ".join(f"{v:+.5f}" for v in c["per_task_seed_loss_advantage"].values())
              + " |")
    A(f"\n![{Path(figs[2]).stem}]({Path(figs[2]).relative_to(REL)})\n")
    A(nv("b_comparisons") + "\n")

    # ------------------------------------------------------------------ C
    A("## C. Structural analysis (observed)\n")
    A("Layer-by-layer ranks as `query/value` per layer. RVM, Newton and OPLoRA allocations come from the probe "
      "stage before training. EVA's ranks are its native explained-variance allocation, read from the trained "
      "model. The no-logdet column is an offline structural ablation and was **not trained**.\n")
    keys = [("group_rvm", "RVM"), ("conditional_newton", "Newton"), ("oplora", "OPLoRA"), ("eva", "EVA"),
            ("rvm_no_logdet (offline ablation, not trained)", "no-logdet†")]
    for cfgkey, a in alloc.items():
        A(f"**{cfgkey}** (rank units: " + ", ".join(f"{n} {sum((a.get(k) or {}).values())}" for k, n in keys) + ")\n")
        A("| Layer | " + " | ".join(n for _, n in keys) + " |")
        A("|---|" + "---:|" * len(keys))
        for layer in range(cfg["model"]["num_layers"]):
            q, v = f"roberta.encoder.layer.{layer}.attention.self.query", f"roberta.encoder.layer.{layer}.attention.self.value"
            A(f"| L{layer} | " + " | ".join(f"{(a.get(k) or {}).get(q, 0)}/{(a.get(k) or {}).get(v, 0)}"
                                             for k, _ in keys) + " |")
        A("")
    A("**Sparsity and stopping per configuration:**\n")
    A("Parameter reduction of RVM: exact RVM parameters relative to Uniform r2 (B48, 73,728) and to the B24 "
      "budget (36,864): " + "; ".join(
          f"{t}/{s} {by[('group_rvm', t, s)]['adapter_params']:,} = "
          f"{by[('group_rvm', t, s)]['adapter_params'] / sb['comparison_parameters']:.1%} of B48, "
          f"{by[('group_rvm', t, s)]['adapter_params'] / sb['budget_parameters']:.1%} of B24"
          for t in tasks for s in seeds if by[("group_rvm", t, s)]["adapter_params"] is not None) + ".\n")
    A("| Task/seed | RVM rank units (params) | RVM stop reason | RVM threshold (nats/param) | Fresh-null activations "
      "(diagnostic) | RVM == Newton? | Newton-with-own-threshold units (diagnostic) | CE RVM | CE Uniform r1 | "
      "CE Newton |")
    A("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    for t in tasks:
        for s in seeds:
            sel = _load(probe_dir(t, s) / "selection.json") or {}
            gr, cn = sel.get("group_rvm", {}), sel.get("conditional_newton", {})
            A(f"| {t}/{s} | {gr.get('rank_units')} ({by[('group_rvm', t, s)]['adapter_params']}) | {gr.get('stop_reason')} | "
              f"{_full(gr.get('threshold'))} | {sel.get('group_rvm_fresh_null', {}).get('activations')}/100 | "
              f"{gr.get('ranks') == cn.get('ranks')} | {cn.get('diagnostic_thresholded_units')} | "
              f"{_f(by[('group_rvm', t, s)]['official_validation_loss'], 5)} | "
              f"{_f(by[('uniform_r1', t, s)]['official_validation_loss'], 5)} | "
              f"{_f(by[('conditional_newton', t, s)]['official_validation_loss'], 5)} |")
    A(f"\n![{Path(figs[3]).stem}]({Path(figs[3]).relative_to(REL)})\n")
    A(nv("c_structure") + "\n")

    # ------------------------------------------------------------------ D
    A("## D. Computational efficiency (observed)\n")
    u1 = st["overhead"]["median_uniform_r1_train_s"]
    A("| Method | Mean selection s | Mean train s | Mean total s | Selection / median Uniform-r1 train | "
      "Mean exact params | Peak GPU MB (max) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        rr = [r for r in ok if r["method"] == m]
        if not rr:
            continue
        sel = float(np.mean([r["selection_wall_s"] or 0 for r in rr]))
        tot = [r["total_wall_s"] for r in rr if r["total_wall_s"] is not None]
        pk_ = [r["peak_gpu_memory_mb"] for r in rr if r["peak_gpu_memory_mb"] is not None]
        A(f"| {LABEL[m]} | {sel:.2f} | {np.mean([r['train_wall_s'] for r in rr]):.1f} | "
          f"{_f(float(np.mean(tot)) if tot else None, 1)} | {sel / u1:.3f} | "
          f"{np.mean([r['adapter_params'] for r in rr]):,.0f} | {_f(max(pk_) if pk_ else None, 0)} |")
    oc = st["overhead"]
    A(f"\n**Overhead criterion (frozen, mandatory):** mean RVM selection {oc['mean_rvm_selection_s']:.3f} s / median "
      f"Uniform-r1 training {oc['median_uniform_r1_train_s']:.3f} s = **{oc['ratio']:.4f}** vs cutoff ≤ {oc['cutoff']} → "
      f"**{oc['status']}** (counterfactual).\n")
    A("RVM selection-cost breakdown per configuration (s; GPU passes are shared with Newton, PROTOCOL_CLARIFICATIONS C5):\n")
    A("| Task/seed | Direction pass (GPU) | Feature pass (GPU) | RVM null calibration + greedy (CPU) | Job start-up | "
      "RVM total charged | Newton total charged | OPLoRA total |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in tasks:
        for s in seeds:
            p = _load(probe_dir(t, s) / "probe_result.json") or {}
            sel = _load(probe_dir(t, s) / "selection.json") or {}
            tm = p.get("timings", {})
            A(f"| {t}/{s} | {_f(tm.get('direction_pass_s'), 2)} | {_f(tm.get('feature_pass_s'), 2)} | "
              f"{_f(sel.get('group_rvm', {}).get('cpu_wall_s'), 2)} | {_f(tm.get('selection_job_startup_s'), 2)} | "
              f"{_f(tm.get('group_rvm_selection_s'), 2)} | {_f(tm.get('conditional_newton_selection_s'), 2)} | "
              f"{_f(tm.get('oplora_selection_s'), 2)} |")
    # non-overlapping sum: probe stage measured once per task/seed; per-run time without the charged selection
    # share (EVA calibrates inside its run, so its whole run time counts)
    run_exec = sum((r["total_wall_s"] or 0) - (0.0 if r["method"] == "eva" else (r["selection_wall_s"] or 0.0))
                   for r in ok if not r["empty_selection"])
    tot_heads = sum(a["wall_s"] or 0 for a in aux if a["kind"] == "head_warm_start")
    tot_probe = sum(a["wall_s"] or 0 for a in aux if a["kind"] == "probe_selection")
    tot_train = sum(r["train_wall_s"] or 0 for r in ok)
    total = run_exec + tot_heads + tot_probe
    span = st["log_span"]
    A(f"\n**Total measured execution cost (non-overlapping):** 4 head warm starts {tot_heads:.1f} s + 4 probe/selection "
      f"stages {tot_probe:.1f} s (GPU direction/feature/OPLoRA passes and the CPU selection job, incl. diagnostics) + "
      f"the adapter runs' own load/train/evaluate time {run_exec:.1f} s (of which 600-step training {tot_train:.1f} s) = "
      f"**{total:.1f} s ≈ {total / 3600:.3f} GPU-hours** on one RTX A4000, one job at a time. Wall-clock span of the "
      f"run script: {span['first_start']} → {span['last_done']} ({span['script_starts']} start(s)), including the "
      f"Gate-0 suite ({_f(g0.get('wall_s'), 1)} s) and model downloads/caching.\n")
    A(f"![{Path(figs[4]).stem}]({Path(figs[4]).relative_to(REL)})\n")
    A(nv("d_compute") + "\n")

    # ------------------------------------------------------------------ E
    A("## E. Weak classification-head limitation\n")
    rh = _load(ref / "gate1" / "head_only_audit.json") or {}
    A(f"In the corrected v2 experiment the frozen QNLI/42 head-only model (RoBERTa backbone + 200-step "
      f"warm-started head, no adapter) reached internal-audit CE **{_full(rh.get('audit_loss'))}** nats and accuracy "
      f"**{_f(rh.get('audit_accuracy'), 4)}**, against ln 2 = {LN2:.6f} for a uniform predictor "
      f"(`{ref.relative_to(ROOT)}/gate1/head_only_audit.json`). The same frozen-head protocol is used here, unchanged, "
      "for all four task/seed heads:\n")
    A("| Head | SHA256 | Warm-start train loss (first 20 steps) | (last 20 steps) | Empty-selection head-only "
      "validation CE (if any) |\n|---|---|---:|---:|---|")
    for t in tasks:
        for s in seeds:
            h = _load(head_meta_path(t, s)) or {}
            empt = [r for r in rows if r["task"] == t and r["seed"] == s and r["empty_selection"]]
            A(f"| {t}/{s} | `{(h.get('head_checkpoint_sha256') or '')[:16]}…` | {_f(h.get('head_train_loss_first20'), 4)} | "
              f"{_f(h.get('head_train_loss_last20'), 4)} | "
              + (", ".join(f"{LABEL[r['method']]} {r['official_validation_loss']:.5f}" for r in empt) or "none") + " |")
    A("\nNo head-only official-validation model was evaluated except where a selector chose no module. "
      "Running extra head-only evaluations would add configurations outside the frozen 24-run design. The "
      "differences between methods cannot come from a stronger classifier or a changed initialisation. Every "
      "method shares the identical frozen head per task/seed. Every non-EVA method shares the identical "
      "zero-update LoRA initialisation. EVA keeps its native initialisation and is reported as such.\n")
    A(nv("e_weak_head") + "\n")

    # ------------------------------------------------------------------ counterfactual gate
    A("## Counterfactual Gate-2 evaluation (frozen criteria, labelled exploratory)\n")
    A(f"Evaluator: `rvm_lora.gates.gate2_criteria`. It is the preregistered Gate-2 code, factored out of the Gate-1 "
      f"check without changing any calculation, as asserted by `tests/test_gate2_override.py`. Thresholds come "
      f"from `configs/preregistered.yaml`. Strongest B24 competitor (post-hoc, lowest 4-run mean CE): "
      f"**{LABEL.get(st['best_b24_baseline'], st['best_b24_baseline'])}**.\n")
    A("Mean CE per method: `" + json.dumps(st["L_mean_loss_per_method"]) + "`\n")
    A(_gate_table(st["criteria"]))
    b = st["bootstrap_vs_best_b24"] or {}
    A(f"\nRoute A bootstrap (RVM vs {st['best_b24_baseline']}): point {_full(b.get('point'))}, 90% interval "
      f"[{_full(b.get('lower'))}, {_full(b.get('upper'))}] ({b.get('n_boot')} resamples, seed {b.get('seed')}).\n")
    A(f"- Route A (quality) satisfied: **{st['route_a']}**; Route B (compression) satisfied: **{st['route_b']}**; "
      f"overhead satisfied: **{oc['status'] == 'PASS'}**.")
    A(f"- `exploratory_gate2_standalone_result` = **{st['exploratory_gate2_standalone_result']}**: what the frozen Gate-2 "
      "rule (`(Route A or Route B) and all mandatory conditions`) gives for these 24 runs on their own.")
    A(f"- `preregistered_overall_decision` = **{st['preregistered_overall_decision']}**. Gate 1 failed on valid "
      "measurements. Under the frozen conditional progression, no Gate-2 outcome can change that.\n")

    # ------------------------------------------------------------------ F
    A("## F. Final assessment (interpretation)\n")
    qs = ["Does Group RVM beat Uniform LoRA at the same budget?", "Does it beat EVA and OPLoRA?",
          "Does it beat Conditional Newton?", "Does its automatic rank sparsification produce useful compression?",
          "Does it have acceptable selection overhead?", "Would Gate 2 independently pass the original criteria?",
          "Do the results provide any evidence that the original `NO_GO` deserves scientific reconsideration?"]
    fa = nar.get("f_answers", {})
    for i, q in enumerate(qs, 1):
        A(f"**{i}. {q}** {fa.get(str(i), '_(not yet written)_')}\n")
    A("## Deviations, failures and limitations\n")
    oom = [r for r in rows if r["oom_events"]]
    fails = [r for r in rows if r["status"] != "COMPLETED"]
    A(f"- Runs not COMPLETED: {len(fails)}" + (": " + ", ".join(f"{r['method']}/{r['task']}/{r['seed']} ({r['status']})"
                                                             for r in fails) if fails else "") + ".")
    A(f"- OOM fallback (microbatch 4 × 8): {len(oom)} runs" + (": " + ", ".join(f"{r['method']}/{r['task']}/{r['seed']}"
                                                                               for r in oom) if oom else "") +
      f"; microbatch used: {sorted({r['microbatch'] for r in ok if r['microbatch']})}.")
    A(f"- Run attempts > 1: {[(r['method'], r['task'], r['seed'], r['attempt']) for r in rows if (r['attempt'] or 1) > 1] or 'none'}.")
    A(nv("deviations") + "\n")
    A("## Reproduction\n")
    A("```bash\n# on the GPU host, fresh checkout of this commit; MLflow on 127.0.0.1:5000 (scripts/start_mlflow.sh)\n"
      "uv sync --frozen\n"
      "tmux new-session -d -s rvm_gate2_exploratory 'bash scripts/run_gate2_exploratory.sh'\n"
      "#   = integrity snapshot of v2 -> env-check -> prepare (reuse splits) -> gate0 -> freeze ->\n"
      "#     train-b --exploratory-override-gate1 --gate1-record artifacts/v2/gate1/gate1.json -> eval-gate2 (same flags)\n"
      f"RVM_RUN_NAMESPACE={RUN_NAMESPACE} uv run --frozen python -m rvm_lora.cli report-exploratory \\\n"
      "    --exploratory-override-gate1 --gate1-record artifacts/v2/gate1/gate1.json\n```\n")
    A(f"Review bundle: `{REL}/review_bundle.zip` (SHA256 in `{REL}/review_bundle.sha256`). It contains this report, "
      "the CSVs, `standalone_gate2.json`, an offline MLflow export, every run's result/curve, per-example "
      "validation losses (so the bootstrap can be recomputed), head/probe/selection records, the v2 reference "
      "records, the integrity snapshots, source, tests, configs, docs and `uv.lock`. It contains no weights, "
      "datasets, databases or credentials.\n")
    A("## Appendix: run index\n")
    A("| Stage | Kind | Method | Task | Seed | Status | Wall s | MLflow run |\n|---|---|---|---|---|---|---:|---|")
    for a in aux:
        A(f"| {a['stage']} | {a['kind']} | {a['method']} | {a['task']} | {a['seed']} | {a['status']} | "
          f"{_f(a['wall_s'], 1)} | `{a['mlflow_run_id']}` |")
    for r in rows:
        A(f"| gate2 | {'head_only_eval' if r['empty_selection'] else 'adapter_run'} | {r['method']} | {r['task']} | "
          f"{r['seed']} | {r['status']} | {_f(r['total_wall_s'], 1)} | `{r['mlflow_run_id']}` |")
    return "\n".join(L) + "\n"


# =============================================================================== bundle
def build_bundle(cfg: Config, ref: Path, ref_rep: Path) -> str:
    zp = REPORTS / "review_bundle.zip"
    files = [REPORTS / f for f in ("FINAL_REPORT.md", "results.csv", "run_index.csv", "standalone_gate2.json",
                                   "mlflow_export.json", "narrative.json", "environment.json", "protocol_freeze.json")]
    files += sorted(FIG.glob("*.png"))
    for sub in ("gate0", "gate2", "integrity", "heads"):
        files += sorted((ART / sub).glob("*.json"))
    if (ART / "gate0" / "pytest_junit.xml").exists():
        files.append(ART / "gate0" / "pytest_junit.xml")
    for pdir in sorted((ART / "probes").glob("*_s*")):
        files += [pdir / f for f in ("probe_result.json", "selection.json", "oplora.json")]
    for d in sorted((ART / "runs" / "gate2").glob("*")):
        files += [d / f for f in ("result.json", "train_curve.json", "eval_validation.npz")]
    files += [ref_rep / "gates.json", ref_rep / "FINAL_REPORT.md", ref / "gate1" / "gate1.json",
              ref / "gate1" / "head_only_audit.json", ref / "heads" / "qnli_s42.json",
              ref / "probes" / "qnli_s42" / "probe_result.json", ref / "probes" / "qnli_s42" / "selection.json"]
    files += [ROOT / f for f in ("configs/preregistered.yaml", "configs/baseline_fidelity.json", "docs/PREREGISTRATION.md",
                                 "docs/METHODS.md", "docs/PROTOCOL_CLARIFICATIONS.md", "pyproject.toml", "uv.lock",
                                 "README.md", "AGENTS.md", "reports/splits/qnli_split_manifest.json",
                                 "reports/splits/sst2_split_manifest.json", "reports/CORRECTNESS_AUDIT.md")]
    files += sorted((ROOT / "src" / "rvm_lora").glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + \
        sorted((ROOT / "scripts").glob("*.sh"))
    manifest = {}
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if not f.exists():
                continue
            rel = str(f.relative_to(ROOT))
            z.write(f, rel)
            manifest[rel] = sha256_file(f)
        if LOG.exists():
            lines = [ln for ln in LOG.read_text(errors="replace").splitlines()
                     if not any(s in ln for s in ("it/s]", "s/it]", "Loading weights", "Generating"))]
            z.writestr(f"logs/{LOG.stem}_compact.log", "\n".join(lines[-6000:]))
        try:
            diff = subprocess.check_output(["git", "log", "--oneline", "4c54b1174b7501f7843407eaf042e8398ead6c2b..HEAD"],
                                           cwd=ROOT, text=True)
        except Exception as e:
            diff = repr(e)
        z.writestr("GIT_LOG_SINCE_4c54b11.txt", diff)
        z.writestr("BUNDLE_MANIFEST.json", json.dumps({"git_sha": git_sha(), "created": now(), "namespace": RUN_NAMESPACE,
                                                       "files_sha256": manifest}, indent=2))
    sha = sha256_file(zp)
    (REPORTS / "review_bundle.sha256").write_text(f"{sha}  {REL}/review_bundle.zip\n")
    return sha


_ = GRID
