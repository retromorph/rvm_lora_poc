"""Stage implementations. Every stage is restartable: completed outputs are detected and reused."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import data as D
from .config import DATASET_REVISION, MODEL_REVISION, PREREG_MD, REPO_ROOT, Config, sha256_bytes, sha256_file
from .tracking import Tracker, mirror_dir, read_mirror, run_key, write_json

ROOT = REPO_ROOT
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def common_tags(cfg: Config) -> dict:
    return {"git_sha": git_sha(), "protocol_sha256": cfg.sha256, "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION}


def data_file(task: str, part: str) -> Path:
    return ROOT / "data" / f"{task}_{part}.npz"


# ============================================================================ prepare
def prepare(cfg: Config) -> dict:
    out = {}
    tok = None
    for task in cfg.tasks:
        mp = D.manifest_path(ROOT, task)
        parts = ["head", "directions", "evidence", "adapter_train", "audit"]
        if mp.exists() and all(data_file(task, p).exists() for p in parts):
            out[task] = D.load_manifest(ROOT, task)["sha256"]
            log(f"prepare {task}: reuse existing manifest/encodings")
            continue
        ds = D.load_glue(task, "train")
        idx = np.asarray(ds["idx"], dtype=np.int64)
        split = D.make_split(idx, cfg)
        D.validate_split(split, cfg, len(ds))
        manifest = D.build_manifest(task, split, len(ds), getattr(ds, "_fingerprint", "unknown"))
        write_json(mp, manifest)
        tok = tok or D.get_tokenizer()
        pos = {int(i): k for k, i in enumerate(idx)}
        for p in parts:
            sub = ds.select([pos[i] for i in split[p]])
            enc = D.encode(task, sub, tok, int(cfg["sequence_length"]))
            assert list(enc.idx) == split[p]
            D.save_encoded(data_file(task, p), enc)
        out[task] = manifest["sha256"]
        log(f"prepare {task}: n_train={len(ds)} manifest sha(audit)={manifest['sha256']['audit'][:12]}")
    return out


def split_hashes(cfg: Config) -> dict:
    return {t: {"manifest_file_sha256": sha256_file(D.manifest_path(ROOT, t)),
                **D.load_manifest(ROOT, t)["sha256"]} for t in cfg.tasks}


# ============================================================================ head warm start
def head_meta_path(task: str, seed: int) -> Path:
    return ART / "heads" / f"{task}_s{seed}.json"


def warm_head(cfg: Config, task: str, seed: int, tracker: Tracker | None = None) -> dict:
    from .modeling import head_path, load_backbone, save_head, warm_start_head

    mp, hp = head_meta_path(task, seed), head_path(ROOT, task, seed)
    if mp.exists() and hp.exists():
        meta = json.loads(mp.read_text())
        if sha256_file(hp) != meta["head_checkpoint_sha256"]:
            raise RuntimeError(f"head checkpoint hash mismatch for {task}/{seed}")
        return meta
    log(f"warm-head {task} seed={seed}")
    t0 = time.perf_counter()
    model = load_backbone(seed)
    enc = D.load_encoded(data_file(task, "head"))
    info = warm_start_head(model, enc, cfg, seed)
    sha = save_head(model, hp)
    meta = {"task": task, "seed": seed, "head_checkpoint": str(hp.relative_to(ROOT)), "head_checkpoint_sha256": sha,
            "head_examples": len(enc), "head_ids_sha256": D.ids_sha256(list(enc.idx)), **info,
            "total_wall_s": time.perf_counter() - t0, "timestamp": now()}
    write_json(mp, meta)
    if tracker:
        gate = "gate1" if (task, seed) == (cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]) else "gate2"
        key = run_key(cfg.sha256, gate, "head_warm_start", task, seed, "head")
        rid, _ = tracker.start(gate, key, {**common_tags(cfg), "gate": gate, "method": "head_warm_start",
                                           "task": task, "seed": seed, "head_checkpoint_sha256": sha},
                               {"steps": cfg["head_warm_start"]["steps"], "lr": cfg["head_warm_start"]["learning_rate"]})
        tracker.log_metrics(rid, {k: v for k, v in info.items() if isinstance(v, (int, float))})
        tracker.log_artifact(rid, mp)
        tracker.finish(rid, "COMPLETED")
        meta["mlflow_run_id"] = rid
        write_json(mp, meta)
    del model
    _free_cuda()
    return meta


def _free_cuda():
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================================ probe + selection
def probe_dir(task: str, seed: int) -> Path:
    return ART / "probes" / f"{task}_s{seed}"


def probe(cfg: Config, task: str, seed: int, gate1: bool, tracker: Tracker | None = None) -> dict:
    """GPU probes (directions, features, OPLoRA importance) then the CPU selection subprocess."""
    import torch

    from .modeling import configure_precision, head_path, load_backbone, load_head
    from .probe import direction_pass, feature_pass, oplora_importance_pass

    pdir = probe_dir(task, seed)
    final = pdir / "probe_result.json"
    if final.exists():
        return json.loads(final.read_text())
    configure_precision()
    meta = warm_head(cfg, task, seed, tracker)
    clip = float(cfg["probe"]["probability_clip"])
    model = load_backbone(seed)
    head_sha = load_head(model, head_path(ROOT, task, seed))
    assert head_sha == meta["head_checkpoint_sha256"]
    model.eval()
    dirs_enc = D.load_encoded(data_file(task, "directions"))
    evid_enc = D.load_encoded(data_file(task, "evidence"))
    log(f"probe {task}/{seed}: direction pass")
    dirs = direction_pass(model, dirs_enc, cfg.modules, cfg.n_dir, clip)
    log(f"probe {task}/{seed}: feature pass")
    feats = feature_pass(model, evid_enc, cfg.modules, dirs["U"], dirs["V"], clip)
    log(f"probe {task}/{seed}: OPLoRA importance pass")
    imp = oplora_importance_pass(model, dirs_enc, cfg.modules)
    pdir.mkdir(parents=True, exist_ok=True)
    torch.save({"U": dirs["U"], "V": dirs["V"], "S": dirs["S"], "modules": cfg.modules}, pdir / "atoms.pt")
    np.savez(pdir / "features.npz", Phi=feats["Phi"], t=feats["t"], J=feats["J"], z=feats["z"], p=feats["p"])
    Phi, t = feats["Phi"], feats["t"]
    K = Phi.T @ Phi
    eig = np.linalg.eigvalsh(K)
    checks = {
        "n_atoms": int(Phi.shape[1]), "n_evidence_rows": int(Phi.shape[0]),
        "all_finite": bool(np.isfinite(Phi).all() and np.isfinite(t).all()),
        "min_column_norm": float(np.min(np.linalg.norm(Phi, axis=0))),
        "max_column_norm": float(np.max(np.linalg.norm(Phi, axis=0))),
        "K_eig_min": float(eig[0]), "K_eig_max": float(eig[-1]),
        "t_mean": float(t.mean()), "t_var": float(t.var()),
        "p_clipped_fraction": float(np.mean((feats["p"] <= clip) | (feats["p"] >= 1 - clip))),
        "svd_top_singular_values": dirs["S"][:, 0].tolist(),
        "u_unit_norm_max_err": float((dirs["U"].norm(dim=1) - 1).abs().max()),
        "v_unit_norm_max_err": float((dirs["V"].norm(dim=1) - 1).abs().max()),
    }
    hashes = {"Phi_sha256": sha256_bytes(np.ascontiguousarray(Phi).tobytes()),
              "t_sha256": sha256_bytes(np.ascontiguousarray(t).tobytes()),
              "atoms_sha256": sha256_file(pdir / "atoms.pt"), "features_file_sha256": sha256_file(pdir / "features.npz"),
              "directions_ids_sha256": D.ids_sha256(list(dirs_enc.idx)),
              "evidence_ids_sha256": D.ids_sha256(list(evid_enc.idx))}
    del model
    _free_cuda()
    # ---- CPU selection in a fresh process with single-threaded BLAS (fork-safe, no CUDA)
    log(f"probe {task}/{seed}: selection job")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "rvm_lora.selection_job", str(pdir), str(pdir / "selection.json"),
           "--config", str(cfg.path)] + (["--gate1"] if gate1 else [])
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    job_wall = time.perf_counter() - t0
    sel = json.loads((pdir / "selection.json").read_text())
    seg = sum(sel[k]["cpu_wall_s"] for k in ("conditional_newton", "group_rvm", "rvm_no_logdet")) + \
        sel["group_rvm_fresh_null"]["wall_s"] + sel["conditional_newton"]["diagnostic_wall_s"] + \
        sel.get("architecture_scores_wall_s", 0.0)
    startup = max(0.0, job_wall - seg)
    gpu = dirs["wall_s"] + feats["wall_s"]
    timings = {
        "direction_pass_s": dirs["wall_s"], "feature_pass_s": feats["wall_s"], "oplora_importance_pass_s": imp["wall_s"],
        "selection_job_wall_s": job_wall, "selection_job_startup_s": startup,
        # method-specific calibration/selection cost (conservative: RVM also pays the job start-up)
        "group_rvm_selection_s": gpu + sel["group_rvm"]["cpu_wall_s"] + startup,
        "conditional_newton_selection_s": gpu + sel["conditional_newton"]["cpu_wall_s"] + startup,
        "oplora_selection_s": None,  # filled by the OPLoRA allocation below
    }
    from .baselines import oplora_allocate

    t1 = time.perf_counter()
    op_ranks, op_obj = oplora_allocate(imp["importance"], int(cfg["stage_b"]["budget_rank_units"]),
                                       list(range(0, cfg.max_rank + 1)))
    timings["oplora_selection_s"] = imp["wall_s"] + (time.perf_counter() - t1)
    result = {"task": task, "seed": seed, "head_checkpoint_sha256": head_sha, "checks": checks, "hashes": hashes,
              "timings": timings, "probe_sizes": {"directions": len(dirs_enc), "evidence": len(evid_enc)},
              "passes": {"direction": 1, "feature": 1, "oplora_importance": 1},
              "oplora": {"importance": imp["importance"], "ranks": op_ranks, "objective": op_obj,
                         "rank_choices": list(range(0, cfg.max_rank + 1)),
                         "source": "https://github.com/cjpark-ai/DPLoRA@6666d8bdb37ebd153a8916b0464c8a3878976aae"},
              "selection_file_sha256": sha256_file(pdir / "selection.json"), "timestamp": now()}
    write_json(pdir / "oplora.json", result["oplora"])
    if gate1:
        scores = {"arch_scores": sel["architecture_scores"], "lambda": sel["architecture_scores_lambda"],
                  "frozen_at": now(), "selection_file_sha256": result["selection_file_sha256"],
                  "note": "precomputed before any audit label was read"}
        write_json(ART / "gate1" / "precomputed_scores.json", scores)
        result["precomputed_scores_sha256"] = sha256_file(ART / "gate1" / "precomputed_scores.json")
    if tracker:
        gate = "gate1" if gate1 else "gate2"
        key = run_key(cfg.sha256, gate, "probe_selection", task, seed, "probe")
        rid, _ = tracker.start(gate, key, {**common_tags(cfg), "gate": gate, "method": "probe_selection", "task": task,
                                           "seed": seed, "head_checkpoint_sha256": head_sha},
                               {"directions": len(dirs_enc), "evidence": len(evid_enc), "n_atoms": Phi.shape[1]})
        tracker.log_metrics(rid, {k: v for k, v in timings.items() if v is not None})
        tracker.log_metrics(rid, {"rvm_threshold": sel["group_rvm"]["threshold"],
                                  "rvm_rank_units": sel["group_rvm"]["rank_units"],
                                  "newton_rank_units": sel["conditional_newton"]["rank_units"],
                                  "rvm_fresh_null_activations": sel["group_rvm_fresh_null"]["activations"]})
        write_json(pdir / "probe_result.json", result)
        for f in ("probe_result.json", "selection.json", "oplora.json"):
            tracker.log_artifact(rid, pdir / f)
        if gate1:
            tracker.log_artifact(rid, ART / "gate1" / "precomputed_scores.json")
        tracker.finish(rid, "COMPLETED")
        result["mlflow_run_id"] = rid
    write_json(final, result)
    return result


# ============================================================================ generic adapter run
def adapter_run(cfg: Config, tracker: Tracker, gate: str, method: str, task: str, seed: int, arch_id: str,
                ranks: dict[str, int] | None, eval_part: str, eval_metric_prefix: str, selection_wall_s: float | None,
                extra_tags: dict | None = None, eva: bool = False) -> dict:
    """Train one adapter with the fixed protocol and evaluate the final checkpoint once."""
    import torch

    from .adapters import assert_exact_layout, build_peft, lora_config, lora_layout
    from .modeling import configure_precision, evaluate, head_path, load_backbone, load_head
    from .train import train_with_oom_fallback

    key = run_key(cfg.sha256, gate, method, task, seed, arch_id)
    prev = read_mirror(ROOT, gate, key)
    if prev and prev.get("status") == "COMPLETED":
        log(f"reuse completed run {gate}/{method}/{task}/{seed}/{arch_id} ({prev.get('mlflow_run_id')})")
        return prev
    configure_precision()
    hmeta = warm_head(cfg, task, seed, tracker)
    mdir = mirror_dir(ROOT, gate, key)
    mdir.mkdir(parents=True, exist_ok=True)
    tags = {**common_tags(cfg), "gate": gate, "method": method, "task": task, "seed": seed, "arch_id": arch_id,
            "head_checkpoint_sha256": hmeta["head_checkpoint_sha256"],
            "data_split_sha256": D.load_manifest(ROOT, task)["sha256"]["adapter_train"], **(extra_tags or {})}
    at = cfg["adapter_training"]
    params = {"ranks_json": json.dumps(ranks, sort_keys=True) if ranks else "EVA", "steps": at["steps"],
              "lr": at["learning_rate"], "weight_decay": at["weight_decay"], "effective_batch": at["effective_batch_size"],
              "lora_alpha": at["lora_alpha"], "use_rslora": at["use_rslora"], "eval_split": eval_part}
    rid, attempt = tracker.start(gate, key, tags, params)
    t_total = time.perf_counter()
    result = {"run_key": key, "gate": gate, "method": method, "task": task, "seed": seed, "arch_id": arch_id,
              "mlflow_run_id": rid, "attempt": attempt, "status": "RUNNING", "started": now()}
    write_json(mdir / "result.json", result)
    try:
        train_enc = D.load_encoded(data_file(task, "adapter_train"))
        sel_s = selection_wall_s
        eva_info = {}

        def build():
            nonlocal sel_s, eva_info
            model = load_backbone(seed)
            load_head(model, head_path(ROOT, task, seed))
            if eva:
                from peft import EvaConfig, get_peft_model

                from .baselines import eva_initialize
                from .modeling import batches

                cfg_l = lora_config({m: 1 for m in cfg.modules}, cfg, init_lora_weights="eva",
                                    eva_config=EvaConfig(rho=float(cfg["stage_b"]["eva_rho"])))
                pm = get_peft_model(model, cfg_l)
                denc = D.load_encoded(data_file(task, "directions"))
                bl = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                      for b in batches(denc, np.arange(len(denc)), 32, "cuda")]
                torch.cuda.synchronize()
                info = eva_initialize(pm, bl)
                torch.cuda.synchronize()
                for n, p in pm.named_parameters():
                    p.requires_grad_("lora_A" in n or "lora_B" in n)
                lay = lora_layout(pm)
                b_zero = all(float(mod.lora_B["default"].weight.abs().max()) == 0.0
                             for mod in pm.modules() if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}))
                eva_info = {"eva_ranks": {k: v["rank"] for k, v in lay.items()},
                            "eva_scaling": {k: v["scaling"] for k, v in lay.items()},
                            "eva_lora_B_zero_at_init": b_zero, "eva_calibration_examples": len(denc), **info}
                sel_s = info["eva_wall_s"]
                return pm, {"ranks": eva_info["eva_ranks"]}
            return build_peft(model, ranks, cfg, seed), {"ranks": ranks}

        def mlog(m, step):
            tracker.log_metrics(rid, m, step)

        peft_model, meta, tr = train_with_oom_fallback(build, train_enc, cfg, seed, log_fn=mlog)
        final_ranks = meta["ranks"]
        acct = assert_exact_layout(peft_model, final_ranks, cfg.modules, cfg.rank_unit_params)
        if eva and acct["adapter_params"] > cfg["stage_b"]["budget_parameters"]:
            raise AssertionError("EVA exceeded B24")
        eval_enc = D.load_encoded(data_file(task, eval_part))
        ev = evaluate(peft_model, eval_enc)
        np.savez(mdir / f"eval_{eval_part}.npz", idx=ev["idx"], loss=ev["per_example_loss"], logits=ev["logits"])
        total = time.perf_counter() - t_total + (sel_s or 0.0)
        metrics = {f"{eval_metric_prefix}_loss": ev["loss"], f"{eval_metric_prefix}_accuracy": ev["accuracy"],
                   "train_loss": tr["train_loss_final"], "train_wall_s": tr["train_wall_s"],
                   "selection_wall_s": sel_s, "total_wall_s": total, "peak_gpu_memory_mb": tr["peak_gpu_memory_mb"],
                   "adapter_params": acct["adapter_params"], "rank_sum": acct["rank_sum"]}
        tracker.log_metrics(rid, metrics)
        result.update(status="COMPLETED", finished=now(), metrics=metrics, ranks=final_ranks,
                      layout=acct["layout"], microbatch=tr["microbatch"], accumulation=tr["accumulation"],
                      precision_train=tr["precision"], precision_eval="float32", oom_events=tr["oom_events"],
                      head_digest=tr["head_digest"], train_loss_first=tr["train_loss_first"], eva=eva_info or None,
                      eval_file_sha256=sha256_file(mdir / f"eval_{eval_part}.npz"))
        write_json(mdir / "train_curve.json", tr["train_loss_curve"])
        write_json(mdir / "result.json", result)
        tracker.log_artifact(rid, mdir / "result.json")
        tracker.log_artifact(rid, mdir / "train_curve.json")
        tracker.finish(rid, "COMPLETED")
        log(f"done {gate}/{method}/{task}/{seed}/{arch_id}: {eval_metric_prefix}_loss={ev['loss']:.6f} "
            f"acc={ev['accuracy']:.4f} train={tr['train_wall_s']:.1f}s params={acct['adapter_params']}")
    except Exception as e:  # keep failures visible, never silently drop
        result.update(status="FAILED", finished=now(), error=repr(e), traceback=traceback.format_exc())
        write_json(mdir / "result.json", result)
        tracker.log_artifact(rid, mdir / "result.json")
        tracker.finish(rid, "FAILED")
        log(f"FAILED {gate}/{method}/{task}/{seed}/{arch_id}: {e!r}")
    finally:
        _free_cuda()
    return result


# ============================================================================ Gate 1
def head_only_audit(cfg: Config, tracker: Tracker) -> dict:
    from .modeling import configure_precision, evaluate, head_path, load_backbone, load_head

    task, seed = cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]
    p = ART / "gate1" / "head_only_audit.json"
    if p.exists():
        return json.loads(p.read_text())
    configure_precision()
    model = load_backbone(seed)
    sha = load_head(model, head_path(ROOT, task, seed))
    audit = D.load_encoded(data_file(task, "audit"))
    ev = evaluate(model, audit)
    key = run_key(cfg.sha256, "gate1", "head_only", task, seed, "head_only")
    rid, _ = tracker.start("gate1", key, {**common_tags(cfg), "gate": "gate1", "method": "head_only", "task": task,
                                          "seed": seed, "arch_id": "head_only", "head_checkpoint_sha256": sha}, {})
    tracker.log_metrics(rid, {"audit_loss": ev["loss"], "audit_accuracy": ev["accuracy"], "adapter_params": 0})
    np.savez(ART / "gate1" / "head_only_audit.npz", idx=ev["idx"], loss=ev["per_example_loss"], logits=ev["logits"])
    out = {"audit_loss": ev["loss"], "audit_accuracy": ev["accuracy"], "n": len(audit), "mlflow_run_id": rid,
           "head_checkpoint_sha256": sha, "timestamp": now()}
    write_json(p, out)
    tracker.log_artifact(rid, p)
    tracker.finish(rid, "COMPLETED")
    del model
    _free_cuda()
    return out


def oracle_a(cfg: Config, tracker: Tracker) -> list[dict]:
    task, seed = cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]
    freeze = REPORTS / "protocol_freeze.json"
    if not freeze.exists() or json.loads(freeze.read_text())["config_sha256"] != cfg.sha256:
        raise RuntimeError("protocol_freeze.json missing or does not match the config; refusing to read audit labels")
    if not (ART / "gate1" / "precomputed_scores.json").exists():
        raise RuntimeError("Gate-1 scores must be precomputed before any audit evaluation")
    head_only_audit(cfg, tracker)
    out = []
    for a in cfg.architectures():
        out.append(adapter_run(cfg, tracker, "gate1", "fixed_architecture", task, seed, a.arch_id, a.ranks,
                               "audit", "audit", None, {"family": a.family}))
    return out


def gate1_rows(cfg: Config) -> tuple[list[dict], list[tuple[str, str, str]], dict]:
    task, seed = cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]
    scores = json.loads((ART / "gate1" / "precomputed_scores.json").read_text())
    head = json.loads((ART / "gate1" / "head_only_audit.json").read_text())
    rows = []
    for s in scores["arch_scores"]:
        key = run_key(cfg.sha256, "gate1", "fixed_architecture", task, seed, s["arch_id"])
        m = read_mirror(ROOT, "gate1", key) or {}
        ok = m.get("status") == "COMPLETED"
        rows.append({**{k: s[k] for k in ("arch_id", "family", "parameters", "gradient_norm", "independent_fisher",
                                          "conditional_newton", "group_rvm")},
                     "ranks": s["ranks"], "rvm_log_alpha": s["rvm_log_alpha"], "rvm_boundary": s["rvm_boundary"],
                     "audit_loss": m["metrics"]["audit_loss"] if ok else None,
                     "audit_accuracy": m["metrics"]["audit_accuracy"] if ok else None,
                     "U": (head["audit_loss"] - m["metrics"]["audit_loss"]) if ok else None,
                     "status": m.get("status", "NOT_RUN"), "mlflow_run_id": m.get("mlflow_run_id"),
                     "train_wall_s": m["metrics"]["train_wall_s"] if ok else None})
    pairs = []
    for a in cfg.architectures():
        if a.family == "two_layer_rank2":
            i, j = a.layers
            pairs.append((a.arch_id, f"L{i:02d}_qv_r2", f"L{j:02d}_qv_r2"))
    return rows, pairs, head


def eval_gate1(cfg: Config) -> dict:
    from .gates import evaluate_gate1

    rows, pairs, head = gate1_rows(cfg)
    sel = json.loads((probe_dir(cfg["stage_a"]["task"], cfg["stage_a"]["run_seed"]) / "selection.json").read_text())
    null = {"fresh_activations": sel["group_rvm_fresh_null"]["activations"],
            "threshold": sel["group_rvm"]["threshold"]}
    g1 = evaluate_gate1(rows, pairs, null, cfg["stage_a"])
    g1["head_only_audit_loss"] = head["audit_loss"]
    g1["rvm_null"] = {"threshold": sel["group_rvm"]["threshold"],
                      "calibration_seed": cfg["probe"]["null_calibration_seed"],
                      "evaluation_seed": cfg["probe"]["null_evaluation_seed"],
                      "fresh_activations": sel["group_rvm_fresh_null"]["activations"],
                      "fresh_selected_units": sel["group_rvm_fresh_null"]["selected_units"],
                      "real_data_rank_units": sel["group_rvm"]["rank_units"],
                      "real_data_ranks": sel["group_rvm"]["ranks"], "real_data_stop": sel["group_rvm"]["stop_reason"]}
    write_json(ART / "gate1" / "gate1.json", {"gate1": g1, "rows": rows, "evaluated_at": now()})
    return g1


# ============================================================================ Gate 2
def gate2_methods_ranks(cfg: Config, task: str, seed: int) -> dict:
    pdir = probe_dir(task, seed)
    sel = json.loads((pdir / "selection.json").read_text())
    pr = json.loads((pdir / "probe_result.json").read_text())
    from .baselines import uniform_ranks

    return {
        "uniform_r1": (uniform_ranks(cfg.modules, 1), 0.0),
        "eva": (None, None),
        "oplora": (pr["oplora"]["ranks"], pr["timings"]["oplora_selection_s"]),
        "conditional_newton": (sel["conditional_newton"]["ranks"], pr["timings"]["conditional_newton_selection_s"]),
        "group_rvm": (sel["group_rvm"]["ranks"], pr["timings"]["group_rvm_selection_s"]),
        "uniform_r2": (uniform_ranks(cfg.modules, 2), 0.0),
    }


def ensure_validation(cfg: Config, task: str) -> None:
    if data_file(task, "validation").exists():
        return
    ds = D.load_glue(task, "validation")
    enc = D.encode(task, ds, D.get_tokenizer(), int(cfg["sequence_length"]))
    D.save_encoded(data_file(task, "validation"), enc)


def train_b(cfg: Config, tracker: Tracker) -> list[dict]:
    g1 = json.loads((ART / "gate1" / "gate1.json").read_text())["gate1"]
    if g1["status"] != "PASS":
        raise RuntimeError(f"Gate 1 status {g1['status']}: Gate 2 must not run")
    sb = cfg["stage_b"]
    out = []
    for task in sb["tasks"]:
        for seed in sb["seeds"]:
            warm_head(cfg, task, seed, tracker)
            probe(cfg, task, seed, gate1=False, tracker=tracker)
    for task in sb["tasks"]:
        ensure_validation(cfg, task)
    for task in sb["tasks"]:
        for seed in sb["seeds"]:
            mr = gate2_methods_ranks(cfg, task, seed)
            for method in sb["methods"]:
                ranks, sel_s = mr[method]
                if method in ("conditional_newton", "group_rvm", "oplora") and not ranks:
                    out.append(_record_empty(cfg, tracker, method, task, seed, sel_s))
                    continue
                out.append(adapter_run(cfg, tracker, "gate2", method, task, seed, method, ranks, "validation",
                                       "official_validation", sel_s, eva=(method == "eva")))
    return out


def _record_empty(cfg, tracker, method, task, seed, sel_s) -> dict:
    """A selector that selects no module yields the frozen head-only model; evaluate it as such."""
    from .modeling import configure_precision, evaluate, head_path, load_backbone, load_head

    key = run_key(cfg.sha256, "gate2", method, task, seed, method)
    prev = read_mirror(ROOT, "gate2", key)
    if prev and prev.get("status") == "COMPLETED":
        return prev
    configure_precision()
    model = load_backbone(seed)
    load_head(model, head_path(ROOT, task, seed))
    ev = evaluate(model, D.load_encoded(data_file(task, "validation")))
    mdir = mirror_dir(ROOT, "gate2", key)
    mdir.mkdir(parents=True, exist_ok=True)
    np.savez(mdir / "eval_validation.npz", idx=ev["idx"], loss=ev["per_example_loss"], logits=ev["logits"])
    rid, attempt = tracker.start("gate2", key, {**common_tags(cfg), "gate": "gate2", "method": method, "task": task,
                                               "seed": seed, "arch_id": method, "empty_selection": True}, {})
    metrics = {"official_validation_loss": ev["loss"], "official_validation_accuracy": ev["accuracy"],
               "selection_wall_s": sel_s, "train_wall_s": 0.0, "adapter_params": 0, "rank_sum": 0}
    tracker.log_metrics(rid, metrics)
    tracker.finish(rid, "COMPLETED")
    res = {"run_key": key, "gate": "gate2", "method": method, "task": task, "seed": seed, "arch_id": method,
           "mlflow_run_id": rid, "attempt": attempt, "status": "COMPLETED", "metrics": metrics, "ranks": {},
           "note": "empty selection: head-only model evaluated", "finished": now()}
    write_json(mdir / "result.json", res)
    return res


def eval_gate2(cfg: Config) -> dict:
    from .gates import NOT_RUN, evaluate_gate2

    g1p = ART / "gate1" / "gate1.json"
    g1s = json.loads(g1p.read_text())["gate1"]["status"] if g1p.exists() else "BLOCKED"
    sb = cfg["stage_b"]
    if g1s != "PASS":
        g2 = evaluate_gate2({}, sb, {}, {}, g1s)
        write_json(ART / "gate2" / "gate2.json", {"gate2": g2, "evaluated_at": now()})
        return g2
    runs, per_ex, structures = {}, {}, {}
    for task in sb["tasks"]:
        for seed in sb["seeds"]:
            sel = json.loads((probe_dir(task, seed) / "selection.json").read_text())
            structures[(task, seed)] = {"group_rvm": sel["group_rvm"]["ranks"],
                                        "conditional_newton": sel["conditional_newton"]["ranks"],
                                        "rvm_no_logdet": sel["rvm_no_logdet"]["ranks"]}
            for method in sb["methods"]:
                key = run_key(cfg.sha256, "gate2", method, task, seed, method)
                m = read_mirror(ROOT, "gate2", key) or {}
                if m.get("status") == "COMPLETED":
                    mt = m["metrics"]
                    runs[(method, task, seed)] = {"status": "COMPLETED", "loss": mt["official_validation_loss"],
                                                  "accuracy": mt["official_validation_accuracy"],
                                                  "adapter_params": mt["adapter_params"],
                                                  "selection_wall_s": mt.get("selection_wall_s") or 0.0,
                                                  "train_wall_s": mt["train_wall_s"]}
                    z = np.load(mirror_dir(ROOT, "gate2", key) / "eval_validation.npz")
                    per_ex[(method, task, seed)] = z["loss"]
                else:
                    runs[(method, task, seed)] = {"status": m.get("status", "NOT_RUN")}
    fidelity = json.loads((ROOT / "configs" / "baseline_fidelity.json").read_text())["faithful"]
    g2 = evaluate_gate2(runs, sb, fidelity, structures, g1s, per_ex if len(per_ex) == len(runs) else None)
    write_json(ART / "gate2" / "per_run_metrics.json", {f"{m}|{t}|{s}": v for (m, t, s), v in runs.items()})
    write_json(ART / "gate2" / "structures.json", {f"{t}|{s}": v for (t, s), v in structures.items()})
    if g2.get("bootstrap"):
        write_json(ART / "gate2" / "bootstrap.json", g2["bootstrap"])
    write_json(ART / "gate2" / "fidelity.json", fidelity)
    write_json(ART / "gate2" / "gate2.json", {"gate2": g2, "evaluated_at": now()})
    _ = NOT_RUN
    return g2


# ============================================================================ freeze + gates.json
def write_freeze(cfg: Config) -> dict:
    p = REPORTS / "protocol_freeze.json"
    if p.exists():
        cur = json.loads(p.read_text())
        if cur["config_sha256"] != cfg.sha256 or cur["preregistration_md_sha256"] != sha256_file(PREREG_MD):
            raise RuntimeError("protocol changed after freeze; refusing to overwrite protocol_freeze.json")
        return cur
    if git_dirty():
        raise RuntimeError("commit the implementation before freezing (tracked files are dirty)")
    import importlib.metadata as md

    out = {"frozen_at": now(), "git_sha": git_sha(), "config_path": "configs/preregistered.yaml",
           "config_sha256": cfg.sha256, "preregistration_md_sha256": sha256_file(PREREG_MD),
           "methods_md_sha256": sha256_file(ROOT / "docs" / "METHODS.md"),
           "clarifications_md_sha256": sha256_file(ROOT / "docs" / "PROTOCOL_CLARIFICATIONS.md"),
           "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
           "model": {"id": cfg["backbone"], "revision": MODEL_REVISION},
           "dataset": {"id": cfg["dataset_repository"], "revision": DATASET_REVISION},
           "split_manifests": split_hashes(cfg),
           "oplora_source": {"repo": "https://github.com/cjpark-ai/DPLoRA",
                             "sha": "6666d8bdb37ebd153a8916b0464c8a3878976aae", "license": "MIT"},
           "packages": {p_: md.version(p_) for p_ in ("torch", "transformers", "peft", "datasets", "mlflow", "numpy",
                                                      "scipy", "pandas")},
           "gate0_file_sha256": sha256_file(ART / "gate0" / "gate0.json") if (ART / "gate0" / "gate0.json").exists()
           else None}
    write_json(p, out)
    return out


def write_gates(cfg: Config) -> dict:
    from .gates import assemble

    g0 = json.loads((ART / "gate0" / "gate0.json").read_text()) if (ART / "gate0" / "gate0.json").exists() else \
        {"status": "BLOCKED", "criteria": []}
    g1p = ART / "gate1" / "gate1.json"
    g1 = json.loads(g1p.read_text())["gate1"] if g1p.exists() else {"status": "BLOCKED", "criteria": []}
    g2p = ART / "gate2" / "gate2.json"
    g2 = json.loads(g2p.read_text())["gate2"] if g2p.exists() else eval_gate2(cfg)
    meta = {"generated_at": now(), "git_sha": git_sha(), "config_sha256": cfg.sha256,
            "thresholds_source": "configs/preregistered.yaml",
            "evaluator": "rvm_lora.gates (mechanical; report text cannot override)"}
    out = assemble(g0, g1, g2, meta)
    write_json(REPORTS / "gates.json", out)
    return out
