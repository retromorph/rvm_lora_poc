"""Phase 2 audit: classification-head warm start (QNLI, seed 42).

Read-only with respect to the original run. Writes only to the output directory given on the command line.
Uses only head / direction / evidence rows (training pool) for any new label read; the audit rows are used
only to re-check that the stored head-only audit logits are reproduced bit-for-bit by the stored checkpoint.

Checks:
  1. pretrained checkpoint identity, classifier-head initialisation, backbone identical across seeds
  2. stored checkpoint: keys, dtype, change vs. the seed-42 initial head, backbone untouched by load_head
  3. exact reproduction of the stored checkpoint by re-running the repository's warm_start_head with
     instrumentation (per-step loss/accuracy, gradient norms, update norms, trainable set, optimiser groups,
     train/eval mode, backbone gradients)
  4. save -> reload logit identity; stored head-only audit logits reproduced exactly; head digest used by
     every Gate-1 run equals the digest of the stored checkpoint
  5. label encoding and (question, sentence) tokenisation
  6. head-only loss/accuracy in eval and train mode on head / direction / evidence rows
  7. EXPLORATORY diagnostic only: L2 logistic regression on frozen eval-mode <s> features (head rows only
     for fitting; direction+evidence rows for scoring) to bound how much linear signal the frozen
     features carry. Not a protocol change and not used anywhere in the gates.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

from rvm_lora import data as D  # noqa: E402
from rvm_lora.config import MODEL_REVISION, load_config, sha256_file  # noqa: E402
from rvm_lora.modeling import (  # noqa: E402
    HEAD_PREFIX, batches, configure_precision, evaluate, head_digest, head_state, load_backbone, load_head,
    save_head, warm_start_head,
)

cfg = load_config(REPO / "configs/preregistered.yaml")
SEED = 42
CKPT = REPO / "checkpoints/head_qnli_seed42.safetensors"
RECORDED_RUN_DIGEST = json.loads(next((REPO / "artifacts/runs/gate1").glob("*/result.json")).read_text())["head_digest"]
res: dict = {"label": "Phase-2 head audit (read-only w.r.t. original artifacts)"}
configure_precision()


def tensor_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def backbone_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(HEAD_PREFIX)}


def enc(part):
    return D.load_encoded(REPO / "data" / f"qnli_{part}.npz")


# ------------------------------------------------------------------ 1. pretrained checkpoint and head init
m42 = load_backbone(SEED)
c = m42.config
res["model_config"] = {"name_or_path": c._name_or_path, "revision_pinned": MODEL_REVISION,
                       "architectures": c.architectures, "num_labels": c.num_labels,
                       "hidden_dropout_prob": c.hidden_dropout_prob,
                       "attention_probs_dropout_prob": c.attention_probs_dropout_prob,
                       "classifier_dropout": c.classifier_dropout, "initializer_range": c.initializer_range,
                       "problem_type": c.problem_type, "class": type(m42).__name__,
                       "head_class": type(m42.classifier).__name__}
init_head = {k: v.detach().cpu().clone() for k, v in head_state(m42).items()}
res["init_head_stats"] = {k: {"shape": list(v.shape), "mean": float(v.mean()), "std": float(v.std())}
                          for k, v in init_head.items()}
bb42 = tensor_hash(backbone_state(m42))
m7 = load_backbone(7)
res["backbone_hash_seed42"] = bb42
res["backbone_identical_across_seeds"] = bb42 == tensor_hash(backbone_state(m7))
res["head_init_differs_across_seeds"] = head_digest(m42) != head_digest(m7)
# the backbone equals the pinned pretrained weights: compare against the raw safetensors of the hub snapshot
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    f = hf_hub_download("FacebookAI/roberta-base", "model.safetensors", revision=MODEL_REVISION)
    raw = load_file(f)
    sd = m42.state_dict()
    matched = mism = 0
    maxdiff = 0.0
    for k, v in raw.items():
        kk = k if k in sd else ("roberta." + k if "roberta." + k in sd else None)
        if kk is None:
            continue
        matched += 1
        dd = float((sd[kk].float().cpu() - v.float()).abs().max())
        maxdiff = max(maxdiff, dd)
        mism += dd != 0
    res["backbone_vs_hub_safetensors"] = {"file_sha256": sha256_file(f), "tensors_compared": matched,
                                          "tensors_differing": mism, "max_abs_diff": maxdiff,
                                          "hub_tensors_total": len(raw)}
except Exception as e:  # noqa: BLE001
    res["backbone_vs_hub_safetensors"] = {"error": repr(e)}
del m7

# ------------------------------------------------------------------ 2. stored checkpoint
from safetensors.torch import load_file as lf  # noqa: E402

ck = lf(str(CKPT))
res["checkpoint"] = {"sha256": sha256_file(CKPT), "keys": sorted(ck), "dtypes": sorted({str(v.dtype) for v in ck.values()}),
                     "shapes": {k: list(v.shape) for k, v in ck.items()}}
res["checkpoint_change_vs_seed42_init"] = {
    k: {"init_norm": float(init_head[k].norm()), "delta_norm": float((ck[k].float() - init_head[k].float()).norm()),
        "rel_change": float((ck[k].float() - init_head[k].float()).norm() / (init_head[k].norm() + 1e-12))}
    for k in ck}
sha = load_head(m42, CKPT)
res["load_head_returned_sha"] = sha
res["head_equal_to_checkpoint_after_load"] = all(torch.equal(head_state(m42)[k].cpu(), ck[k]) for k in ck)
res["backbone_unchanged_by_load_head"] = tensor_hash(backbone_state(m42)) == bb42
res["digest_after_load"] = head_digest(m42)
res["recorded_gate1_run_head_digest"] = RECORDED_RUN_DIGEST
res["gate1_runs_used_this_checkpoint"] = head_digest(m42) == RECORDED_RUN_DIGEST
fresh_digest = head_digest(load_backbone(SEED))
res["seed42_random_init_digest"] = fresh_digest
res["gate1_runs_did_not_use_random_init_head"] = RECORDED_RUN_DIGEST != fresh_digest

# ------------------------------------------------------------------ 4a. stored audit logits reproduced exactly
audit = enc("audit")
ev = evaluate(m42, audit)
stored = np.load(REPO / "artifacts/gate1/head_only_audit.npz")
res["head_only_audit_logits_reproduced"] = {
    "max_abs_diff": float(np.abs(ev["logits"] - stored["logits"]).max()),
    "bitwise_equal": bool(np.array_equal(ev["logits"], stored["logits"])), "loss": ev["loss"], "acc": ev["accuracy"]}
# probe features were computed with this head: z stored in features.npz equals eval logit difference on evidence rows
feats = np.load(REPO / "artifacts/probes/qnli_s42/features.npz")
evid = enc("evidence")
ev_e = evaluate(m42, evid)
zf = ev_e["logits"][:, 1] - ev_e["logits"][:, 0]
res["probe_used_same_head"] = {"max_abs_diff_z_evidence": float(np.abs(zf - feats["z"]).max())}

# ------------------------------------------------------------------ 5. labels and tokenisation
ds = D.load_glue("qnli", "train")
res["label_names"] = ds.features["label"].names
tok = D.get_tokenizer()
head_e = enc("head")
pos = {int(i): k for k, i in enumerate(ds["idx"])}
ex = []
for r in range(3):
    i = int(head_e.idx[r])
    row = ds[pos[i]]
    ids = head_e.input_ids[r][head_e.attention_mask[r] == 1]
    ex.append({"idx": i, "question": row["question"], "sentence": row["sentence"][:200], "label": row["label"],
               "encoded_label": int(head_e.labels[r]), "decoded": tok.decode(ids)[:400]})
res["tokenisation_examples"] = ex
labels_ok = all(int(ds[pos[int(i)]]["label"]) == int(l) for i, l in zip(head_e.idx[:500], head_e.labels[:500]))
res["encoded_labels_match_dataset_first500_head"] = labels_ok
# re-tokenise 200 head rows and compare
sub = ds.select([pos[int(i)] for i in head_e.idx[:200]])
re_ = D.encode("qnli", sub, tok, 128)
res["retokenisation_matches_first200"] = bool(np.array_equal(re_.input_ids, head_e.input_ids[:200]) and
                                             np.array_equal(re_.attention_mask, head_e.attention_mask[:200]))
res["fraction_truncated_at_128"] = {p: float((enc(p).attention_mask.sum(1) == 128).mean())
                                    for p in ["head", "directions", "evidence", "audit"]}
two_sep = tok.sep_token_id
res["pair_format_has_double_sep"] = bool(((head_e.input_ids[:50] == two_sep).sum(1) >= 3).all())
del ds

# ------------------------------------------------------------------ 6. head-only quality on training-pool rows
dirs_e = enc("directions")


def summarize(model, e):
    r = evaluate(model, e)
    return {"loss": r["loss"], "acc": r["accuracy"], "label_mean": float(e.labels.mean()),
            "frac_pred1": float((r["logits"][:, 1] > r["logits"][:, 0]).mean()),
            "zstd": float((r["logits"][:, 1] - r["logits"][:, 0]).std())}


res["head_only_eval_mode"] = {"head_rows": summarize(m42, head_e), "direction_rows": summarize(m42, dirs_e),
                              "evidence_rows": summarize(m42, evid)}
init_model = load_backbone(SEED)
res["random_init_head_eval_mode"] = {"head_rows": summarize(init_model, head_e), "evidence_rows": summarize(init_model, evid)}
del init_model


@torch.no_grad()
def train_mode_loss(model, e, seed=0):
    torch.manual_seed(seed)
    model.train()
    tot = 0.0
    n = 0
    for b in batches(e, np.arange(len(e)), 128, "cuda"):
        lg = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
        tot += float(torch.nn.functional.cross_entropy(lg, b["labels"], reduction="sum"))
        n += len(b["labels"])
    model.eval()
    return tot / n


res["head_only_train_mode_loss_head_rows"] = train_mode_loss(m42, head_e)

# ------------------------------------------------------------------ 3. instrumented exact re-run of warm_start_head
log = {"steps": [], "pre": None}
orig_adamw = torch.optim.AdamW


class InstrumentedAdamW(orig_adamw):
    def __init__(self, params, *a, **k):
        params = list(params)
        super().__init__(params, *a, **k)
        self._model_params = params
        log["optimizer"] = {"n_params": len(params), "numel": int(sum(p.numel() for p in params)),
                            "defaults": {kk: (vv if isinstance(vv, (int, float, bool)) else str(vv))
                                         for kk, vv in self.defaults.items()}}
        self._before = None

    def step(self, closure=None):
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in self._model_params if p.grad is not None)))
        nz = sum(p.grad is not None for p in self._model_params)
        before = [p.detach().clone() for p in self._model_params]
        out = super().step(closure)
        un = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(self._model_params, before))))
        log["steps"].append({"grad_norm": gn, "n_with_grad": nz, "update_norm": un})
        return out


torch.optim.AdamW = InstrumentedAdamW
model = load_backbone(SEED)
bb_before = tensor_hash(backbone_state(model))
mode_log = []
loss_log = []
acc_log = []
orig_forward = model.forward


def fwd(*a, **k):
    out = orig_forward(*a, **k)
    mode_log.append(model.training)
    return out


model.forward = fwd
# per-step batch loss/accuracy via a forward hook on the classifier (logits are the classifier output)
head_order = __import__("rvm_lora.modeling", fromlist=["sample_order"]).sample_order(2000, 200 * 32, SEED, 101)
batch_labels = [head_e.labels[head_order[i * 32:(i + 1) * 32]] for i in range(200)]
step_i = {"i": 0}


def clf_hook(mod, inp, out):
    lg = out.detach().float()
    yb = torch.as_tensor(batch_labels[step_i["i"]], device=lg.device)
    loss_log.append(float(torch.nn.functional.cross_entropy(lg, yb)))
    acc_log.append(float((lg.argmax(1) == yb).float().mean()))
    step_i["i"] += 1


hh = model.classifier.register_forward_hook(clf_hook)
requires = {n: p.requires_grad for n, p in model.named_parameters()}
t0 = time.perf_counter()
info = warm_start_head(model, head_e, cfg, SEED)
hh.remove()
model.forward = orig_forward
torch.optim.AdamW = orig_adamw
res["rerun_info"] = info
res["rerun_wall_s"] = time.perf_counter() - t0
res["rerun_trainable_after_return"] = [n for n, p in model.named_parameters() if p.requires_grad]
res["rerun_model_training_flag_after_return"] = model.training
res["rerun_forward_modes"] = {"n_forwards": len(mode_log), "all_train_mode": all(mode_log)}
res["rerun_backbone_unchanged"] = tensor_hash(backbone_state(model)) == bb_before
res["rerun_backbone_grads_none"] = all(p.grad is None for n, p in model.named_parameters() if not n.startswith(HEAD_PREFIX))
res["optimizer"] = log["optimizer"]
st = log["steps"]
res["rerun_steps"] = len(st)
res["rerun_grad_norm"] = {"first": st[0]["grad_norm"], "median": float(np.median([s["grad_norm"] for s in st])),
                          "last": st[-1]["grad_norm"], "min": float(min(s["grad_norm"] for s in st))}
res["rerun_update_norm"] = {"first": st[0]["update_norm"], "median": float(np.median([s["update_norm"] for s in st])),
                            "last": st[-1]["update_norm"]}
res["rerun_all_head_params_had_grads"] = all(s["n_with_grad"] == log["optimizer"]["n_params"] for s in st)
res["rerun_batch_loss_curve_first20_last20"] = [float(np.mean(loss_log[:20])), float(np.mean(loss_log[-20:]))]
res["rerun_batch_acc_curve_first20_last20"] = [float(np.mean(acc_log[:20])), float(np.mean(acc_log[-20:]))]
res["rerun_batch_loss_by_epoch"] = [float(np.mean(loss_log[i:i + 62])) for i in range(0, 200, 62)]
res["rerun_batch_acc_by_epoch"] = [float(np.mean(acc_log[i:i + 62])) for i in range(0, 200, 62)]
rec_meta = json.loads((REPO / "artifacts/heads/qnli_s42.json").read_text())
res["recorded_meta_losses"] = [rec_meta["head_train_loss_first20"], rec_meta["head_train_loss_last20"]]
res["rerun_meta_losses_equal_recorded"] = (info["head_train_loss_first20"] == rec_meta["head_train_loss_first20"] and
                                           info["head_train_loss_last20"] == rec_meta["head_train_loss_last20"])
rh = head_state(model)
res["rerun_head_vs_recorded_checkpoint"] = {
    "bitwise_equal_all": all(torch.equal(rh[k].cpu(), ck[k]) for k in ck),
    "max_abs_diff": max(float((rh[k].cpu().float() - ck[k].float()).abs().max()) for k in ck)}
# save -> reload identity (sandbox path), logits before vs after
probe_rows = D.Encoded(head_e.idx[:256], head_e.input_ids[:256], head_e.attention_mask[:256], head_e.labels[:256])
before = evaluate(model, probe_rows)["logits"]
tmp = OUT / "rerun_head_qnli_seed42.safetensors"
sha_rerun = save_head(model, tmp)
fresh = load_backbone(SEED)
load_head(fresh, tmp)
after = evaluate(fresh, probe_rows)["logits"]
res["save_reload"] = {"rerun_ckpt_sha256": sha_rerun, "equals_recorded_sha": sha_rerun == sha256_file(CKPT),
                      "logits_bitwise_equal": bool(np.array_equal(before, after)),
                      "max_abs_logit_diff": float(np.abs(before - after).max())}
tmp.unlink()
del fresh, model

# ------------------------------------------------------------------ 7. EXPLORATORY: linear signal in frozen <s> features
@torch.no_grad()
def cls_features(model, e):
    model.eval()
    feats_ = []
    for b in batches(e, np.arange(len(e)), 128, "cuda"):
        hs = model.roberta(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        feats_.append(hs[:, 0].float().cpu())
    return torch.cat(feats_).numpy().astype(np.float64)


Xh = cls_features(m42, head_e)
Xt = np.concatenate([cls_features(m42, dirs_e), cls_features(m42, evid)])
yh, yt = head_e.labels, np.concatenate([dirs_e.labels, evid.labels])
res["cls_feature_stats"] = {"mean_norm": float(np.linalg.norm(Xh, axis=1).mean()),
                            "mean_pairwise_cosine_sample": float(np.mean(
                                (Xh[:200] / np.linalg.norm(Xh[:200], axis=1, keepdims=True)) @
                                (Xh[:200] / np.linalg.norm(Xh[:200], axis=1, keepdims=True)).T)),
                            "top3_abs_mean_dims": np.argsort(-np.abs(Xh.mean(0)))[:3].tolist(),
                            "top3_abs_mean_vals": np.sort(-np.abs(Xh.mean(0)))[:3].tolist()}
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lr_res = {}
    sc = StandardScaler().fit(Xh)
    for C in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xh), yh)
        lr_res[str(C)] = {"train_acc_head_rows": float(clf.score(sc.transform(Xh), yh)),
                          "acc_direction_evidence_rows": float(clf.score(sc.transform(Xt), yt))}
    res["EXPLORATORY_logreg_frozen_cls_features"] = lr_res
except Exception as e:  # noqa: BLE001
    res["EXPLORATORY_logreg_frozen_cls_features"] = {"error": repr(e)}

(OUT / "phase2_head.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=1, default=str))
