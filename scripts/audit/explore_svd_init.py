"""EXPLORATORY (not part of any gate, not a protocol amendment): SVD-aligned zero-update LoRA initialisation.

Question: does the evidence/LoRA tangent mismatch (reports/CORRECTNESS_AUDIT.md §4) explain why the structural
scores predict 600-step utility poorly? For each of the 20 frozen Gate-1 architectures, train the identical
recipe (same frozen v2 head, data order, seed 42, 600 steps, rsLoRA alpha 4, AdamW 2e-4) with ONE change:
lora_A of module m with rank r = the first r gradient-SVD right singular vectors v_m1..v_mr of the v2 probe,
each row rescaled to the norm of the protocol's deterministic Kaiming row it replaces; lora_B = 0 (zero initial
update, asserted). The initial trainable tangent {X A} then contains every scored atom u_mj v_mj^T.
Outputs: <out>/svdinit_runs.json, MLflow experiment rvm-lora/exploratory-svd-init. Audit labels are read only to
compute U for this exploratory comparison; the protocol's random-init results are untouched.

argv: <repo> <namespace with head/probe, e.g. v2> <out dir>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO, NS, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from rvm_lora import data as D  # noqa: E402
from rvm_lora.adapters import assert_exact_layout, build_peft, canonical_name  # noqa: E402
from rvm_lora.config import load_config, sha256_file  # noqa: E402
from rvm_lora.modeling import configure_precision, evaluate, load_backbone, load_head  # noqa: E402
from rvm_lora.train import train_with_oom_fallback  # noqa: E402

cfg = load_config(REPO / "configs/preregistered.yaml")
SEED = cfg["stage_a"]["run_seed"]
head = REPO / "checkpoints" / NS / "head_qnli_seed42.safetensors"
pdir = REPO / "artifacts" / NS / "probes" / "qnli_s42"
atoms = torch.load(pdir / "atoms.pt")
head_audit = json.loads((REPO / "artifacts" / NS / "gate1" / "head_only_audit.json").read_text())
train_enc = D.load_encoded(REPO / "data" / "qnli_adapter_train.npz")
audit = D.load_encoded(REPO / "data" / "qnli_audit.npz")
configure_precision()

from mlflow.tracking import MlflowClient  # noqa: E402

client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
exp = client.get_experiment_by_name("rvm-lora/exploratory-svd-init")
exp_id = exp.experiment_id if exp else client.create_experiment("rvm-lora/exploratory-svd-init")

z0 = None
results = []
for a in cfg.architectures():
    def build(a=a):
        model = load_backbone(SEED)
        load_head(model, head)
        pm = build_peft(model, a.ranks, cfg, SEED)
        with torch.no_grad():
            for name, mod in pm.named_modules():
                if not (hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {})):
                    continue
                m = cfg.modules.index(canonical_name(name))
                A = mod.lora_A["default"].weight
                r = A.shape[0]
                norms = A.norm(dim=1, keepdim=True)
                V = atoms["V"][m, :, :r].T.to(A.device, A.dtype)  # r x in, unit rows
                A.copy_(V * norms)
                assert float(mod.lora_B["default"].weight.abs().max()) == 0.0
        return pm, {"ranks": a.ranks}

    pm, _ = build()
    probe_rows = D.Encoded(audit.idx[:32], audit.input_ids[:32], audit.attention_mask[:32], audit.labels[:32])
    base = load_backbone(SEED)
    load_head(base, head)
    zero_update = bool(np.array_equal(evaluate(pm, probe_rows)["logits"], evaluate(base, probe_rows)["logits"]))
    del pm, base
    run = client.create_run(exp_id, run_name=f"svdinit-{a.arch_id}",
                            tags={"label": "EXPLORATORY_NOT_A_GATE", "arch_id": a.arch_id, "init": "svd_aligned_A_zero_B",
                                  "head_checkpoint_sha256": sha256_file(head), "namespace": NS})
    t0 = time.perf_counter()
    peft_model, meta, tr = train_with_oom_fallback(build, train_enc, cfg, SEED)
    acct = assert_exact_layout(peft_model, a.ranks, cfg.modules, cfg.rank_unit_params)
    ev = evaluate(peft_model, audit)
    U = head_audit["audit_loss"] - ev["loss"]
    rec = {"arch_id": a.arch_id, "family": a.family, "parameters": a.parameters, "audit_loss": ev["loss"],
           "audit_accuracy": ev["accuracy"], "U_svdinit": U, "train_loss_first": tr["train_loss_first"],
           "train_loss_final": tr["train_loss_final"], "train_wall_s": tr["train_wall_s"],
           "zero_initial_update": zero_update, "adapter_params": acct["adapter_params"], "mlflow_run_id": run.info.run_id,
           "microbatch": tr["microbatch"], "head_digest": tr["head_digest"]}
    for k in ("audit_loss", "audit_accuracy", "U_svdinit", "train_loss_final", "train_wall_s", "adapter_params"):
        client.log_metric(run.info.run_id, k, float(rec[k]))
    client.set_terminated(run.info.run_id, "FINISHED")
    results.append(rec)
    print(f"{a.arch_id:14s} U_svd={U:+.5f} acc={ev['accuracy']:.4f} train={tr['train_wall_s']:.1f}s "
          f"zero_update={zero_update}", flush=True)
    del peft_model
    torch.cuda.empty_cache()
    (OUT / "svdinit_runs.json").write_text(json.dumps({"label": "EXPLORATORY_NOT_A_GATE", "namespace": NS,
                                                       "head_only_audit_loss": head_audit["audit_loss"],
                                                       "runs": results}, indent=2))
