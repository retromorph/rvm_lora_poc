# Engineering design and acceptance checklist

## 1. Supported system

Single RTX A4000 16 GB, Linux, SSH, Python 3.11, `uv`, compatible NVIDIA driver/PyTorch CUDA wheels. Prefer modern stable `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `scipy`, `numpy`, `pandas`, `mlflow`, `pytest`, `ruff`, `matplotlib`, `seaborn` optional (the plots can use Matplotlib). Pin versions in `uv.lock`. For CUDA choose an official PyTorch wheel compatible with the observed driver; do not guess CUDA from the board model. Start with native PyTorch/PEFT, no FSDP/DeepSpeed/quantization/FlashAttention for RoBERTa-base.

Use BF16 only if `torch.cuda.is_bf16_supported()` and a real forward/backward smoke test pass; else FP16 with scaler, else FP32. RVM calculations always in `float64`. Reserve GPU VRAM headroom; use microbatch 8, gradient accumulation 4 to reach effective batch 32; documented OOM fallback is 4 × 8. Do **not** change effective batch or 600 optimizer steps. Gradient checkpointing is allowed if needed and must be identical across compared methods.

## 2. Suggested source layout

```text
src/rvm_lora/
  cli.py                # argparse or Typer entry point; typed configs
  data.py               # frozen GLUE split IDs, SHA, tokenization
  head.py               # warm-start/frozen-head checkpoint
  directions.py         # projection hooks, loss gradient-SVD
  features.py           # per-example d(logit1-logit0)/ds; Phi,t,K,b
  evidence.py           # float64 marginal evidence and Type-II ARD
  selectors.py          # group RVM, independent/conditional scorers
  nulls.py              # null threshold calibration and fresh trials
  adapters.py           # exact PEFT module names/ranks, counters
  baselines.py          # EVA/OPLoRA integration, budget checks
  train.py              # training/eval, GPU resource telemetry
  mlflow_utils.py       # runs/artifacts, robust retry and resume
  gates.py              # deterministic frozen gate evaluation
  report.py             # outputs tables, graphs and review bundle
tests/
  test_evidence.py
  test_features.py
  test_ranks.py
  test_splits.py
  test_nulls.py
  test_gates.py
  test_end_to_end_smoke.py
scripts/
  bootstrap_remote.sh
  start_mlflow.sh
  run_remote.sh
  export_review_bundle.sh
configs/
  preregistered.yaml
reports/
  FINAL_REPORT.md
  gates.json
  run_index.csv
  architecture_scores.csv
  figures/
```

Implement `python -m rvm_lora.cli` with subcommands `env-check`, `prepare`, `warm-head`, `probe`, `oracle-a`, `eval-gate1`, `train-b`, `eval-gate2`, `report`, `all`, `status`. `all` must stop at failed gates and be resumable. Include `--dry-run` printing the entire run plan, estimated parameter counts, expected artifacts and SSH/MLflow prerequisites **without reading reserved labels**.

## 3. Frozen data and model contract

Use `load_dataset("nyu-mll/glue", "qnli")` and similarly for `sst2`. Resolve and pin a dataset revision if supported; save original row `idx`, task, split name, full example ID lists and SHA256. The 10,000/1,500 sampling is from the official training split only; GLUE validation is an independent, later split. Exact 2,000+512+512+6,976 partition among the 10,000. Direction/evidence rows may appear **later** in LoRA adapter training (this is intentional); audit rows never may. Validate absence of duplicates and splits intersections.

RoBERTa QNLI input `(question, sentence)`; SST-2 `sentence`. Head trainable only in warm-start, all adapter baseline runs load identical task/seed checkpoint and freeze classifier head permanently. Ensure warm-start results are adequate to produce nontrivial gradients, but **do not change the protocol using audit/official validation**. If there is a fundamental pipeline failure, return `INCONCLUSIVE`.

## 4. Objective and feature implementation tests

- Probe in `eval()` with all stochastic dropout inactive; compute binary **logit difference** `logits[:,1]-logits[:,0]`, not the sum of probabilities or loss gradient directly. Use `torch.autograd.grad` on captured projection outputs and validate the directional derivative formula with JVP/finite differences.
- Frozen base projections ordinarily have `requires_grad=False`. Ensure hooks capture correct gradients despite this; an all-zero or missing Jacobian is a **hard error**. Do not temporarily unfreeze the backbone during training.
- SVD of mean-loss gradient with deterministic signs, unit-norm u/v and decreasing singular values. Different layers have distinct full module names; match exact projection shape 768×768.
- Compute the Gaussian local surrogate `Phi=sqrt(p*(1-p))*J`, `t=(y-p)/sqrt(p*(1-p))` with clipped p. Save float64 `K`/`b`, feature metadata and calibration hashes; avoid storing logits/labels in public artifacts.
- Test `E(S)` against `C=I+Phi A^-1 Phi.T` on toy matrices, including one/two module groups and alpha at numerical limits. Test no-logdet and zero-gradient cases. Cholesky-with-jitter must log jitter; excessive jitter is test failure.

## 5. Exact adapters and cost assertions

PEFT `target_modules` accepts suffix matching for lists; do not pass bare `query`, `value` for selective placement. Use exact full names or verified regex with unit tests asserting precisely the requested modules. For nonzero ranks, pass a correct `rank_pattern` with anchored names; for rank zero **exclude module entirely**. Assert the number of trainable tensors, precise parameter total, no LoRA weights in unselected layers, and an unchanged classifier head after one training step.

Expected: `query+value`, rank2 in a single layer: `2*2*(768+768)=6144`. Uniform rank1 for 24 matrices: `24*1536=36864`. Uniform rank2: `73728`. Count **only** LoRA trainable tensors for adapter budgets; log all trainable tensors separately to detect unintended head adaptation. All selected architectures must have `max(r_m)<=8` in Gate 2.

## 6. Logging, provenance, idempotence

Use a shared root `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. MLflow run tags: `gate`, `method`, `task`, `seed`, `git_sha`, `protocol_sha256`, `data_split_sha256`, `model_revision`, `head_checkpoint_sha256`, `precision`, `hostname`, `gpu_name`, `status`. Metrics: `train_loss`, `audit_loss` (only Gate 1), `official_validation_loss` (only Gate 2), `official_validation_accuracy`, `train_wall_s`, `selection_wall_s`, `total_wall_s`, `peak_gpu_memory_mb`, `adapter_params`, `rank_sum`. Artifacts: full run config, rank JSON, selection trace, numerical issue log, checksums, aggregate metrics.

Every run ID is deterministically keyed by `(protocol SHA, gate, method, task, seed, architecture-id)`. Completed runs with matching artifact hashes should be reused on restart; mismatched completed runs require an explicit invalidation record. Exclude raw datasets and credentials from MLflow artifacts. A run interrupted by SSH disconnection must keep its partial trace and be safely resumed or rerun with a new attempt ID; never mix two attempts' metrics.

Make a **local CSV/JSON mirror** of final summary metrics so they are reviewable without a running MLflow server. Capture exact commands and raw task logging separately from final metrics.

## 7. Report automation and plots

Produce:

1. `architecture_scores.csv`: 20 rows × all four score methods and actual audit utility, cost, rank pattern, eligibility for comparisons; pairwise concordances and top-3 aggregate table.
2. `run_index.csv`: all 44 **planned** rows, with actual status, run ID, seed, architecture/method and paths; gate-failed later rows marked `NOT_RUN_GATE_FAILURE`.
3. `gates.json`: each metric, exact condition, value, cutoff, comparator, source artifact and independent status; one overall decision.
4. `figures/gate1_scatter*.png`: predicted scores vs realized utility, stratified by parameter cost, annotate Spearman and pairwise concordance.
5. `figures/gate2_quality_vs_params.png`, `figures/allocations.png`, `figures/overhead.png` if Gate 2 executed.
6. Human-readable `FINAL_REPORT.md` and a small `review_bundle.zip` that excludes checkpoints/MLflow DB/datasets and contains all final decision evidence and hashes.

Compute Go/No-Go mechanically from YAML; the report writer **may describe**, but must not override, the gate evaluator result. Output `BLOCKED` or `INCONCLUSIVE` when necessary.

## 8. Required test commands

```bash
uv sync --frozen
uv run pytest -q
uv run python -m rvm_lora.cli env-check
uv run python -m rvm_lora.cli all --dry-run
# After Gate 0: from the remote tmux session
uv run python -m rvm_lora.cli all --config configs/preregistered.yaml
uv run python -m rvm_lora.cli report --config configs/preregistered.yaml
```

The coding agent may adjust the CLI ergonomics, but must document working equivalents and make the entire procedure reproducible from a fresh checkout.
