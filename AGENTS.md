# AGENTS.md — RVM-LoRA pilot

## Mission and authority

Build, validate and **execute** a reproducible proof-of-concept of Group-RVM-LoRA on one RTX A4000 16 GB GPU, and produce an auditable final Go/No-Go report. Follow `docs/PREREGISTRATION.md` and `configs/preregistered.yaml`; they control scientific decisions. The user wants an *unbiased negative result* if RVM does not help. Do not optimize for a positive verdict.

Read in this order: `README.md`, `docs/PREREGISTRATION.md`, `configs/preregistered.yaml`, `docs/METHODS.md`, `docs/ENGINEERING.md`, `docs/REMOTE_RUNBOOK.md`, `docs/REPORT_TEMPLATE.md`, `docs/REFERENCES.md`.

**Authority order:** explicit user instruction > this file > frozen preregistration/YAML (interpret together) > method/engineering docs > implementation choices. If YAML and protocol contradict, STOP the conflicting stage, record the discrepancy, and resolve *before* seeing reserved labels/results. Do not silently choose an advantageous interpretation.

## Non-negotiable science controls

1. Never use QNLI/SST-2 official validation results for design, tuning, preflight, stopping, or model selection before Gate 1 is fixed; once Gate 2 begins, use it only for the predeclared final evaluation. Use the internal audit set only for scoring the 20 Gate-1 architectures, not for selecting priors, thresholds or training hyperparameters.
2. Create deterministic, disjoint data-ID manifests and hashes; keep a fixed dataset split seed. Run-seed variation must not alter the example identities.
3. Share one classifier-head checkpoint per task/seed across selectors. Freeze the backbone and classifier head during adapter training. The eligible adapter matrices are precisely the 24 attention `query`/`value` projections.
4. Compare identical parameter budgets, optimizer steps, eligible modules, training data, evaluation pipeline and seeds. Report any EVA-native initialization advantage/disadvantage separately. Track exact trainable parameter counts by assertion, not merely nominal ranks.
5. Implement RVM in **float64** on CPU or stable device, with Cholesky solves and log-determinants, no explicit matrix inverses. Preserve `r=0` by **omitting modules**, not by passing zero rank to PEFT.
6. The optimized Gaussian single-atom evidence is monotonic in the conditional whitened score above the inclusion threshold. **Never claim improved single-atom ranking versus that score**. The falsifiable claims concern group interactions, sparsity/stopping, useful architecture selection, and compute.
7. Never call approximate Gauss–Newton pseudo-evidence the exact nonlinear Transformer evidence; never call greedy selection globally optimal.
8. No experimental threshold, baseline, metric, selected architecture family, seed or cost accounting can be revised in-place after the corresponding holdout results are viewed. Amendments must be committed with a timestamp, reason, and clearly labeled exploratory status; they do not count toward the pre-registered Go verdict.
9. **Conditional progression:** Gate 0 must pass before Gate 1; Gate 1 must pass before Gate 2. If a gate fails, stop further GPU experiments, compile the report and mark unrun experiments `NOT_RUN_GATE_FAILURE`.
10. Do not hide failed, interrupted, inconclusive or OOM runs. Log their status and recovery actions to MLflow and run index.

## Agent permissions and boundaries

You may create source code, tests, `pyproject.toml`, `uv.lock`, MLflow experiment names, local files, `tmux` sessions, nonprivileged directories and SSH tunnels, and install dependencies into a project-local `uv` environment. You may invoke `nvidia-smi` and other read-only system checks. You may fetch public Hugging Face datasets/models and examine cited public repositories, respecting licenses.

**Do not:** modify system firewall/UFW rules, enable UFW, start publicly exposed unauthenticated services, store SSH keys/tokens in the repository or MLflow, use `sudo` without express authorization, remove other users' processes/data, or change the remote host's global Python/CUDA installation. MLflow must bind to `127.0.0.1` and be inspected via SSH forwarding. If SSH or storage access is unavailable, implement and test what is possible and report `BLOCKED`; do not claim runs were executed.

## Engineering requirements

- Use Python 3.11 (unless unavailable, then document a supported alternative), `uv`, PyTorch, Transformers, PEFT, Datasets, NumPy/SciPy, pandas, MLflow, pytest and plotting. Lock package versions and log `uv.lock` checksum, CUDA driver, GPU name, peak allocated memory, precision and Git SHA.
- Adopt a simple `src/` package and CLI (`prepare`, `warm-head`, `features`, `oracle-a`, `select`, `train-b`, `gates`, `report`). All runs must be restartable and idempotent; re-running a completed run does not silently replace its result.
- Prefer effective batch 32 via microbatch 8 × 4 gradient accumulation; auto-reduce microbatch to 4 × 8 for OOM **without changing effective batch**, documenting any deviation. BF16 if supported; otherwise FP16 with loss scaling, otherwise FP32. Selection linear algebra stays FP64.
- Enforce one active GPU training run at a time; background tracking server and run manager can coexist. Never run a heavy GPU benchmark merely to decorate the report.
- Unit-test evidence identities, null behavior, two-class logit-Jacobian formula, PEFT exact module/rank placement, parameter counts, split separation, frozen weights, resume semantics, and gate logic. A small end-to-end smoke test must pass before full execution.
- Prefer supported APIs. If referencing EVA or OPLoRA official code, pin repository commit and record integration differences/licenses. If a prescribed baseline cannot be reproduced faithfully, report `INCONCLUSIVE` for its dependent comparison rather than quietly replacing it.

## Execution and evidence contract

Before the first audit evaluation, commit the source/protocol/config and write `reports/protocol_freeze.json` (SHA256 of preregistration YAML and protocol Markdown, Git SHA, timestamp). Preserve all run configs, data split hashes, stdout/stderr paths and MLflow links. MLflow experiment names: `rvm-lora/gate0`, `rvm-lora/gate1`, `rvm-lora/gate2`.

Every run logs: stage, task, seed, method, architecture rank JSON, nominal/actual parameters, model/dataset revisions, head checkpoint hash, probe sizes, optimizer and precision, train/validation cross-entropy, accuracy, run wall time, calibration time when applicable, peak GPU memory, artifacts and status. Use the explicit metric name `official_validation_loss` only for the **official validation** results; never label calibration/audit results as test performance.

Expected end state:

- `reports/FINAL_REPORT.md` based on the supplied template, including **all** positive/negative comparisons and limitations.
- `reports/gates.json` showing each condition, numeric measurement, cutoff, evidence link, PASS/FAIL/INCONCLUSIVE/BLOCKED, and overall verdict.
- `reports/run_index.csv`, `reports/architecture_scores.csv`, `reports/figures/*.png`, `reports/protocol_freeze.json`, a version/environment manifest, and `reports/review_bundle.zip` excluding credentials, raw datasets and large checkpoints.
- Git commit of code and final docs, plus exact commands to reproduce all stages from a fresh checkout.

**End-of-task reporting:** Say precisely which stages were run and which were not. Do not say “all experiments passed” unless the frozen gate evaluator computes that verdict from real logged results.
