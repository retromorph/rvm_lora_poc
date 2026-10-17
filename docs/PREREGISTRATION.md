# Pre-registration: RVM-LoRA Go/No-Go pilot

**Version:** 1.0. **Design date:** 2026-10-17. **Status:** FROZEN BEFORE OUTCOME DATA. Machine-readable counterpart: [`../configs/preregistered.yaml`](../configs/preregistered.yaml).

## 0. Scientific decision

**H0 (practical):** Conditional group-level RVM pseudo-evidence provides no meaningful benefit over cheap conditional Newton/Fisher selection for the final quality–parameter–calibration-cost trade-off. **H1 (pilot):** RVM scores predict actually useful architectures better and lead to a competitive final structure with acceptable overhead. Passing this pilot authorizes a larger replication, **not** a publication claim.

A *negative* answer is scientifically useful. No open-ended tuning of a failed RVM is permitted under this preregistration. Preserve failures, negative scores, empty selected architectures, and baseline failures.

### Status semantics

- `PASS`: all required measurable conditions are satisfied.
- `FAIL`: a valid measurement violates at least one required condition.
- `INCONCLUSIVE`: required measurement unavailable/underpowered, invalid data isolation, or irreproducible baseline; cannot declare GO.
- `BLOCKED`: infrastructure or permission prevents completion; cannot declare GO.
- `NOT_RUN_GATE_FAILURE`: later stage intentionally skipped because an earlier gate failed.

Only `Gate 0 PASS` **and** `Gate 1 PASS` **and** `Gate 2 PASS` yields project decision `GO_TO_3B_REPLICATION`. Otherwise `NO_GO`, `INCONCLUSIVE` or `BLOCKED` with evidence.

## 1. Fixed environment, samples and budgets

- Backbone: `FacebookAI/roberta-base` sequence classifier (12 encoder blocks, hidden size 768). Binary tasks QNLI and SST-2 from `nyu-mll/glue`. Input tokens max 128. QNLI tokenize `(question, sentence)`; SST-2 `(sentence)`.
- Eligible matrices: exactly `roberta.encoder.layer.{0..11}.attention.self.{query,value}`, i.e. 24 projections. Head and all backbone weights are frozen during LoRA training. All 24 candidates are 768×768; one rank = 1,536 parameters.
- Fixed split seed: `1729`, applied independently to each task to randomly shuffle official **training** example IDs. Take first 10,000 for training pool, next 1,500 for internal audit; save exact IDs and SHA256 checksums. From training pool allocate 2,000 head warm-start; 512 disjoint direction generation; 512 disjoint evidence; remaining 6,976 other training examples. Final adapter-training pool is all 8,000 examples not in head warm-start, including the 512+512 calibration examples. No audit sample appears anywhere in training, calibration, or hyperparameter fitting.
- Warm start per task/seed: frozen RoBERTa plus trainable classification head only; 200 AdamW steps on head split, learning rate 1e-3, batch 32. Save checkpoint + hash, then freeze head for all subsequent methods. The head is not counted as LoRA trainable parameters. Do not refit head during adapter training.
- LoRA training: AdamW (LR `2e-4`, weight decay `0.01`), 600 optimizer steps, 36 warmup steps then linear decay, effective batch 32, microbatch 8 × gradient accumulation 4 (fallback 4 × 8 on A4000 OOM), sequence 128, native backbone dropout, LoRA dropout 0, `use_rslora=True`, `lora_alpha=4`, deterministic seed per run. Use one identical zero-update PEFT initialization for all non-EVA methods. No checkpoint-selection by audit or official validation: evaluate the fixed final checkpoint.
- Numerical selection: evaluate probes with `model.eval()` (dropout OFF); compute classifier binary-logit-difference Jacobians; logit and curvature calculations in FP32; RVM/Gram statistics and solves in FP64. Clip sigmoid probabilities to [1e-4,1-1e-4].
- Probe split: 512 direction examples for mean-loss gradient SVD, and 512 disjoint evidence examples for feature matrix + score estimation. Methods may use at most these 1,024 calibration inputs; the precise samples and pass counts are logged. RVM group ranks are prefixes of each module's first eight gradient-SVD directions, with SVD sign convention fixed in code.
- Primary outcome: binary cross-entropy (mean **nats per example**, float64 aggregation); secondary accuracy, exact trainable LoRA parameters, selected ranks, selection wall time, total wall time, peak CUDA memory.

If the pretrained dataset/model revision changes after freezing, record and justify this as an invalidating protocol deviation rather than silently using a new revision.

## 2. Gate 0 — implementation correctness

**Must pass before either held-out set is read:**

1. Gaussian linear-regression oracle: evidence increments from sequential Cholesky/Gram updates agree with direct `slogdet` and stable linear-solve computation to `rtol <=1e-5` (and `atol <=1e-7` near zero).
2. Single rank-one analytic maximizer agrees with numerical scalar optimization, including exclusion at `q²<=h`. Verify optimized RVM ordering equals `q²/h` on eligible equal-cost singleton candidates: **a negative novelty control**.
3. Confirm group evidence with block rank-one or rank-two updates agrees with direct evidence differences; handle collinearity, zero gradients, all-infinite-precision and no-improvement cases.
4. Jacobian identity for `query`/`value` rank-one updates matches direct finite differences or `torch.func.jvp` on a tiny sequence-classification case (relative error <1e-3 away from zero). Probe in `eval()` mode, disable dropout and keep epsilon/precision suitable for this test.
5. Confirm all model weights except requested adapters are frozen; classifier head stays exactly unchanged; PEFT selected module names/ranks are exact, and no accidental suffix matches/`rank=0` instances occur. Assert `(q2+v2)=6144`, `(q4+v4)=12288`, `B24=36864`, `B48=73728`.
6. Verify all splits disjoint, stable under different model seeds and never include official validation. Test idempotent/resumable MLflow run IDs, gate evaluator from synthetic metrics, and two-step CUDA smoke test within 16 GB.
7. Lock `uv.lock`, Git SHA, YAML/MD SHA256, data/model revisions and CUDA/PyTorch versions into `reports/protocol_freeze.json` before the first *internal audit* prediction is inspected.

If a true implementation test fails, fix without peeking at held-out data; repeat Gate 0. Log the failures and fixes.

## 3. Gate 1 — QNLI architecture oracle (Experiment A)

Run at seed `42`; share the one QNLI warm-started classifier head across 20 architecture training runs. The fixed set is:

- **12** single-layer configurations: for each layer `l=0..11`, rank 2 in `query` and rank 2 in `value` of layer `l` only. Each 6,144 parameters.
- **4** single-layer configurations: layer in `{0,3,7,11}`, rank 4 in `query` and rank 4 in `value`. Each 12,288 parameters.
- **4** two-layer configurations: layer pairs `{(0,1),(5,6),(10,11),(2,9)}`, rank 2 in `query` and `value` of **each** layer. Each 12,288 parameters.

For architecture A, actual oracle utility is

`U(A) = audit_CE(frozen_head_only) - audit_CE(fixed_600_step_LoRA_A)`.

Precompute the prediction score for all 20 architectures **before any audit labels are read**. Baseline scores: naive group-gradient magnitude; independent ridge/Fisher-normalized group score; conditional Newton/regularized quadratic group score; proposed optimized RVM group log evidence. Evaluate each score **within** the 6,144 and 12,288 parameter strata; never obtain a spurious correlation by mixing budgets.

**Pairwise concordance:** Over cost-matched pairs with `|U(A)-U(B)|>0.005` nats/example, the fraction whose predicted ordering matches actual utility. If <30 informative pairs across both strata, criterion = `INCONCLUSIVE` (not a positive result). Compare RVM with the **best** of the three non-Bayesian competitors, selected by its measured concordance in this diagnostic experiment.

**Top-3 realized utility test:** For each stratum, take the three configurations having the largest *precomputed score* for RVM, and analogously for each competitor. Across the 6 selected architectures (3 per stratum), average **actual** `U(A)`; duplicates across selectors are reused, not retrained. RVM must beat the best non-Bayesian selector's six-choice realized utility by ≥0.005 nats/example. No favorable tie-breaking: ties sorted by architecture ID.

**Conditional interaction diagnostic:** For the four fixed two-layer pairs `(i,j)`, evaluate the observed **non-additivity** `I_oracle(i,j) = U(A_ij) - U(A_i) - U(A_j)` (all U measured against the same frozen-head audit loss). Compare its sign to `I_evidence(i,j) = E(A_ij) - E(A_i) - E(A_j)`, with E optimized for each architecture. An effect is informative if `|I_oracle| > 0.005` nats/example. Since evidence is in log units over 512 calibration examples, compare **signs, not raw scale**. For ≥3 informative effects, ≥3 correct signs are required; if fewer than 3, label the diagnostic `NOT_EVALUABLE` and do not count it as a scientific success (but it does not independently block the gate). **Important:** optimized evidence for adding an optional module is never negative when `alpha -> infinity` recovers the old model. Therefore, using the sign of the raw *conditional evidence increment* to predict a negative retraining gain would be invalid; non-additivity is the intended diagnostic.

**Automatic stopping null check:** Fit the fixed feature dictionary on real training/calibration data. Generate 100 pseudo-responses sampled `N(0,I)` (seed 1931), calibrate an initial-step maximum-gain threshold at the empirical 95th percentile. With that threshold and 100 **fresh** pseudo-responses (seed 1932), run the same selection algorithm from empty support; at most 10 of 100 may activate any group. This is an empirical *complete-null* diagnostic; do not claim exact sequential false-discovery control.

**Gate 1 PASS iff:**

1. ≥30 informative cost-matched pairs, RVM pairwise concordance ≥0.65, and RVM minus the *best non-Bayesian* concordance ≥0.05.
2. RVM six-choice top-3 mean actual utility minus best competitor ≥0.005 nats/example.
3. ≤10/100 fresh null replicates select any group.
4. If ≥3 informative interaction effects exist, at least 3 sign predictions are correct.

Otherwise: `FAIL` when genuinely contradicted, `INCONCLUSIVE` if necessary comparisons lack enough informative pairs, or `BLOCKED` for infrastructure. Stop before Gate 2 unless `PASS`.

## 4. Gate 2 — matched-budget competition (Experiment B)

Tasks `{QNLI,SST-2}`, model/training seeds `{42,2025}`, six methods, exactly four runs per method, **24 training runs**. The QNLI 42 head checkpoint is identical to Gate 1; training split identities are unchanged across methods and seeds.

- `uniform_r1`: rank 1 on all 24 modules, **B24=36,864** trainable adapter parameters.
- `eva`: PEFT-native EVA explained-variance ranks, base rank 1 with `rho=8.0` to allow up to rank 8, same B24, using only 512 direction/calibration examples. Record native EVA initialization as distinct from zero-update baselines. Check actual rank/parameter count; if version/API cannot implement budget, do not silently substitute a variant.
- `oplora`: official DPLoRA repository **stage 1 only**, with source SHA recorded; budget B24 and identical eligible module set and head protocol. If adapting the reference implementation changes its objective/constraints materially, label method `OPLoRA-style` and record a baseline fidelity risk. If faithful baseline impossible, Gate 2 cannot be marked unequivocally PASS.
- `conditional_newton`: same gradient-SVD dictionary as RVM, same candidate groups, a fixed ridge selected on evidence-only cross-validation, greedy conditional regularized quadratic utility per parameter and same budget.
- `group_rvm`: same dictionary/groups as conditional Newton; optimize marginal pseudo-evidence and use calibrated null stopping, budget ≤B24.
- `uniform_r2`: rank 2 on all 24 modules, **B48=73,728**; compression reference (not an equal-budget primary competitor).

**For every run:** final-checkpoint official validation mean binary cross-entropy and accuracy, exact rank JSON and actual trainable parameter count; calibration wall time (gradient-SVD/Jacobian/evidence/ILP/EVA initialization), adapter-training wall time, end-to-end wall time, peak GPU allocation, MLflow ID and training status. No method-specific LR tuning. All official validations occur only in Gate 2, after Gate-1 lock.

Define `L_m` as the mean of four losses (two tasks × two seeds). `L_best = min(L_uniform_r1,L_eva,L_oplora,L_conditional_newton)` at B24. Use full-precision values before rounding. The best baseline is **selected post-hoc** for a conservative target; report its identity and all alternatives.

**Route A (quality):** `L_best - L_rvm >= 0.010` nats/example **and** RVM has strictly lower loss in at least 3/4 task-seed pairs versus that chosen baseline **and** the 90% lower percentile bound of a stratified *paired example bootstrap* (2,000 resamples, seed 991; fixed run seeds) on the aggregate difference is >0. This CI conditions on the two training seeds and is not a population-of-seeds guarantee.

**Route B (compression):** `L_rvm_B24 <= L_uniform_r2_B48+0.005` **and** `L_conditional_newton_B24 - L_rvm_B24 >=0.005`, and on each individual task, the two-seed mean RVM loss is no more than `0.010` above the uniform-r2 reference. Demonstrates at least nominal 50% parameter reduction relative to uniform-r2 (count actual numbers).

**Mandatory for either route:**

1. `RVM_specificity`: Bayesian logdet/precision/stopping changes at least one selected module or rank relative to conditional Newton **and** RVM has better mean held-out loss than Newton (by at least 0.005 nats on Route B; on Route A strictly positive is enough), all under the same feature data and budget. Also compute a *no-logdet* allocation as an offline structural ablation; do not count it as a trained additional baseline unless explicitly reported exploratory.
2. `overhead`: mean RVM calibration/selection time over four task-seed configurations divided by median 600-step uniform-r1 *adapter-training* time across its four runs ≤0.25. Head warm start is excluded from both but disclosed separately. Report end-to-end total cost even when the ratio passes.
3. `baseline_integrity`: no required comparator missing/invalid, leakage, budget breach, or severe training failure; otherwise the decision is `INCONCLUSIVE`, not GO.

**Gate 2 PASS iff** `(Route A OR Route B) AND all three mandatory conditions`. No other metric can override this. If both routes fail, `NO_GO` even if RVM improves vs uniform-r1.

## 5. Measurement and inference discipline

- Keep all runs (including failures) in an append-only run index; MLflow artifacts include feature hashes and selection trace. Never drop underperforming seeds.
- Gate-1 pairwise comparisons are correlated and based on one training seed; they are a screening signal, **not** a significance test. The gate thresholds are practical investment choices.
- Gate-2 paired bootstrap resamples the *same evaluation IDs* for RVM and the selected baseline **and reuses the same sampled ID indices across the two training seeds within each task**; resample QNLI and SST-2 independently, then aggregate four mean-loss differences evenly. This preserves the correlation arising from evaluating two seed models on the same task examples. The sign test requirement is independent of the bootstrap.
- If comparisons are invalid because the frozen-head model or LoRA training is broken, report `INCONCLUSIVE` and diagnostics rather than weakening thresholds. The agent may implement a new exploratory protocol in a separate branch but not reinterpret the preregistered answer.
- Save numeric outputs before rounding and evaluation code/version hashes. Neither selection nor reporting code may tune on official validation metrics.

## 6. Budget and early stopping

The intended upper bound is **44 full adapter training runs** (20 for A + 24 for B), plus 4 possible shared head warm-starts, null simulations, small CPU tests and minimal CUDA smoke testing. Run one GPU-intensive job at a time. If any stage fails, stop the project path without consuming the remaining GPU budget; still write the report.

The experiment has only one final question: **does *Bayesian* conditional group evidence improve predictive architectural selection and downstream cost/quality over cheaper conditional selection, sufficiently to warrant scaling?**
