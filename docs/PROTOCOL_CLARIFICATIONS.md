# Protocol clarifications (pre-freeze implementation interpretations)

**Status:** written and committed **before** `reports/protocol_freeze.json`, before any internal-audit or
official-validation outcome was computed or viewed. The frozen YAML and the frozen Markdown are unchanged; this
file records how underspecified points are implemented. Its SHA256 is part of the protocol freeze. Wherever
two documents could be read differently, the reading taken is the one prescribed by the authority order in
`AGENTS.md` (preregistration/YAML over the method documents) and, if still open, the one **less favourable to
RVM**.

## C1. Conditional Newton budget and stopping (resolved discrepancy)

PREREGISTRATION §4 defines `conditional_newton` as greedy conditional quadratic utility per parameter with the
"same budget", and lists *stopping* among the Bayesian mechanisms whose effect `RVM_specificity` tests
(`group_rvm`: "use calibrated null stopping, budget ≤B24"). METHODS §3/§4 says a selector-specific null threshold
"can be used" for Newton and to "recalibrate Newton's null threshold". Resolution (preregistration has priority):
**Newton greedily fills B24 and stops only at budget exhaustion.** Its own null threshold (seed 1931, 95th
percentile) is still computed and reported, together with the structure it *would* select, as a diagnostic only.
This reading gives the cheap competitor full capacity, so it does not favour RVM.

## C2. Gate-1 statistics

* Pairwise concordance: within each cost stratum, all pairs with `|U(A)-U(B)| > 0.005`; a pair counts as
  concordant only if `sign(score_A-score_B) == sign(U_A-U_B)` and the score difference is nonzero. Score ties
  count as discordant (no favourable tie handling). Pooled concordance = concordant / informative pairs over
  both strata. "Best non-Bayesian" for criterion 1 = maximum concordance among gradient norm, independent
  Fisher and conditional Newton; for criterion 2 = maximum six-choice mean realised utility among the same three.
* Top-3: per stratum, sort by `(-score, arch_id)` and take three.
* Non-additivity: `I_oracle(i,j) = U(A_ij) - U(L_i rank-2) - U(L_j rank-2)`; `I_evidence` analogously with the
  optimised full-architecture evidence. `I_evidence == 0` counts as an incorrect sign.
* Spearman per stratum is a diagnostic only.
* Status precedence: FAIL > BLOCKED > INCONCLUSIVE > PASS; `NOT_EVALUABLE` does not block.

## C3. Scores, ridge and null thresholds

* Ridge `lambda` (Newton and independent Fisher): ridge regression of `t` on all 192 dictionary columns, 3 folds
  by `row_index % 3` in the frozen evidence-set order, fixed `Phi`/`t` from the fixed head, mean held-out squared
  error; ties go to the larger `lambda`. The same `lambda` is used for the Gate-1 independent Fisher score.
* Null calibration: `t_null = numpy.random.default_rng(seed).standard_normal((100, 512))`; per null response the
  maximum initial gain per adapter parameter over all 60 candidate groups; threshold = `numpy.percentile(..., 95)`
  (linear interpolation). Units: log-evidence nats per adapter parameter. Fresh null evaluation runs the full
  selector (same threshold, same search) on seed-1932 responses and counts runs selecting at least one module.
* RVM optimisation: L-BFGS-B over one log-precision per active module within the YAML bounds, deterministic
  starts (warm start from the parent state when available, `log alpha = 0`, `log alpha = log median diag K_S`);
  best value kept. Single-column groups use the exact bounded closed form (tested equal to the numerical
  optimiser). Boundary pruning: `log alpha >= upper - 1e-6`. Backward-delete sweep: each active module in
  canonical order, decrement its prefix rank by one, accept if evidence increases by more than `1e-9`.
  Iteration limit 200 and cycle detection on visited rank states.
* Candidate groups (60): per module `+1`, `+2`; per layer `query+1,value+1`; strictly feasible w.r.t. remaining
  budget and max rank 8. Ties: earliest group in this fixed order.
* Optional fork pools (8 CPU workers, BLAS pinned to one thread) evaluate independent candidates; every task
  carries its explicit warm start, so results are identical to sequential execution (tested).

## C4. Training and evaluation details not fixed by the YAML

* AdamW default betas/eps; weight decay on all LoRA tensors; **no gradient clipping**; linear warmup 36 steps then
  linear decay to 0 at step 600 (`transformers.get_linear_schedule_with_warmup`).
* Data order: concatenated permutations of the 8,000-row adapter pool from `PCG64([run_seed, 202])` — identical
  for all methods with the same seed. Inputs padded/truncated to exactly 128 tokens.
* Training precision BF16 autocast (FP32 master weights); evaluation in true FP32 (TF32 disabled), loss aggregated
  in float64. Same for all methods.
* Shared zero-update initialisation for all non-EVA methods: for each (run seed, module) a fixed 8 x 768
  Kaiming-uniform matrix (bound `1/sqrt(768)`, PEFT's default distribution); `lora_A` = its first `r` rows,
  `lora_B = 0`.
* Head warm start: `RobertaClassificationHead` initialised under `torch.manual_seed(seed)`, 200 AdamW steps,
  batch 32, lr 1e-3, train mode (native dropout), BF16 autocast, order `PCG64([seed, 101])` over the 2,000 head rows.
* If a selector selects no module, its Gate-2 "model" is the frozen head-only model (evaluated, not trained).
* Official validation = all rows of the GLUE `validation` split (QNLI 5,463; SST-2 872), read only in Gate 2.

## C5. Cost accounting

`selection_wall_s` of a method = its method-specific calibration: RVM = direction pass + feature pass (GPU) +
RVM null calibration + RVM greedy search + selection-subprocess start-up (charged to RVM); Newton = the same GPU
passes + ridge CV + greedy (+ start-up); OPLoRA = its importance pass + exact ILP; EVA = PEFT
`initialize_lora_eva_weights` wall time; uniform = 0. Model loading and head warm start are excluded from the ratio
and disclosed separately. Diagnostics (fresh nulls, Newton threshold, no-logdet ablation) are excluded.

## C6. Baselines

* **EVA:** PEFT 0.21.2 native `EvaConfig(rho=8.0)` (other fields default: `tau=0.99`, `whiten=False`,
  `adjust_scaling_factors=True`), `r=1` on the 24 modules, `use_rslora=True`, `lora_alpha=4`. Batches of 32
  direction examples carry only `input_ids`/`attention_mask` (no `labels`), so PEFT's token mask equals the
  attention mask; PEFT cycles this 512-example loader until its SVDs converge. Native rank-0 removal. The run
  fails (and is reported) if actual parameters exceed B24.
* **OPLoRA (DPLoRA stage 1)**, `cjpark-ai/DPLoRA@6666d8bdb37ebd153a8916b0464c8a3878976aae` (MIT): importance
  `I_m = mean_n mean_ij (dL_n/dW_m)^2` from exact per-example CE gradients in eval mode, gain
  `I_m (1-exp(-r/768))`, cost `1536 r`, one rank per module, total ≤ 24 units, ≥1 rank among `query` and among
  `value` modules, exact optimum (DP; cross-checked against MILP). Deviations dictated by this protocol:
  q/v-only scope (reference: all 72 linear layers), rank set {0..8} (reference scripts {0..16}), budget B24,
  the 512 direction examples (reference: first 500 training rows), the frozen warm-started head (reference:
  random head), our common training recipe (`lora_alpha=4`, rsLoRA; reference `alpha=2r`), our zero-update init
  (reference: loralib Kaiming-uniform A, zero B — same distribution). Objective and constraints are unchanged, so
  the port is classified as faithful for `baseline_integrity`, with these fidelity risks reported.

## C7. Repository housekeeping

`.gitignore` gained `artifacts/` (local run mirrors, probes, per-example evaluations); the original
`SHA256SUMS.txt` entry for `.gitignore` therefore no longer matches. No protocol file was modified.
