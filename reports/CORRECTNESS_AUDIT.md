# Correctness audit of the RVM-LoRA Go/No-Go experiments

**Audit date:** 2026-10-18 (UTC) · **Audited run:** v1 (`reports/`, `artifacts/`; protocol freeze `7eb7092`, runs logged at `d57ac05`, report `21752e2`) · **Repair commit:** `d1aa145` (branch `rvm-correctness-audit`) · **Corrected run:** namespace `v2` (`reports/v2/`, `artifacts/v2/`, `checkpoints/v2/`)

**Frozen protocol untouched.** `configs/preregistered.yaml` (SHA256 `d242007e…`), `docs/PREREGISTRATION.md` (`20aea886…`), `docs/PROTOCOL_CLARIFICATIONS.md` (`52307892…`) and `docs/METHODS.md` (`603a5135…`) are byte-identical to the v1 freeze record. No threshold, dataset, model, architecture, baseline, budget, seed or metric was changed. The v1 reports and artifacts were not modified.

**Repair verdict:** one material implementation defect (A1) was confirmed and repaired without changing the protocol, so the original Gate‑0 → Gate‑1 chain was rerun in the isolated `v2` namespace (Phase 6). A second defect (A2) was latent: it did not affect v1, but it would have made a corrected rerun silently reuse v1 outputs. It was repaired before the rerun. The two concerns raised in the task were the near-chance classification head and the mismatch between the gradient-SVD evidence directions and the randomly initialized LoRA subspace. Both are **intentional protocol design**, not coding errors. The implementation of each was verified to behave exactly as specified.

All evidence below was produced by read-only audit scripts (`scripts/audit/*.py`). Their outputs are stored in `reports/audit/*.json`. None of them feeds a gate.

## 1. Summary of findings

| ID | Concern | Classification | Affected v1 numbers? | Repair |
|---|---|---|---|---|
| A1 | RVM precision optimiser returns local optima of the Type‑II evidence | `CONFIRMED_IMPLEMENTATION_BUG` (material) | Yes: 3/20 Gate‑1 RVM scores (by +10.2, +1.27, +11.8 nats), one interaction sign input, the real-data RVM structure | Repaired (`d1aa145`), regression tests, Gate 1 rerun as `v2` |
| A2 | Stage caches (head, probe, head-only audit, precomputed scores, completed runs, Gate 0) reused without checking head hash or code version | `CONFIRMED_IMPLEMENTATION_BUG` (latent) | No. v1 computed every stage fresh in one pass (timeline and hashes below) | Repaired (`d1aa145`): provenance-checked reuse + `RVM_RUN_NAMESPACE` isolation |
| H | QNLI head is near chance after the warm start (audit acc. 0.502, CE 0.69092) | `INTENTIONAL_PROTOCOL_DESIGN` (prescribed recipe, implemented correctly) | — | Not repairable without changing the frozen setting; documented as a limitation |
| L | RVM scores gradient-SVD atoms `u_j v_jᵀ`, while LoRA starts at `B=0` with random `A` | `INTENTIONAL_PROTOCOL_DESIGN` (METHODS §6; PREREG §1; C4) | — | Not changed; limitation of the hypothesis; one separately labelled exploratory SVD-initialisation check (§8) |
| N | Null pseudo-responses `N(0,I)` while the real `t` has mean −0.098 (biased head) | `INTENTIONAL_PROTOCOL_DESIGN` (PREREG §3, METHODS §3) | — | Not changed; noted for interpretation |
| — | Gate evaluator, splits, hashes, MLflow metadata, Jacobians, LoRA placement/initialisation, Newton/Fisher/gradient-norm scores, ridge CV, null calibration code, determinism | Verified correct | — | — |
| U1 | Bitwise reproducibility of the 20 Gate‑1 trainings | Verified by the v2 rerun: logits and loss curves bit-identical (§7) | — | — |
| U2 | Gate‑2 code paths (EVA/OPLoRA/Newton/RVM training, bootstrap, overhead) | `UNVERIFIED` in execution (unit-tested only; Gate 2 never ran) | — | — |

## 2. Phase 1: reproduction of the recorded v1 calculations

`scripts/audit/phase1_recompute.py` → `reports/audit/phase1_recompute.json`. The gate statistics were recomputed with code written for the audit, not with `rvm_lora.gates`, starting from the stored per-example logits.

| Quantity | Reported | Recomputed from raw artifacts |
|---|---|---|
| Head-only audit accuracy / CE | 0.502 / 0.690920654655885 | 0.502 / 0.690920654655885 (raw logits; per-example losses identical) |
| 20 run audit CEs | `run_index.csv` | max abs. discrepancy 0.0 (all recomputed from saved logits; eval-file hashes match) |
| Informative cost-matched pairs | 86 | 86 |
| RVM / gradient-norm concordance | 0.709 / 0.814 | 61/86 = 0.7093023 / 70/86 = 0.8139535 |
| Concordance margin vs best non-Bayes | −0.10465 | −0.10465116 (best = gradient norm) |
| RVM top-3 margin | +0.00405 | +0.004051394741717146 (best = gradient norm) |
| Interaction signs | 0/3 | 0/3 informative (L00+L01, L05+L06, L10+L11 wrong; L02+L09 uninformative) |
| Fresh null activations | 5/100 | 5/100 (selection job re-run bit-identical) |
| Gate 1 / overall | FAIL / NO_GO | FAIL (criteria 1b, 2, 4) / NO_GO |

Consistency checks (all true): YAML/PREREG/METHODS/C-doc/`uv.lock` hashes equal the freeze record. QNLI split IDs re-hash to the manifest and freeze values; splits are disjoint with sizes 2000/512/512/6976/1500, and adapter_train = directions+evidence+other. Encoded arrays align with the manifests. The head checkpoint file hash equals its metadata and the probe record (`08d1ce5a…`). Features/atoms/selection/precomputed-score hashes equal their records. Scores were frozen at 19:32:06, before the first audit read at 19:32:12. The 20 Gate‑1 MLflow runs are found, FINISHED, with tags `head_checkpoint_sha256=08d1ce5a…`, `protocol_sha256=d242007e…`, `data_split_sha256=880378c3…` (adapter_train), `git_sha=d57ac05`. Commit `d57ac05` differs from the freeze commit `7eb7092` only in timing accounting (`git diff 7eb7092 d57ac05 -- src`: two lines, `architecture_scores_wall_s`).

Observation: the head predicts class 1 for 94.3% of audit rows (label mean 0.475), logit-difference std 0.110. Its CE (0.69092) is only 0.001 nats below a constant predictor at the audit base rate (0.69193).

## 3. Phase 2: classification-head warm start

Code: `src/rvm_lora/modeling.py:108-135` (`warm_start_head`), `:70-86` (`save_head`/`load_head`), `src/rvm_lora/pipeline.py:96-128` (v1 line numbers), `src/rvm_lora/data.py:21,128-135` (tokenisation). Evidence: `scripts/audit/phase2_head.py` → `reports/audit/phase2_head.json` (GPU, read-only; new label reads only on head/direction/evidence rows).

| Required check | Execution evidence | Result |
|---|---|---|
| Pretrained checkpoint | All 197 backbone tensors bit-identical to `FacebookAI/roberta-base@e2da8e2f` `model.safetensors` (sha `5bde1d28…`); backbone identical under seeds 42 and 7 | correct |
| Head initialisation | `RobertaClassificationHead`, dense/out_proj weights std 0.0200 (= `initializer_range`), biases 0; differs across seeds | correct (C4) |
| Label encoding | GLUE names `[entailment, not_entailment]` ↔ 0/1; encoded labels equal dataset labels (first 500 head rows) | correct |
| Question–sentence tokenisation | decoded `<s>question</s></s>sentence</s>`; re-tokenisation of 200 rows bit-identical; truncated at 128: 0.2–0.8% of rows | correct |
| Optimiser parameters | AdamW over exactly the 4 head tensors (592,130 params), lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, 200 steps × batch 32 | as prescribed |
| Gradients and steps | all 4 tensors have gradients at all 200 steps; grad norm 2.16 → median 1.07 → 0.71; update norm 0.77 → 0.12 | optimiser works |
| Head changed / backbone frozen | dense weight changed by ‖Δ‖=7.92 (51% of init norm), out_proj 40%, biases moved from 0; backbone hash unchanged, all backbone grads `None` | correct |
| Initial / final training loss and accuracy | batch loss 0.727 → 0.684 (first/last 20 steps, train mode), batch accuracy 0.488 → 0.558; per epoch 0.707/0.698/0.696/0.681 | learns, slowly |
| Train vs eval mode | all 200 forward passes in train mode (native dropout, C4); model returned in eval mode with no trainable parameters | as prescribed |
| Save / restore | re-running the repository function reproduces the checkpoint **bit-for-bit** (sha `08d1ce5a…`, identical first/last losses); save→reload logits bit-identical (max diff 0.0) | correct |
| Logits before saving vs after reloading | bitwise equal (256 rows); stored head-only audit logits reproduced bit-for-bit | correct |
| Same head for all Gate‑1 adapters and probes | digest of the loaded checkpoint `fd60b6ee…` equals the `head_digest` recorded by all 20 runs (≠ random-init digest `9e0866c8…`); probe `z` equals eval logits (max diff 7.6e‑7, batch-size rounding) | identical head |

Head-only quality (eval mode): head rows 0.543 acc / 0.6787 CE (its own training rows), direction rows 0.512 / 0.6894, evidence rows 0.498 / 0.6906, audit 0.502 / 0.6909. The head predicts class 1 for 94–96% of rows.

**Why it stays near chance.** The frozen `<s>` vectors are extremely anisotropic. Mean pairwise cosine is 0.9965, and dimension 588 has mean −11.29 against a mean vector norm of 11.46, so nearly all of the norm sits in one constant direction. An *exploratory diagnostic*, not used anywhere: standardised L2 logistic regression fitted on the 2,000 head rows reaches 0.65–0.67 accuracy on the 1,024 direction+evidence rows. The features therefore carry linear signal that the prescribed recipe does not extract in 200 steps. That recipe is an unstandardised tanh head from random init, lr 1e‑3, native dropout that also randomly zeroes the dominant coordinate, and BF16 autocast per C4. Which of these factors dominates was not isolated, because doing so would require running the recipe with changes.

**Classification:** `INTENTIONAL_PROTOCOL_DESIGN`. The recipe is prescribed by PREREG §1 and C4 and is implemented exactly. **Scientific consequence:** Gate‑1 utilities measure how much each adapter *rescues a near-chance, miscalibrated head*. Part of every U is a calibration correction (the head over-predicts class 1). The probe's pseudo-response is essentially `t≈±1` at nearly constant `p≈0.54`. **Not repaired:** unfreezing the backbone, changing the head recipe, adding data or a task-specific pretrained head would change the frozen experimental setting.

## 4. Phase 3: evidence dictionary vs trainable LoRA subspace

Code: `src/rvm_lora/probe.py:97-164`, `src/rvm_lora/adapters.py:24-66,127-134`, `src/rvm_lora/selectors.py:42-73`. Evidence: `scripts/audit/phase3_lora_map.py` → `reports/audit/phase3_lora_map.json`.

| Check | Evidence | Result |
|---|---|---|
| Directional Jacobians | real model + real head, JVP of `z=l1−l0` along `u_mj v_mjᵀ` for the first two atoms of all 24 modules vs stored `J`: max rel. error 1.1e‑5 | correct |
| Projection matrices / transpositions / module order | the same JVP check addresses columns `m·8+j` in `cfg.modules` order; `G_m` is (out×in), `u` output side, `v` input side | correct |
| PEFT orientation and scaling | PEFT LoRA with `A=V_rᵀ`, `B=U_r diag(s)/scaling` reproduces `J s` (rel. 1e‑4–5e‑3, finite-difference); swapped orientation gives ~100% error; scaling = `4/√r` (rsLoRA) | correct |
| Selected ranks → modules | `assert_exact_layout` per run; all 20 architectures' PEFT layouts equal the YAML ranks | correct |
| Zero initial update | all 20 architectures: logits after `build_peft` bit-identical to head-only; `B=0` | correct |
| Shared rank prefixes | per-(seed, module) 8×768 Kaiming-uniform matrix; rank‑r `A` = first r rows across all architectures | correct |
| Probe determinism | atoms, `J`, `Φ`, `t` recomputed bit-identically | correct |

**The mismatch is real and large.** At `B=0` the trainable update is `(dB)A`, so its row space is the random row space of `A`. Measured over all Gate‑1 module/rank cases:

- The gradient-SVD atoms put 0.31% of their energy in that row space (random expectation r/768).
- The LoRA tangent captures 0.36% of the direction-set mean-loss gradient ‖G‖²_F.
- The gradient-SVD prefix that RVM scores captures 96.7%.

RVM therefore scores the best-aligned rank‑r capacity, while the trained adapter starts almost orthogonal to it and must rotate `A` during 600 steps.

**Classification:** `INTENTIONAL_PROTOCOL_DESIGN`. METHODS §6 freezes candidate directions only for selection ("selection predicts capacity, not an exact final factorisation"). PREREG §1 and C4 prescribe one identical zero-update random initialisation for all non-EVA methods. **Consequence:** a limitation of the experimental hypothesis. A failure of the scores to predict utility may reflect the surrogate gap rather than the evidence criterion. Not changed. A separately labelled exploratory check with SVD-aligned `A` is reported in §8.

## 5. Phase 4: remaining implementation

| Component | Evidence | Result |
|---|---|---|
| RVM objective and log-determinant | Gram/Cholesky form vs direct n×n `log N(t|0, I+ΦA⁻¹Φᵀ) − log N(t|0,I)` at the recorded precisions on the real features: max rel. diff 1.8e‑12 (20 architectures) | correct |
| **Gaussian precision optimisation** | **local optima, see A1** | **defect** |
| Group gain ("importance") | gain per parameter = ΔE / (units × 1536); 60 candidate groups (+1, +2, q+1&v+1), feasibility, prefix ranks | correct |
| Null calibration | threshold = 95th percentile (linear) of max initial gain over 100 seed‑1931 `N(0,I)` responses; fresh seed‑1932 run of the full selector; re-run bit-identical | correct as specified |
| Conditional Newton | ridge CV recomputed independently (λ=0.01, CV‑MSE identical to 1e‑13); score `½bᵀ(K+λI)⁻¹b` re-implemented, max rel. diff 6e‑15; fills B24 (C1) | correct |
| Gradient norm / independent Fisher | re-implemented, max rel. diff 8e‑16 / 6e‑16 | correct |
| Gate decisions | independent recomputation identical (§2) | correct |
| Thresholded Newton = RVM structure (v1 exploratory claim) | reproduced on v1 code: both select L3Q:2, L4Q:1, L6Q:1, L7V:1, L10V:1 | true for v1, but the v1 RVM path was itself affected by A1 (see A1 and §7) |
| Caching | see A2 | defect (latent) |

### A1: evidence optimiser stops in local optima (`CONFIRMED_IMPLEMENTATION_BUG`, material)

*Where:* `src/rvm_lora/evidence.py:164-208` (v1 `optimize_evidence`, starts at lines 181-183). It is used by `selectors.py:159-176` (`RVMObjective._value`, i.e. greedy search, null calibration and fresh nulls) and `selectors.py:429-433` (`architecture_scores`, the Gate‑1 RVM score).

*Evidence:* `scripts/audit/phase4_evidence.py`, `phase4b_localopt.py`, `reports/audit/phase4_evidence.json`, `phase4b_localopt.json`. An independent dense multi-start search of the same bounded objective found higher evidence for 3 of the 20 architectures. The higher optima were confirmed with the direct n×n formula (cond(C) ≤ 1.2e7):

| Architecture | v1 recorded E | maximised E | v1 log α (q, v, …) | maximising log α |
|---|---:|---:|---|---|
| L11_qv_r2 | 3.653186 | 13.849842 | (2.88, 23.03) | (−10.99, 6.52) |
| L03_qv_r4 | 16.373105 | 17.644929 | (−4.64, 23.03) | (−3.96, −6.92) |
| L10+L11_qv_r2 | 3.653186 | 15.447057 | (23.03, 23.03, 2.88, 23.03) | (−1.72, 23.03, −11.1, 23.03) |

The evidence profile on the straight line between the two solutions falls from 3.65 to 1.17 before rising to 13.85, so these are separate basins and not a convergence tolerance issue.

*Cause:* one precision is shared per module across its prefix columns, and column scales differ by orders of magnitude. For example, L11 query atoms have norms 13.1 and 0.03, and L11 value atoms 209 and 23.8. The C3 starts (warm start, log α=0, a single global log median diag K) all fall into the basin that fits only the large column. The prescribed objective is the *maximised* evidence: PREREG §3 "E optimized for each architecture"; METHODS §3 "max_α E(S,α)", "the algorithm uses the reoptimized difference". The implementation did not compute it.

*Scope on real data (v1 code):*

- Greedy search: at step 1 the true best group is L11Q:+2 (gain 4.37e‑3/param), not L10V:+1 (3.80e‑3). The RVM path and structure change.
- Null calibration: 3 of 3,600 group problems are under-maximised (all 100 calibration nulls). The calibrated threshold is nonetheless identical in v2 (1.3718244e‑3), and fresh-null activations stay 5/100.
- Synthetic scale-disparate groups: 15/300 under-maximised.

*Consequences:*

- The RVM Gate‑1 scores of L11_qv_r2, L03_qv_r4 and L10+L11_qv_r2 are wrong.
- The interaction input `I_evidence` for L10+L11 is wrong, and so is the RVM structure on real features.
- Two v1 exploratory interpretations were partly artefacts of this defect: "value projections excluded in 14 of 15 boundary cases" (e.g. L11 value is active at the true optimum of L11_qv_r2) and "thresholded Newton selects RVM's exact structure".

*Repair (protocol-preserving), commit `d1aa145`:*

- The C3 starts are kept and retained unless beaten by > 1e‑9 nats.
- Added deterministic scale-aware starts: each module at its own column scale, and at its columns' bounded singleton optima `log(h²/(q²−h))`.
- Added a block-coordinate global line search over each module's log α (grid step 0.5 over the YAML bounds), polished by L‑BFGS‑B.

Nothing in the objective, bounds, dictionary, candidate groups, threshold rule or search changed. *Verification:* the repaired optimiser is never below the dense optimum on any of these sets:

- the 20 architectures (max gap 5e‑10);
- every candidate at every step of the repaired real-data greedy search (max gap 1.1e‑9);
- 3,600 null-calibration group problems (1.2e‑9);
- 300 synthetic problems (9e‑10; it exceeds the dense search once).

By comparison, the v1 start set falls short on 3/20 architectures, on 1–2 candidates at 5 of the 6 steps of the v1 greedy path, on 3/3,600 null-group problems and on 15/300 synthetic problems. *Regression tests* (Gate‑0 check `regression_A1_evidence_global_maximisation`):

- `tests/test_evidence.py::test_optimizer_reaches_global_maximum_on_real_subproblems` uses a fixture of the three affected 4×4/8×8 Gram blocks of the training-pool evidence rows, `tests/fixtures/qnli42_evidence_subproblems.json`. It also asserts that the C3 starts alone fall short by >1 nat.
- `tests/test_evidence.py::test_optimizer_matches_dense_search_scale_disparate`.

*Note on C3:* `docs/PROTOCOL_CLARIFICATIONS.md` C3 lists the three starts as the implementation of "multiple deterministic starts; best value kept". The repair keeps those starts and the best-value rule and adds further deterministic starts, because the higher-authority definition (PREREG/METHODS) is the maximum. C3 itself is unchanged.

### A2: stale-cache reuse (`CONFIRMED_IMPLEMENTATION_BUG`, latent)

*Where (v1):* `pipeline.py:100` (head reused if a file exists), `:154` (probe/features/selection reused if `probe_result.json` exists), `:375` (head-only audit), `:272` and `:512` (completed runs keyed only by `(protocol SHA, gate, method, task, seed, arch)`), `cli.py:203-204` (Gate 0 reused if any PASS exists, regardless of code), `tracking.py:20` (run keys without a version namespace).

*Defect:* none of these checked the head checkpoint hash, feature hash or code version, contrary to ENGINEERING §6 ("Completed runs with matching artifact hashes should be reused"). Rerunning v1 after a fix would have silently reused the v1 head, features, scores, audit results and all 20 completed trainings, and skipped Gate 0.

*Did it affect v1?* No. The v1 log shows a single fresh pass: head 19:31:26, probe 19:31:28–19:32:06, audit and runs 19:32:12–19:47. All 20 runs record the same head digest and MLflow head-hash tag, and each MLflow run is attempt 1.

*Repair:*

- Every reuse now requires matching provenance (`check_cached` → `StaleCacheError`): head hash plus namespace for probe, audit, runs and empty selections; selection-file and head hash for the precomputed scores.
- Gate 0 is reused only for the same git SHA.
- `RVM_RUN_NAMESPACE` gives isolated `artifacts/<ns>`, `checkpoints/<ns>`, `reports/<ns>` and namespace-salted run keys. v1 keys are unchanged.

*Tests:* `tests/test_cache.py` (8 tests; Gate‑0 check `regression_A2_cache_provenance_and_namespace_isolation`).

### N: null pseudo-response vs biased head (`INTENTIONAL_PROTOCOL_DESIGN`)

PREREG §3 prescribes `t_null ~ N(0, I)`. With the near-chance head, the real `t` has mean −0.098 and variance 0.996, i.e. a 2.2‑σ calibration offset along the all-ones direction. Real-data gains can therefore partly reflect directions that fix calibration, which the complete-null threshold does not model. Not changed.

## 6. Phase 5: repair decision

A1 is material and repairable within the protocol: the protocol defines the score as the maximised evidence, and the repair only makes the code compute it. A2 had to be repaired for any corrected run to be valid. H, L and N are design properties of the frozen protocol and cannot be changed. Decision: **proceed to Phase 6** (not `NO_PROTOCOL_PRESERVING_REPAIR_IDENTIFIED`).

**Integrity caveat.** The corrected evaluation necessarily comes *after* the v1 audit outcomes were known. Four safeguards limit outcome-driven tuning:

- A1 was found by an outcome-blind check: the global optimality of the defined objective for every architecture.
- The repair is a uniform rule applied to every evidence evaluation (scores, greedy, nulls), not to selected architectures.
- The repair was committed (`d1aa145`) and verified on audit-free data before any corrected gate statistic was computed.
- Thresholds, architectures, data, seeds and the decision rule are unchanged.

Even so, the v2 Gate‑1 verdict is a post‑hoc corrected re-evaluation, not a fresh blinded test.

## 7. Phase 6: corrected run (`v2`)

**Setup.**

- Code `d1aa145` (repairs A1/A2), executed as `RVM_RUN_NAMESPACE=v2 bash scripts/run_remote.sh` in tmux `rvm_pilot_v2` on the RTX A4000. The MLflow server was local at 127.0.0.1:5000; no firewall change was made and no port was exposed.
- Outputs are isolated: `artifacts/v2/`, `checkpoints/v2/` and `reports/v2/`, with namespace-salted run keys and new MLflow run IDs. The v1 outputs are untouched.
- Stages ran in the original order: env-check, prepare, Gate 0, freeze, head warm start, probe + selection, head-only audit, the 20 Gate‑1 trainings, Gate‑1 evaluation, Gate 2 (gated), report.
- The prepared, deterministic split encodings in `data/` were shared. Their manifests are hash-identical to the v1 freeze (`split_manifests_equal: true` in `reports/v2/protocol_freeze.json`).
- Training budget used: **20 of 44**. Gate 2 was not run because Gate 1 failed.

**Provenance** (`reports/v2/protocol_freeze.json`, frozen 2026‑10‑18T05:37:10Z, before the v2 head, probe or audit):

| Item | Original v1 | Corrected v2 |
|---|---|---|
| Code SHA (freeze / runs) | `7eb7092` / `d57ac05` | `d1aa145` / `d1aa145` |
| Config / PREREG / C-doc / METHODS SHA256 | `d242007e…` / `20aea886…` / `52307892…` / `603a5135…` | identical (checked by `write_freeze`) |
| `uv.lock` SHA256 | `dd27a90b…` | identical |
| Model | `FacebookAI/roberta-base@e2da8e2f…` (backbone bit-identical to hub safetensors) | identical |
| Split manifests (QNLI adapter_train / audit / head) | `880378c3…` / `0d5f185a…` / `55d2474d…` | identical |
| Head checkpoint SHA256 | `08d1ce5a…` | `08d1ce5a…` (bit-identical; new file `checkpoints/v2/head_qnli_seed42.safetensors`) |
| Probe features / atoms / Φ SHA256 | `31be4fda…` / `3d360e4c…` / `b5cd954d…` | identical |
| MLflow: Gate 0 / head / probe / head-only audit | `64d3c655…` / `ebc1c7a0…` / `2bd67ec2…` / `538d92c0…` | `9416f174…` / `4405cbcc…` / `0020a49d…` / `0c34ff0e…` |

**Gate 0 (v2): PASS, 56/56.** All 12 original checks pass, plus the two regression checks: `regression_A1_evidence_global_maximisation` (2 tests) and `regression_A2_cache_provenance_and_namespace_isolation` (8 tests). This includes the CUDA smoke tests and the live-MLflow round trip. The A1 fixture test measured v1 under-maximisation of 10.20 / 1.27 / 11.79 nats, which the repaired optimiser closes.

**Determinism of the rerun.** All 20 Gate‑1 trainings reproduced v1 **bit-for-bit**: identical audit logits and per-step training losses. The head-only audit logits and the three non-Bayesian scores are also identical (`reports/audit/v1_v2_comparison.json`). Only the three A1-affected RVM scores changed: L11_qv_r2 3.653 → 13.850, L03_qv_r4 16.373 → 17.645, L10+L11_qv_r2 3.653 → 15.447. This resolves U1.

**Corrected Gate 1** (mechanical evaluator, unchanged YAML thresholds; `reports/v2/gates.json`):

| Criterion | Cutoff | v1 | v2 (corrected) | v2 status |
|---|---|---:|---:|---|
| informative cost-matched pairs | ≥ 30 | 86 | 86 | PASS |
| RVM pairwise concordance | ≥ 0.65 | 0.7093 (61/86) | **0.7907 (68/86)** | PASS |
| RVM − best non-Bayes concordance (gradient norm 0.8140, 70/86) | ≥ +0.05 | −0.1047 | **−0.0233** | **FAIL** |
| RVM six-choice top‑3 utility − best non-Bayes (gradient norm 0.07972) | ≥ +0.005 | +0.00405 | **+0.01179** (RVM 0.09151) | PASS (was FAIL) |
| fresh null activations | ≤ 10 | 5 | 5 | PASS |
| non-additivity signs (3 informative) | ≥ 3 | 0/3 | **1/3** (L10+L11 now correct) | **FAIL** |
| **Gate 1** | | FAIL | **FAIL** | |

**Gate 2: `NOT_RUN_GATE_FAILURE`** (conditional progression). No official-validation set was read, and no Gate‑2 number exists. **Overall: `NO_GO`**, unchanged.

**Before/after on the requested quantities:**

| Quantity | v1 | v2 |
|---|---|---|
| Head quality (audit acc. / CE; head-rows acc.) | 0.502 / 0.690920654655885; 0.543 | identical (bit-identical head) |
| Gate‑1 utilities U (20 architectures) | 0.0077–0.1795 nats | identical (bit-identical trainings) |
| Concordance: gradient norm / indep. Fisher / cond. Newton / RVM | 0.814 / 0.593 / 0.698 / 0.709 | 0.814 / 0.593 / 0.698 / **0.791** |
| Spearman RVM (6,144 / 12,288) | 0.406 / 0.619 | 0.517 / 0.762 |
| Top‑3 mean U: gradient norm / Fisher / Newton / RVM | 0.0797 / 0.0713 / 0.0773 / 0.0838 | 0.0797 / 0.0713 / 0.0773 / **0.0915** |
| RVM top‑3 choices | 6,144: L03, L06, L05; 12,288: L05+L06, L03r4, L11r4 | 6,144: L03, L06, **L11**; 12,288: L05+L06, L03r4, **L10+L11** |
| Interaction sign accuracy (RVM) | 0/3 | 1/3 |
| RVM structure on real features | L3Q:2, L4Q:1, L6Q:1, L7V:1, L10V:1 (6 units) | **L4Q:1, L11Q:2, L6Q:1 (4 units)**, null-threshold stop |
| Null-thresholded Newton (diagnostic) | identical to RVM | L3Q:2, L4Q:1, L6Q:1, L7V:1, L10V:1, **now differs from RVM** |
| RVM null threshold / fresh activations | 1.3718244e‑3 / 5 | identical |
| RVM selection cost charged | 28.7 s | 35.5 s (repaired optimiser +7 s CPU) |

**Regularisation and stopping vs Bayesian evidence (observations, then interpretation).** The Gate‑1 scores are full-architecture scores, so no stopping is involved in them. RVM and conditional Newton share the same Φ, t and local quadratic surrogate. They differ only in the per-module ARD precisions and the log-determinant (RVM) versus a fixed ridge λ=0.01 (Newton). In v2, RVM beats Newton by +0.093 concordance and +0.0142 nats in top‑3, so that difference is attributable to the evidence term. It still loses on concordance to the raw gradient norm, which uses no curvature or regularisation at all. In selection, the v1 finding that a null-thresholded Newton reproduces RVM's structure was an A1 artefact. In v2 the evidence selects a different, sparser structure (4 vs 6 units), and the no-logdet ablation selects a third one. Whether any of these structures trains better at equal budget was never measured (Gate 2 not run), so no benefit of Bayesian stopping or regularisation can be claimed.

## 8. Exploratory SVD-initialisation check (not part of any gate)

**Status:** EXPLORATORY. It is not a protocol amendment, not counted in the corrected run (20/44 used there), feeds no gate and changes no verdict. It ran after the v2 Gate‑1 verdict was final. Code: `scripts/audit/explore_svd_init.py`. Raw results: `reports/audit/exploratory_svdinit_runs.json`; statistics: `reports/audit/v1_v2_comparison.json` → `EXPLORATORY_svdinit`. MLflow experiment `rvm-lora/exploratory-svd-init` (20 runs).

**Design.** The same 20 architectures were trained with the same v2 head, data order, seed 42, 600 steps, rsLoRA α=4 and AdamW 2e‑4, with **one** change. For each module with rank r, `lora_A` = the first r gradient-SVD right singular vectors `v_m1..v_mr` of the v2 probe, i.e. the atoms RVM scores. Each row is rescaled to the norm of the protocol's deterministic Kaiming row it replaces, and `lora_B = 0`. The zero initial update was verified bit-exactly for all 20. The initial trainable tangent `{X·A}` then contains every scored atom `u_mj v_mjᵀ`. The random initialisation of the protocol was **not** replaced in any gate-relevant run.

**Observations** (same frozen scores as v2, applied to the SVD-init utilities, 91 informative pairs):

| | Random init (protocol, v2) | SVD-aligned init (exploratory) |
|---|---:|---:|
| Mean U over 20 architectures (nats) | 0.0554 | 0.1278 |
| Concordance: gradient norm / Fisher / Newton / RVM | 0.814 / 0.593 / 0.698 / 0.791 | 0.703 / 0.681 / **0.758** / 0.692 |
| Top‑3 mean U: gradient norm / Fisher / Newton / RVM | 0.0797 / 0.0713 / 0.0773 / 0.0915 | 0.1550 / 0.1623 / **0.1657** / 0.1561 |
| Interaction signs (RVM), informative | 1/3 | 1/4 (I_oracle: L05+L06 −0.158, L10+L11 −0.069, L00+L01 −0.006, L02+L09 +0.112) |
| Gate‑1 rule applied mechanically (illustration only) | FAIL | FAIL (best non-Bayes = Newton; margins −0.066 / −0.0096) |

**Interpretation.**

- Aligning the initial LoRA subspace with the scored directions more than doubles the realised 600-step utility.
- It removes the raw gradient norm's advantage: conditional Newton then ranks best (0.758 vs 0.703 concordance).
- The two pairs that were synergistic under random init, (5,6) and (10,11), become clearly redundant, as the tangent model predicts. (2,9) becomes synergistic.

So much of the v1/v2 pattern is plausibly a consequence of the evidence/initialisation mismatch (L), not of a coding error. Even in the aligned setting, Bayesian evidence does **not** improve on its non-Bayesian analogue (conditional Newton) for ranking or top‑3 selection. This single-seed exploratory check cannot support a positive claim for RVM, and it does not alter the pre-registered decision.

## 9. Answers

**1. Were the original Go/No-Go experiments implemented incorrectly?**
Partly. The data pipeline, head warm start, probes, Jacobians, LoRA placement and initialisation, training, evaluation, non-Bayesian scores, null calibration and gate evaluator are all correct and reproduce bit-for-bit. One material defect existed: the RVM evidence optimiser (A1) returned local optima. As a result, 3 of the 20 Gate‑1 RVM scores, one interaction-sign input and the RVM-selected structure were wrong. The caching logic (A2) was also defective, but it did not affect any v1 number.

**2. Which concerns were implementation bugs, and which were deliberate but potentially problematic protocol choices?**
- Bugs: A1 (material) and A2 (latent).
- Deliberate protocol choices:
  - H: the prescribed frozen-backbone 200-step head recipe yields a near-chance, miscalibrated head on these anisotropic features.
  - L: RVM scores best-aligned gradient-SVD atoms while LoRA trains from a random zero-update initialisation whose initial tangent captures 0.36% of the gradient energy versus 96.7% for the scored atoms.
  - N: the `N(0,I)` null ignores the head's calibration offset.

  All three are implemented exactly as specified, and all three limit what Gate 1 can show.

**3. What exactly was corrected?**
Commit `d1aa145`:
- `optimize_evidence` now reaches the maximised evidence. It keeps the C3 starts and adds scale-aware deterministic starts plus a block-coordinate global line search.
- Every cache reuse is provenance-checked, Gate 0 is reused only for the same commit, and corrected runs live in an isolated `RVM_RUN_NAMESPACE`.
- 10 regression tests were added and registered as Gate‑0 checks.

Commits `4d4a6fb` and `577fd0d` fixed a reporting bug in the new namespaced report code and a path mix in a post-hoc test. Later commits touch reports and documentation only. None of them affects any result. No protocol file, threshold, dataset, model, architecture, baseline, budget, seed or metric changed.

**4. Was a valid protocol-equivalent rerun possible and completed?**
Yes, for the stages the protocol allowed:
- Gate 0: rerun, PASS 56/56.
- Gate 1: the head was recreated (bit-identical), and features, scores and null thresholds were recomputed before the v2 audit evaluation. The 20 original architectures were retrained (seed 42, 600 steps; bit-identical to v1) and evaluated with the unchanged rules.
- Gate 2: correctly not run.

The rerun used 20 of the 44 allowed trainings. Caveat: it is a post-hoc correction performed after the v1 audit outcomes were known (§6), not a fresh blinded test.

**5. Do the corrected results change the original decision, or only limit its interpretation?**
They do **not** change the decision: Gate 1 still FAILs and the project decision remains **`NO_GO`**. They change and limit its interpretation:
- v1 understated RVM. The top‑3 criterion now passes (+0.0118 ≥ 0.005), concordance rises to 0.791, and RVM clearly beats conditional Newton.
- The failure now rests on two criteria: concordance versus the raw gradient norm (−0.023 < +0.05) and non-additivity signs (1/3 < 3).
- Two v1 exploratory explanations were partly artefacts of A1: the identical thresholded-Newton structure and the value-exclusion pattern.
- The near-chance head and the evidence/LoRA-initialisation mismatch mean this negative result is specific to the frozen setting. It is not evidence that RVM evidence could not help under a better-posed protocol, which would require a new, separately registered study.
