# Methods and mathematics: exact contracts for the pilot

This file defines what code must compute. See `PREREGISTRATION.md` for what it may decide.

## 1. Why RVM is under test

Ordinary single-atom Gaussian RVM evidence **does not produce a new ranking** beyond a conditional normalized squared gradient for equal-cost candidates. For residual feature `phi`, target `y`, active covariance `C_S`, let

\[
q=\phi^\top C_S^{-1}y,\qquad h=\phi^\top C_S^{-1}\phi,\qquad t=q^2/h.
\]

Optimizing candidate precision gives

\[
\Delta^*=\begin{cases}\tfrac12(t-1-\log t),&t>1\\0,&t\le1.\end{cases}
\]

This is strictly increasing in `t` for `t>1`. The non-trivial experiments therefore evaluate **correlated groups**, automatic stopping and final retraining outcomes, with conditional Newton as a required non-Bayesian competitor. This is a Type-II empirical Bayes approximation—not a proof of exact nonlinear LoRA model selection.

## 2. Supervised pseudo-Gaussian feature construction

For an input with two class logits `l0,l1`, let `z=l1-l0`, `p=sigmoid(z)`, label `y in {0,1}`, and `w=p(1-p)`. Clip `p` to `[1e-4,1-1e-4]` first. For candidate rank-one direction `D_mj=u_mj v_mj^T`, define the directional output derivative:

\[
J_{ij}=\left.\frac{\partial z_i}{\partial s_j}\right|_{s=0}.
\]

With binary cross-entropy, the second-order (Gauss–Newton) approximation at `s=0` yields

\[
\Phi_{ij}=\sqrt{w_i}J_{ij}, \qquad t_i=(y_i-p_i)/\sqrt{w_i},
\]

\[
\mathcal L(s)\approx\tfrac12\|t-\Phi s\|^2+\text{constant}.
\]

Use `t` rather than `y` for the pseudo-response variable in code to prevent confusion. **There is no extra factor of `1/n` inside the row-wise design**; loss reporting on audit/validation averages over examples, whereas evidence is fit to the summed quadratic surrogate. A change of objective scaling changes evidence priors/thresholds; do not mix conventions.

For a projection whose input tensor is `h_i[t,:]` and output directional derivative is `delta_i[t,:]=d z_i / d output_i[t,:]`,

\[
J_{ij}=\sum_t (\delta_i[t,:]^\top u_{mj})\,(v_{mj}^\top h_i[t,:]).
\]

Calculate this with forward hooks and the backward derivative of the **logit difference**, not of cross-entropy. Use masks as appropriate; padding contributes only through the model's valid computational path, and the forward pass must use the same attention mask as training. Check against `torch.func.jvp` or finite differences. Carefully make hooked activation outputs differentiable despite the frozen backbone (test that `delta` is nonzero). Never materialize a `[samples,all_weights]` Jacobian.

For direction generation, accumulate the *mean-loss* full projection gradient from 512 distinct examples, using CE derivative `(p-y)*d z/d output` times projection input; perform rank-8 SVD. Candidate `u`/`v` are unit norm, prefix-ordered by singular value. Fix the SVD sign ambiguity consistently (`u` pivot positive, flip matching `v`). No validation/audit examples enter candidate generation.

## 3. Group evidence objective and numerics

A selected structure `S` is a set of **prefixes** of each module's eight candidate directions. A selected module `m` with rank `r_m` has columns `1..r_m` and one positive ARD precision `alpha_m` shared within the module. Let `A` be diagonal over selected columns, repeating `alpha_m` for each rank component of that module.

\[
C_S=I+\Phi_S A^{-1}\Phi_S^\top.
\]

Relative to the empty model, the optimized evidence objective is

\[
E(S,\alpha)=\tfrac12 b_S^\top(K_S+A)^{-1}b_S
-\tfrac12\left[\log\det(K_S+A)-\log\det A\right],
\]

where `K=Phi.T @ Phi`, `b=Phi.T @ t`. Use float64, Cholesky with controlled jitter, never explicit inversion; record jitter and any optimization failure. Optimize log precisions within `[-log(1e10), log(1e10)]` (the YAML gives bounds), with multiple deterministic starts. The **empty group has E=0**. Candidates driven to the exclusion boundary may be deleted; count only finite active modules.

For an addition `G` given selected `S`, conditional evidence is equivalently

\[
\Delta(G\mid S)=\tfrac12\left[b_G^\top(A_G+K_G)^{-1}b_G
-\log\det(I+A_G^{-1}K_G)\right]
\]

with `b_G=Phi_G.T @ C_S^-1 @ t`, `K_G=Phi_G.T @ C_S^-1 @ Phi_G`, when **existing** precisions are held fixed. If re-optimizing all active precisions, the actual score is `max_alpha E(S union G,alpha) - max_alpha E(S,alpha)`. Do not mix the two definitions; the algorithm uses the reoptimized difference. Compare both formulations in tests for the no-refit case.

### Proposed structure-search rule

1. Start with all `r_m=0`.
2. For each module with `r_m<8`, consider prefix rank increments of `+1` and `+2` (if valid). For each layer, also consider a paired `query+1, value+1` group. Every candidate must fit remaining global budget of 24 rank units.
3. Score each group by **globally reoptimized** evidence increment divided by **new adapter parameter count** (all selected projections cost 1,536 per rank unit). Use a stable deterministic ordering to break ties.
4. A fixed, calibrated null threshold determines whether an addition is eligible. The initial max-gain threshold is calibrated from 100 `N(0,I)` pseudo-responses, with the fixed `Phi` and exactly the same candidate set/scoring rule. Apply the threshold to the max-scored *eligible* addition; document its units (gain per parameter). Use the same threshold at subsequent steps as an approximation; do not claim rigorous sequential FWER.
5. Add the highest eligible group. Refit active precisions, remove boundary-pruned groups, and allow one backward-delete sweep that increases total evidence. Continue until no eligible positive addition or budget exhaustion. Keep a loop-iteration limit and cycle detection so swaps/deletions cannot oscillate.
6. Output **integer ranks and exact selected module names**. `r=0` is absence of adapter. Store the detailed selection trace (`S`, prior precisions, gains, cost, delete decisions) and freeze the final structure before training.

Do not add beam search, residual refinement or a hierarchical hyperprior in the pre-registered experiments.

### Null calibration details

The target for null calibration is `t_null ~ N(0,I)` with *the actual fixed design matrix* `Phi`. Threshold the **maximum candidate gain-per-parameter from empty support**, as calculated by the particular selector. For each of 100 **fresh** null trials, run the full selector and count whether at least one module is selected. A corresponding selector-specific null threshold can be used for conditional Newton. The 100 fresh trials measure empirical complete-null activation; no claim of family-wise Type-I error after arbitrary adaptive model search is warranted.

## 4. Non-Bayesian scoring baselines

Compute the same `Phi`, `K` and `b`; cost-match all architectures. On Gate 1, architecture `A` means the full set of selected prefix directions, not an RVM-produced structure.

- **Naive gradient norm:** `score(A)=||b_A||²` (normalize only for comparing same-cost architectures).
- **Independent Fisher/ridge:** `score(A)=0.5*sum_{j in A} b_j²/(K_jj+lambda)` with fixed `lambda`. It sums **unconditional singleton** scores and intentionally ignores correlations within and between modules.
- **Conditional Newton:** At each search step, fit a ridge-regularized local quadratic model to the current active columns and compute the incremental improvement from admitting the proposed new group. For any *full* architecture in Gate 1, the corresponding score is the regularized quadratic improvement `0.5*b_A.T @ (K_A+lambda*I)^-1 @ b_A` (for pairs, the incremental score uses the difference between nested full-architecture fits). There is no evidence log-determinant complexity penalty. Use the **same** candidate-group search space as RVM in Gate 2.

Select `lambda` only by 3-fold cross-validation of quadratic-surrogate predictive error **inside the 512-example evidence set**, from grid `[1e-3,1e-2,1e-1,1,10,100]`. Do not select `lambda` from audit/official validation scores. Document whether a fold's pseudo-response features were recomputed (prefer fixed head and fixed `Phi`, disjoint row folds). Recalibrate Newton's null threshold using its own fixed statistics.

RVM evidence can change group rankings and automatic stopping; **it cannot claim a singleton-ranking advantage over `q²/h` under identical fixed covariance**.

## 5. Official baselines and implementation limitations

**EVA:** use Hugging Face PEFT's native `EvaConfig(rho=8.0)` starting from base rank 1, with budget B24 and max permissible rank 8. Its singular-vector initialization is part of the method, not equivalent to zero-update initialization; disclose. Assert actual rank sum/parameter count and record calibration time. If the pinned PEFT version cannot honor the needed constraint, document the failure and mark Gate 2 `INCONCLUSIVE` unless a faithful alternative is available **without touching held-out results**.

**OPLoRA:** implement/port *only* the ILP initial allocation of the DPLoRA authors (the `OPLoRA` variant), not progressive pruning. Source repo and commit are recorded. Validate the marginal-gain surrogate, per-module rank/cost choices and exact budget. Do not label a substantially modified optimizer an exact reproduction. A careful OPLoRA-style reimplementation may be reported separately, but it does not automatically satisfy the faithful-baseline gate.

**Uniform baselines:** all query/value modules present with rank 1 or 2, same optimizer and frozen head. Use zero-update PEFT initialization and `use_rslora=True`. All methods report both nominal and actual adapter parameters. The 36,864-parameter B24 target applies to all equal-budget methods; using fewer is allowed, using more is forbidden.

## 6. Controls and failure modes

- Posterior precision/scale alone does not identify meaningful rank if factor directions move during optimization. We **freeze candidate directions only during selection**; after selection train ordinary unconstrained LoRA factors. Selection predicts *capacity*, not an exact final factorization.
- LoRA training need not remain in the local tangent model. If all methods' scores fail to correlate with actual gains, report a **negative feasibility finding**.
- Duplicate/correlated directions can make exact support non-identifiable. Compare downstream performance and selected **functional span**, not support recovery alone.
- Record optimizer numerical failures and null behavior; changing likelihood variance/prior hyperparameter bounds post hoc can radically change automatic stopping.
- For QNLI/SST-2, the official validation split is evaluable, but it does not have to be an unseen Kaggle-like test set. Call it **official validation**, not `test` in narrative.
