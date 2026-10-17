# Report template — fill into `reports/FINAL_REPORT.md`

> **Important:** This is a template, not an outcome. Replace `[NOT RUN]` placeholders with measured numbers **only after execution**. Use `BLOCKED`, `INCONCLUSIVE`, `FAIL`, `PASS`, or `NOT_RUN_GATE_FAILURE` where appropriate. No claim of GO without a machine-computed gates.json verdict.

## RVM-LoRA pilot: final Go/No-Go assessment

**Date:** [timestamp UTC]  
**Overall decision:** [GO_TO_3B_REPLICATION / NO_GO / INCONCLUSIVE / BLOCKED]  
**Gate 0:** [status] · **Gate 1:** [status] · **Gate 2:** [status]  
**Git SHA:** [sha] · **Protocol YAML SHA256:** [sha] · **uv.lock SHA256:** [sha]  
**MLflow tracking URI:** remote-local `127.0.0.1:5000`, UI through SSH tunnel  
**MLflow experiments:** [IDs/links/paths]

### Executive judgment (≤300 words)

[Answer in ordinary language: Does group RVM improve over cheap conditional Newton? Are the effect size and selection overhead compelling? Which criteria passed or failed? What outcome recommends proceeding or stopping?]

### Experimental integrity and environment

| Evidence item | Value / hash | Verified? |
|---|---|---|
| Hardware GPU / VRAM / driver | [ ] | [ ] |
| OS / Python / torch / CUDA / Transformers / PEFT / MLflow | [ ] | [ ] |
| Model ID and exact revision | [ ] | [ ] |
| QNLI training/audit split hash | [ ] | [ ] |
| SST-2 training/audit split hash | [ ] | [ ] |
| Classifier-head checkpoint hashes per task/seed | [ ] | [ ] |
| Protocol freeze timestamp / git / YAML / MD | [ ] | [ ] |
| Pilot run manager log and run index | [ ] | [ ] |
| Evidence/calibration/official validation isolation | [ ] | [ ] |
| Parameter-count assertions, LoRA-only gradients | [ ] | [ ] |
| Baseline fidelity and source commits | [ ] | [ ] |

### Gate 0: numerical and pipeline correctness

| Required test | Maximum error / measured result | Status | Artifact |
|---|---:|---|---|
| Gaussian evidence matches direct computation | [ ] | [ ] | [ ] |
| Single-atom RVM analytical solution | [ ] | [ ] | [ ] |
| Equal-cost singleton ranking equivalence | [ ] | [ ] | [ ] |
| Conditional group evidence equals direct difference | [ ] | [ ] | [ ] |
| Logit-direction derivative finite difference/JVP | [ ] | [ ] | [ ] |
| Exact adapter placements / 6,144, 36,864, 73,728 parameter checks | [ ] | [ ] | [ ] |
| Frozen head, splits, resume, MLflow round-trip, CUDA smoke | [ ] | [ ] | [ ] |

**Gate 0 determination:** [PASS/FAIL/INCONCLUSIVE/BLOCKED]. Explain unresolved numerical deviations.

### Gate 1: QNLI architecture oracle (20 full runs)

Each `U(A)=audit_loss(head_only)-audit_loss(A)` uses the **same** frozen-head baseline and a predeclared final-checkpoint training run. Predictions/scores were frozen before audit evaluation.

| Structural score | Pairwise concordance (n pairs) | Spearman 6,144 | Spearman 12,288 | Six-choice top-3 mean U | Margin vs best cheap score |
|---|---:|---:|---:|---:|---:|
| Gradient norm | [ ] | [ ] | [ ] | [ ] | [ ] |
| Independent Fisher | [ ] | [ ] | [ ] | [ ] | [ ] |
| Conditional Newton | [ ] | [ ] | [ ] | [ ] | [ ] |
| Group RVM | [ ] | [ ] | [ ] | [ ] | [ ] |

**Interaction non-additivity (not raw evidence increments):**

| Layer pair | Realized U(i,j)−U(i)−U(j), nats | Predicted E(i,j)−E(i)−E(j), log evidence | Correct sign? |
|---|---:|---:|---|
| (0,1) | [ ] | [ ] | [ ] |
| (5,6) | [ ] | [ ] | [ ] |
| (10,11) | [ ] | [ ] | [ ] |
| (2,9) | [ ] | [ ] | [ ] |

**Stopping/null:** threshold at 95th percentile from 100 calibration nulls `[value]`; activation frequency on 100 fresh nulls `[count/100]`; selected ranks/unused budget on real features `[value]`.

| Gate-1 mandatory criterion | Threshold | Actual value | Status |
|---|---|---|---|
| Informative cost-matched pairs | ≥30 | [ ] | [ ] |
| Group-RVM concordance | ≥0.65 | [ ] | [ ] |
| Concordance over best cheap score | ≥0.05 | [ ] | [ ] |
| Six-choice utility advantage | ≥0.005 nats | [ ] | [ ] |
| Fresh null activations | ≤10 of 100 | [ ] | [ ] |
| Non-additivity sign correct when ≥3 informative pairs | ≥3 correct | [ ] | [ ] |

**Gate 1:** [PASS/FAIL/INCONCLUSIVE/BLOCKED]. Include figures and a note on ties/underpowered comparisons. If gate failed, explain Gate 2 was not run.

### Gate 2: matched-budget comparisons (if Gate 1 passed)

| Method | Mean QNLI CE (seeds) | Mean SST-2 CE (seeds) | Mean accuracy | Actual LoRA params | Mean selection sec | Mean adapter-train sec | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform r1 | [ ] | [ ] | [ ] | 36,864 | [ ] | [ ] | [ ] |
| EVA (native init) | [ ] | [ ] | [ ] | ≤36,864 | [ ] | [ ] | [ ] |
| OPLoRA | [ ] | [ ] | [ ] | ≤36,864 | [ ] | [ ] | [ ] |
| Conditional Newton | [ ] | [ ] | [ ] | ≤36,864 | [ ] | [ ] | [ ] |
| Group RVM | [ ] | [ ] | [ ] | ≤36,864 | [ ] | [ ] | [ ] |
| Uniform r2 | [ ] | [ ] | [ ] | 73,728 | [ ] | [ ] | [ ] |

**Primary competitor:** [best B24 competitor name]. Provide the exact four task×seed loss differences for RVM versus that competitor. Report any failures and integration deviations **in the main table, not a footnote**.

| Gate-2 condition | Threshold | Actual value | Status |
|---|---|---|---|
| Route A aggregate loss advantage | ≥0.010 nats | [ ] | [ ] |
| Route A task-seed wins | ≥3/4 | [ ] | [ ] |
| Route A 90% bootstrap lower bound | >0 | [ ] | [ ] |
| Route B loss vs uniform r2 | ≤ +0.005 nats | [ ] | [ ] |
| Route B loss advantage vs Conditional Newton | ≥0.005 nats | [ ] | [ ] |
| Route B each-task mean excess vs uniform r2 | ≤0.010 nats | [ ] | [ ] |
| Mandatory specificity of Bayesian mechanism | different allocation + improvement | [ ] | [ ] |
| Mandatory calibration overhead ratio | ≤0.25 | [ ] | [ ] |
| Mandatory baseline/data fidelity | all complete | [ ] | [ ] |

**Gate 2:** [PASS/FAIL/INCONCLUSIVE/BLOCKED]. Explicitly identify whether Route A, Route B, both, or neither passed. Plot final loss vs exact LoRA parameters and show layer/module rank heatmaps.

### Failure analysis / critical ablations

[What happens when logdet penalty is removed? Does RVM select a meaningfully different module/rank pattern from conditional Newton? Does the posterior favor sparse support? Was the feature probe informative? How stable are α optimization and ranks? What fraction of overhead is GPU feature collection vs CPU evidence fitting?]

### Resource accounting

[Total GPU-hours for training; GPU-hours for all calibration including SVD/Jacobian and competing methods; warm-head time; peak GPU memory; wall-clock with/without shared setup. Explain one GPU sequential execution, failed runs, reattempts, timeouts, and exact CUDA precision. Do not claim rank reduction equals the same reduction in wall-clock.]

### Deviations, missing evidence, and limitations

[Every deviation from frozen config with commit/time; failed or incomplete baselines; small-sample and two-seed limitations; any validation leakage; pre-existing novelty overlap. Mark exploratory results as exploratory. Report unresolved integration issues rather than inferring favorable performance.]

### Final decision and next experiment

**Machine-evaluated decision:** [value from reports/gates.json].  
**Scientific interpretation:** [why evidence was or was not justified over Conditional Newton].  
**If GO:** propose a narrow 1B–3B causal LM replication with two budgets and three seeds; this is a recommendation, **not** part of this pilot.  
**If NO-GO:** specify which hypothesis failed and why merely adding more priors would not fix it.  
**If INCONCLUSIVE/BLOCKED:** specify the exact missing prerequisite without claiming success.

### Reproduction and evidence index

- `configs/preregistered.yaml` and protocol hashes: [ ]
- MLflow experiment / run IDs: [ ]
- `reports/gates.json`: [ ]
- `reports/run_index.csv`: [ ]
- `reports/architecture_scores.csv`: [ ]
- `reports/figures/`: [ ]
- Exact shell commands: [ ]
- Review bundle hash and path: [ ]
