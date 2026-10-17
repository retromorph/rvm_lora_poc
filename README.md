# RVM-LoRA: decision-oriented proof of concept

This repository specifies a **pre-registered feasibility test**, not a claimed research result. Its purpose is to decide whether conditional RVM marginal evidence adds practical value for joint LoRA rank/module selection beyond cheaper, equally informed structural selectors.

## Start here

1. Read [`AGENTS.md`](AGENTS.md) — requirements and agent autonomy.
2. Read [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) and [`configs/preregistered.yaml`](configs/preregistered.yaml) — frozen experiments, budgets and Go/No-Go rules.
3. Read [`docs/METHODS.md`](docs/METHODS.md) — estimators and mathematical tests.
4. Read [`docs/ENGINEERING.md`](docs/ENGINEERING.md) — code architecture and acceptance tests.
5. Read [`docs/REMOTE_RUNBOOK.md`](docs/REMOTE_RUNBOOK.md) — `uv`, A4000, SSH, `tmux` and local-on-remote MLflow.
6. Read [`prompts/CODING_AGENT_PROMPT.md`](prompts/CODING_AGENT_PROMPT.md) — complete instruction to a coding agent.
7. Complete [`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md) into `reports/FINAL_REPORT.md` and produce reviewable machine-readable artifacts.

## Experiment synopsis

- **Gate 0:** solver correctness, numerical and integration tests, A4000 smoke test.
- **Gate 1 (A):** 20 pre-defined QNLI LoRA architectures trained on one seed; compare whether RVM scores predict *actual* retraining gains better than cheap baselines; test null stopping.
- **Gate 2 (B):** only if Gate 1 passes, 6 methods × 2 tasks × 2 seeds = 24 runs. Same final parameter budget for the primary comparisons; test quality or compression and calibration overhead.
- **Maximum main training runs:** 44, plus classifier-head warm-starts, probes, null simulations and tests. A minimal one- or two-step CUDA smoke test is separate from these full training runs.

The agent must not change gate thresholds after inspecting audit or GLUE validation metrics. A negative result is a valid final outcome.

## Resources and access

The intended machine is a single NVIDIA RTX A4000 16 GB accessible via SSH. The agent is expected to create the code and environment with `uv` and run tracking on the **remote machine** using MLflow with SQLite and local artifacts. View the MLflow UI by forwarding the remote `127.0.0.1:5000` port over SSH; **do not open port 5000 publicly**. Supply an SSH host alias and working directory out of band; do not put credentials in Git.

## Deliverables for independent review

`reports/FINAL_REPORT.md`, `reports/gates.json`, `reports/run_index.csv`, `reports/architecture_scores.csv`, `reports/figures/`, an immutable protocol snapshot and hashes, MLflow run IDs, machine/environment manifest, and an archive `reports/review_bundle.zip`. A completed report distinguishes `PASS`, `FAIL`, `INCONCLUSIVE` and `BLOCKED` and never substitutes missing runs with invented results.

This starting package contains **instructions and protocol**, not a working implementation or executed results. The coding agent must implement and run them.
