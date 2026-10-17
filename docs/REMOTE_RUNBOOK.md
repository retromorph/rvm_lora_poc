# A4000 / SSH / tmux / uv / MLflow runbook

Use this from a trusted machine with SSH access. Commands with `$REMOTE_HOST` and `$REMOTE_PROJECT_DIR` require local environment values or SSH config aliases. **Do not check credentials into Git.** The coding agent must first verify host OS/driver, storage/permissions and whether the GPU is shared; no concurrent heavy GPU runs.

## 1. Inspect the remote GPU and filesystem

```bash
# Local computer (REMOTE_HOST is an SSH-config alias, not an IP embedded in source):
ssh "$REMOTE_HOST" 'hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h "$HOME"; tmux -V; uv --version'
# Optional transfer after a local checkout:
rsync -az --exclude='.git' --exclude='.venv' --exclude='mlruns' --exclude='data' --exclude='logs' ./ "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"
ssh "$REMOTE_HOST"
cd "$REMOTE_PROJECT_DIR"
```

If `uv` is not installed, install as an unprivileged user following https://docs.astral.sh/uv/getting-started/installation/; do not overwrite system Python. Prefer `uv python install 3.11`; `uv sync` builds the project environment once the agent has produced `pyproject.toml` and `uv.lock`. If CUDA torch fails to load, inspect `nvidia-smi`, supported wheel CUDA variant, `torch.version.cuda`, `torch.cuda.is_available()`; do not guess compatibility. Pin the selected CUDA PyTorch wheel via a supported `uv` index (https://docs.astral.sh/uv/guides/integration/pytorch/).

## 2. Start MLflow inside a detached remote tmux session

```bash
# Run on REMOTE, from repository root; use a path without spaces.
cd "$REMOTE_PROJECT_DIR"
mkdir -p mlruns/artifacts logs
# Verify no existing server/tmux session already owns port 5000.
ss -ltn | grep ':5000' || true
tmux new-session -d -s rvm_mlflow \
  "cd '$PWD' && uv run mlflow server \
     --backend-store-uri sqlite:///mlruns/mlflow.db \
     --default-artifact-root file://$PWD/mlruns/artifacts \
     --host 127.0.0.1 --port 5000 \
     > logs/mlflow.log 2>&1"
sleep 3
curl -fsS http://127.0.0.1:5000/ >/dev/null && echo 'MLflow responding'
tmux ls
```

The database URI `sqlite:///mlruns/mlflow.db` is **relative to the server's working directory**. `file://$PWD/mlruns/artifacts` is the absolute local artifact root; both code and tracking server run on the same remote machine. Monitor `tail -f logs/mlflow.log` and verify read/write using a small MLflow smoke run before starting GPU jobs. If an existing server is active, reuse it rather than launching a second one.

**Network security:** Do **not** replicate a public `--host 0.0.0.0 --allowed-hosts "*" --cors-allowed-origins "*"` configuration without an authenticated, controlled network setup. The pilot requires **none** of `sudo ufw allow 5000/tcp`, `sudo ufw enable`, or opening public HTTP; enabling UFW remotely can inadvertently sever SSH connectivity. Leave UFW unchanged. MLflow's default local-only security behavior is intended here.

## 3. Browse MLflow via an SSH tunnel

```bash
# Run on LOCAL computer in a separate terminal:
ssh -N -L 15000:127.0.0.1:5000 "$REMOTE_HOST"
# Open http://127.0.0.1:15000 in the local browser.
# On REMOTE, where the training code executes:
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

If the remote SSH port is non-standard, configure it in `~/.ssh/config`; do not assume port 22. No direct public MLflow URL is needed.

## 4. Launch and monitor the run manager

The coding agent shall create `scripts/run_remote.sh`, a bash script with `set -euo pipefail`, explicit `cd` to repo root, `MLFLOW_TRACKING_URI` export, a `uv run python -m rvm_lora.cli all --config configs/preregistered.yaml` invocation, and stdout/stderr log capture preserving the Python exit code.

```bash
# On REMOTE:
cd "$REMOTE_PROJECT_DIR"
uv sync --frozen
uv run pytest -q
uv run python -m rvm_lora.cli env-check
uv run python -m rvm_lora.cli all --dry-run
tmux new-session -d -s rvm_pilot "cd '$PWD' && bash scripts/run_remote.sh"
tmux attach -t rvm_pilot           # detach without terminating: Ctrl-b then d
# Later:
tmux ls
tail -n 100 logs/pilot.log
nvidia-smi
uv run python -m rvm_lora.cli status
```

Use exactly one GPU training session and persist run-index/status after each completed experiment. If SSH disconnects, the `tmux` session continues; reconnect and reattach. If the run exits, `tmux` may close, but the logs and MLflow statuses remain. Any unexpected crash requires preserving run artifacts and an explicit resume/retry record.

## 5. Resource controls and recovery

- A4000 VRAM is 16 GB: max sequence 128, microbatch 8, gradient accumulation 4. If OOM, reduce microbatch to 4 and accumulation to 8, with fixed effective batch 32. Do not silently change batch/evaluation precision per method.
- Keep selection features/Gram matrices and SVD metadata on disk; store checkpoints as tiny LoRA adapters, not full RoBERTa copies. Preserve shared head checkpoints and pin model download to one revision.
- Run numerical evidence optimization on CPU float64. Prefer Cholesky/Gram math; caching the 512×192 feature system avoids repeated network passes during structural search.
- MLflow SQLite works for a single writer/training manager. Back up `mlruns/mlflow.db` **after** runs end (or use safe SQLite backup); do not copy a hot database blindly for the review bundle. Do not store secrets/large datasets inside MLflow.
- Check disk free space before each stage. Set sensible Hugging Face cache location, e.g. `$HOME/.cache/huggingface`; do not delete shared caches.

## 6. Export review artifacts

```bash
# On REMOTE, after the agent writes FINAL_REPORT and manifests:
uv run python -m rvm_lora.cli report --config configs/preregistered.yaml
ls -lh reports/FINAL_REPORT.md reports/gates.json reports/review_bundle.zip
# On LOCAL:
scp "$REMOTE_HOST:$REMOTE_PROJECT_DIR/reports/review_bundle.zip" .
```

The review archive contains report, tables, plots, code revision/config hashes, MLflow run IDs, environment and deviations; **no model weights, datasets, access keys or databases**. A reviewer should be able to inspect the full decision without contacting the tracking service.

## 7. Shutdown

Leave the MLflow server running only as long as desired. To stop it cleanly on remote: `tmux kill-session -t rvm_mlflow` *after* training finishes. Do not kill other users' tmux sessions or processes.
