#!/usr/bin/env bash
# Full frozen gate chain; intended to run inside tmux session `rvm_pilot` on the GPU host.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs
set +e
uv run --frozen python -m rvm_lora.cli all --config configs/preregistered.yaml 2>&1 | tee -a logs/pilot.log
rc=${PIPESTATUS[0]}
set -e
echo "[run_remote] exit code $rc" | tee -a logs/pilot.log
exit "$rc"
