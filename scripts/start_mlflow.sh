#!/usr/bin/env bash
# Start (or reuse) the MLflow tracking server in tmux session `rvm_mlflow`, bound to 127.0.0.1 only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p mlruns/artifacts logs
if curl -fsS http://127.0.0.1:5000/health >/dev/null 2>&1; then
  echo "MLflow already responding on 127.0.0.1:5000 (reusing)"; exit 0
fi
if ss -ltn | grep -q ':5000 '; then
  echo "port 5000 is owned by another process; refusing to start a second server" >&2; exit 1
fi
tmux new-session -d -s rvm_mlflow \
  "cd '$PWD' && uv run --frozen mlflow server \
     --backend-store-uri sqlite:///mlruns/mlflow.db \
     --default-artifact-root file://$PWD/mlruns/artifacts \
     --host 127.0.0.1 --port 5000 > logs/mlflow.log 2>&1"
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:5000/health >/dev/null 2>&1; then echo "MLflow responding"; tmux ls; exit 0; fi
  sleep 1
done
echo "MLflow did not come up; see logs/mlflow.log" >&2; exit 1
