#!/usr/bin/env bash
# EXPLORATORY Gate 2 despite the failed Gate 1 (explicitly authorised by the user, 2026-10-18). It runs the 24
# frozen Gate-2 configurations in the isolated namespace `gate2_exploratory`; the preregistered decision (NO_GO,
# reports/v2/gates.json) is not changed. Intended for tmux session `rvm_gate2_exploratory` on the GPU host.
# Resumable: completed, provenance-verified heads/probes/runs are reused, never silently replaced.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export RVM_RUN_NAMESPACE=gate2_exploratory
GATE1_RECORD=artifacts/v2/gate1/gate1.json
REQUIRED_COMMIT=4c54b1174b7501f7843407eaf042e8398ead6c2b  # corrected (post-audit) v2 implementation
OV=(--exploratory-override-gate1 --gate1-record "$GATE1_RECORD")
mkdir -p logs
LOG="logs/pilot_${RVM_RUN_NAMESPACE}.log"
cli() { uv run --frozen python -m rvm_lora.cli "$@" --config configs/preregistered.yaml; }

set +e
(
  set -euo pipefail
  echo "[run_gate2_exploratory] start $(date -u +%FT%TZ) HEAD=$(git rev-parse HEAD)"
  git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD || { echo "HEAD does not contain $REQUIRED_COMMIT"; exit 3; }
  uv run --frozen python -m rvm_lora.integrity snapshot "artifacts/$RVM_RUN_NAMESPACE/integrity/v2_before.json" \
    --namespace v2 --keep
  cli env-check > /dev/null
  cli prepare                 # reuses the frozen split manifests/encodings (no new split identities)
  cli gate0                   # full test suite incl. CUDA smoke on this commit (namespace-local record)
  cli freeze                  # provenance record; refuses if protocol files or split manifests changed
  cli train-b "${OV[@]}"      # 4 heads + 4 probes (QNLI/42 verified against v2) + 24 runs, one GPU job at a time
  cli eval-gate2 "${OV[@]}" > /dev/null
  echo "[run_gate2_exploratory] experiments done $(date -u +%FT%TZ)"
) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[run_gate2_exploratory] exit code $rc" | tee -a "$LOG"
exit "$rc"
