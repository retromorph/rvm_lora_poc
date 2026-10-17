#!/usr/bin/env bash
# Rebuild gates/report/bundle from logged artifacts (no training).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv run --frozen python -m rvm_lora.cli report --config configs/preregistered.yaml
ls -lh reports/FINAL_REPORT.md reports/gates.json reports/review_bundle.zip
sha256sum reports/review_bundle.zip
