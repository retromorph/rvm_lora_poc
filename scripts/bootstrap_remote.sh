#!/usr/bin/env bash
# One-time unprivileged setup on the GPU host: uv (user-local), Python 3.11, locked environment, checks.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv sync --frozen
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --frozen python -m rvm_lora.cli all --dry-run
