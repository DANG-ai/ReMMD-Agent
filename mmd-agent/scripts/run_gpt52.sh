#!/usr/bin/env bash
# Run the unified MMD-Agent with the GPT-5.2 backend.
#
# Usage on the server:
#   1. Edit configs/gpt52.env (model URL/API key is already filled in for GPT-5.2;
#      adjust paths if needed).
#   2. Edit MMD_BENCH_ROOT and MMD_SERPER_KEY_FILE in scripts/_common.sh
#      (those are the same for every model; only set once).
#   3. bash scripts/run_gpt52.sh
#
# Optional extra arguments are forwarded to run_mmd_agent.py, e.g.:
#   bash scripts/run_gpt52.sh --max_samples 5 --num_workers 4

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# Load model-specific config (MODEL_NAME / BASE_URL / API_KEY / SERPER_KEY_INDEX / ...)
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/gpt52.env"

# Load shared launcher (defines run_mmd_agent ...)
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

run_mmd_agent "$@"
