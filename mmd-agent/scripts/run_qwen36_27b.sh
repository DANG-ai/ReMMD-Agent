#!/usr/bin/env bash
# Run the unified MMD-Agent with the Qwen3.6-27B backend.
#
# Usage on the server:
#   1. Edit configs/qwen36_27b.env and fill in MMD_BASE_URL and MMD_API_KEY.
#   2. Make sure MMD_BENCH_ROOT and MMD_SERPER_KEY_FILE in scripts/_common.sh
#      point to the right absolute paths.
#   3. bash scripts/run_qwen36_27b.sh
#
# Optional extra arguments are forwarded to run_mmd_agent.py, e.g.:
#   bash scripts/run_qwen36_27b.sh --max_samples 5 --num_workers 4

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/qwen36_27b.env"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

run_mmd_agent "$@"
