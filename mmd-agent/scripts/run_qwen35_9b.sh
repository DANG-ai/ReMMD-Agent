#!/usr/bin/env bash
# Run the unified MMD-Agent with the Qwen3.5-9B backend.
#
# Usage on the server:
#   1. Edit configs/qwen35_9b.env and fill in MMD_BASE_URL and MMD_API_KEY.
#   2. Make sure MMD_BENCH_ROOT and MMD_SERPER_KEY_FILE in scripts/_common.sh
#      point to the right absolute paths.
#   3. bash scripts/run_qwen35_9b.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/qwen35_9b.env"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

run_mmd_agent "$@"
