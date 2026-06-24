#!/usr/bin/env bash
# Entrypoint for the Qwen3.5-4B backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/qwen3_5_4b.yaml"

CONDA_ENV="${T2AGENT_CONDA_ENV:-mmd}"
MAX_WORKERS="${T2AGENT_MAX_WORKERS:-10}"

HTTP_PROXY_VAL="${MY_HTTP_PROXY:-http://YOUR_HTTP_PROXY:PORT}"
HTTPS_PROXY_VAL="${MY_HTTPS_PROXY:-http://YOUR_HTTP_PROXY:PORT}"
NO_PROXY_VAL="${MY_NO_PROXY:-10.0.0.0/8,100.96.0.0/12,.your-internal-domain.local,.your-internal-domain.local,YOUR_GPT_HOST}"
export http_proxy="${HTTP_PROXY_VAL}" https_proxy="${HTTPS_PROXY_VAL}" no_proxy="${NO_PROXY_VAL}"
export HTTP_PROXY="${HTTP_PROXY_VAL}" HTTPS_PROXY="${HTTPS_PROXY_VAL}" NO_PROXY="${NO_PROXY_VAL}"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
    conda run -n "${CONDA_ENV}" --no-capture-output python "scripts/run_remmdbench.py" \
        --config "${CONFIG_FILE}" \
        --max-workers "${MAX_WORKERS}" \
        "$@"
else
    python "scripts/run_remmdbench.py" \
        --config "${CONFIG_FILE}" \
        --max-workers "${MAX_WORKERS}" \
        "$@"
fi
