#!/usr/bin/env bash
# =============================================================================
# Shared launcher used by run_gpt52.sh / run_qwen36_27b.sh / run_qwen35_9b.sh /
# run_qwen35_4b.sh.
#
# Each per-model script does roughly:
#   - source the matching configs/<model>.env (exports MMD_MODEL_NAME, MMD_BASE_URL, ...)
#   - source this _common.sh
#   - call run_mmd_agent
#
# Path configuration (set near the top of THIS file; same for every model):
#   * MMD_BENCH_ROOT:      ABSOLUTE path to ReMMDBench
#   * MMD_SERPER_KEY_FILE: ABSOLUTE path to serper_api.txt
#   * MMD_OUTPUT_ROOT:     where per-model run folders are created
#   * MMD_PYTHON:          which python to call (default: python in current env)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Server-side absolute paths
# -----------------------------------------------------------------------------
# Where the benchmark lives (the folder that contains 001/, 002/, ...).
: "${MMD_BENCH_ROOT:=/path/to/ReMMDBench}"

# Where the 8-line Serper key pool lives (1..4 reserved for other agents,
# 5..8 used by gpt-5.2 / qwen3.6-27b / qwen3.5-9b / qwen3.5-4b respectively).
: "${MMD_SERPER_KEY_FILE:=/path/to/serper_api.txt}"

# -----------------------------------------------------------------------------
# Sensible defaults (rarely need to change).
# -----------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# 使用 mmd conda 环境（已安装 httpx / tqdm / numpy / matplotlib / seaborn / pandas）。
: "${MMD_PYTHON:=/path/to/conda/envs/mmd/bin/python}"
: "${MMD_OUTPUT_ROOT:=${PROJECT_ROOT}/outputs}"
: "${MMD_DATASET_NAME:=remmdbench}"
: "${MMD_RUN_NAME:=}"     # if empty, run_mmd_agent.py picks a timestamped tag
: "${MMD_SAMPLE_FILTER:=}"
: "${MMD_MAX_SAMPLES:=0}" # 0 = run all
: "${MMD_SEED:=42}"

: "${MMD_TEMPERATURE:=0.0}"
: "${MMD_MAX_NEW_TOKENS:=0}"  # 0 表示不限 (payload 也不会带 max_tokens)
: "${MMD_REQUEST_TIMEOUT:=240}"
: "${MMD_RETRY_TIMES:=5}"
: "${MMD_RETRY_INTERVAL:=6}"
: "${MMD_IMAGE_DETAIL:=low}"
: "${MMD_MAX_IMAGES:=10}"
: "${MMD_NUM_WORKERS:=10}"

# -----------------------------------------------------------------------------
# 代理 / no_proxy 配置
#   - Serper 走公网，需要 HTTPS_PROXY。
#   - gpt-5.2 也走公网，必须借助代理。
#   - Qwen 系列走 internal-cluster 内网，通过 NO_PROXY 排除。
#
#   注意：父 shell 中可能继承到末尾带 CRLF 的变量（环境变量 leak from
#   某些 IDE 终端），所以这里显式过滤掉 \r / \n 再 export。
# -----------------------------------------------------------------------------
_strip_cr() { printf '%s' "${1//$'\r'/}" | tr -d '\n'; }

# 强制覆盖（不再用 :=，因为父 shell 可能传入过时的 NO_PROXY_VAL，
# 缺失 Qwen 实际域名 .your-internal-domain.local 会导致 Qwen 请求走代理 -> 502）。
HTTP_PROXY_VAL="${MMD_OVERRIDE_HTTP_PROXY:-http://YOUR_HTTP_PROXY:PORT}"
HTTPS_PROXY_VAL="${MMD_OVERRIDE_HTTPS_PROXY:-http://YOUR_HTTP_PROXY:PORT}"
# 内网域名集合（**必须**包含 Qwen 三个模型所在的 .your-internal-domain.local）：
#   .your-internal-domain.local / .your-internal-domain.local : 内部代理 + 内部服务
#   .your-internal-domain.local          : 三个 Qwen 模型实际所在的内网域
NO_PROXY_VAL="${MMD_OVERRIDE_NO_PROXY:-10.0.0.0/8,100.96.0.0/12,.your-internal-domain.local,.your-internal-domain.local,.your-internal-domain.local}"

HTTP_PROXY_VAL=$(_strip_cr "${HTTP_PROXY_VAL}")
HTTPS_PROXY_VAL=$(_strip_cr "${HTTPS_PROXY_VAL}")
NO_PROXY_VAL=$(_strip_cr "${NO_PROXY_VAL}")

export http_proxy="${HTTP_PROXY_VAL}"
export https_proxy="${HTTPS_PROXY_VAL}"
export HTTP_PROXY="${HTTP_PROXY_VAL}"
export HTTPS_PROXY="${HTTPS_PROXY_VAL}"
export no_proxy="${NO_PROXY_VAL}"
export NO_PROXY="${NO_PROXY_VAL}"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
require_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" || "${value}" == REPLACE_ME_* ]]; then
    echo "ERROR: required variable ${name} is not set (current value: '${value}')." >&2
    echo "Edit the config file in configs/ (or set it inline) and re-run." >&2
    exit 1
  fi
}

require_var MMD_MODEL_NAME
require_var MMD_BASE_URL
require_var MMD_API_KEY
require_var MMD_SERPER_KEY_INDEX
require_var MMD_BENCH_ROOT
require_var MMD_SERPER_KEY_FILE

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
run_mmd_agent() {
  local extra_args=("$@")
  mkdir -p "${MMD_OUTPUT_ROOT}"

  echo "=============================================================="
  echo "Launching unified MMD-Agent"
  echo "  model            : ${MMD_MODEL_NAME}"
  echo "  base_url         : ${MMD_BASE_URL}"
  echo "  bench_root       : ${MMD_BENCH_ROOT}"
  echo "  serper_key_file  : ${MMD_SERPER_KEY_FILE}"
  echo "  serper_key_index : ${MMD_SERPER_KEY_INDEX}"
  echo "  output_root      : ${MMD_OUTPUT_ROOT}"
  echo "=============================================================="

  cd "${PROJECT_ROOT}/eval"
  "${MMD_PYTHON}" run_mmd_agent.py \
    --sampled_root        "${MMD_BENCH_ROOT}" \
    --serper_key_file     "${MMD_SERPER_KEY_FILE}" \
    --serper_key_index    "${MMD_SERPER_KEY_INDEX}" \
    --model_name          "${MMD_MODEL_NAME}" \
    --base_url            "${MMD_BASE_URL}" \
    --api_key             "${MMD_API_KEY}" \
    --answer_path         "${MMD_OUTPUT_ROOT}" \
    --dataset_name        "${MMD_DATASET_NAME}" \
    ${MMD_RUN_NAME:+--run_name "${MMD_RUN_NAME}"} \
    --max_samples         "${MMD_MAX_SAMPLES}" \
    --max_images          "${MMD_MAX_IMAGES}" \
    --seed                "${MMD_SEED}" \
    ${MMD_SAMPLE_FILTER:+--sample_filter "${MMD_SAMPLE_FILTER}"} \
    --temperature         "${MMD_TEMPERATURE}" \
    --max_new_tokens      "${MMD_MAX_NEW_TOKENS}" \
    --request_timeout     "${MMD_REQUEST_TIMEOUT}" \
    --retry_times         "${MMD_RETRY_TIMES}" \
    --retry_interval      "${MMD_RETRY_INTERVAL}" \
    --image_detail        "${MMD_IMAGE_DETAIL}" \
    --num_workers         "${MMD_NUM_WORKERS}" \
    "${extra_args[@]}"
}
