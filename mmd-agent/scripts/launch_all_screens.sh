#!/usr/bin/env bash
# 在 4 个独立 screen 中并发启动 4 个模型的全量评测。
# 每个 screen 内部：
#   - 进入 mmd conda 环境
#   - 由 _common.sh 自动配置代理（已过滤 CRLF）
#   - 启动 run_<model>.sh，10 并发跑 ReMMDBench 500 个样本
#   - 输出同时落盘到 outputs/logs/<model>.log，方便后续监控
#
# 用法：
#   bash scripts/launch_all_screens.sh
#
# 监控某个模型：
#   screen -r mmd_gpt52     # 或 mmd_qwen36_27b / mmd_qwen35_9b / mmd_qwen35_4b
# 退出 screen（不杀任务）：Ctrl-A, D
#
# 一次性查看所有日志：
#   tail -f /path/to/ReMMD-Agent/mmd-agent/outputs/logs/*.log

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"

# 时间戳作为 run_name，方便后面 resume 同一个 run。
TS="$(date +%Y%m%d_%H%M%S)"
echo "Launching all 4 models with run_name suffix _${TS}"

start_one() {
  local screen_name="$1"
  local script="$2"
  local log_file="$3"
  local run_name="$4"

  # 先确保旧的同名 screen 已经退出，避免冲突
  if screen -ls | grep -q "[^a-zA-Z0-9_]${screen_name}[^a-zA-Z0-9_]"; then
    echo "  warning: a screen named '${screen_name}' already exists, leaving it untouched."
    echo "           detach with Ctrl-A,D and resume with: screen -r ${screen_name}"
    return
  fi
  screen -dmS "${screen_name}" bash -lc "
    set -e
    source /path/to/conda/etc/profile.d/conda.sh
    conda activate mmd
    export MMD_RUN_NAME='${run_name}'
    cd '${PROJECT_ROOT}'
    {
      echo '== START $(date -Iseconds) =='
      bash '${script}'
      echo '== DONE  $(date -Iseconds) =='
    } 2>&1 | tee '${log_file}'
    echo
    echo '[screen finished] press Ctrl-A,D to detach without closing.'
    bash
  "
  echo "  started: screen='${screen_name}'  script='${script}'  log='${log_file}'"
}

start_one mmda_gpt52       "${HERE}/run_gpt52.sh"        "${LOG_DIR}/gpt52.log"        "full_${TS}"
start_one mmda_qwen36_27b  "${HERE}/run_qwen36_27b.sh"   "${LOG_DIR}/qwen36_27b.log"   "full_${TS}"
start_one mmda_qwen35_9b   "${HERE}/run_qwen35_9b.sh"    "${LOG_DIR}/qwen35_9b.log"    "full_${TS}"
start_one mmda_qwen35_4b   "${HERE}/run_qwen35_4b.sh"    "${LOG_DIR}/qwen35_4b.log"    "full_${TS}"

echo
echo "All four screens dispatched. Verify with:"
echo "  screen -ls | grep mmda_"
echo "  tail -F ${LOG_DIR}/gpt52.log"
echo "  tail -F ${LOG_DIR}/qwen36_27b.log"
echo "  tail -F ${LOG_DIR}/qwen35_9b.log"
echo "  tail -F ${LOG_DIR}/qwen35_4b.log"
echo "Attach: screen -r mmda_gpt52  (Ctrl-A,D to detach)"
