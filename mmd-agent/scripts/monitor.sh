#!/usr/bin/env bash
# 4 个模型并发评测的实时进度监控
# 用法：
#   bash scripts/monitor.sh         # 单次打印一次
#   watch -n 30 bash scripts/monitor.sh   # 每 30 秒刷新一次

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

declare -A DIRS=(
  [gpt52]=gpt-5.2
  [qwen36_27b]=qwen3.6-27b
  [qwen35_9b]=qwen3.5-9b
  [qwen35_4b]=qwen3.5-4b
)

echo "==============================================="
echo "  MMD-Agent monitor @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================="
echo

printf "%-14s %-10s %-8s %-8s %-12s\n" "model" "done" "retry" "fb" "speed"
printf "%-14s %-10s %-8s %-8s %-12s\n" "-----" "----" "-----" "--" "-----"
for m in gpt52 qwen36_27b qwen35_9b qwen35_4b; do
  f="${ROOT}/outputs/logs/${m}.log"
  d="${ROOT}/outputs/${DIRS[$m]}"
  ckpt=$(ls -d ${d}/full_*/remmdbench.jsonl 2>/dev/null | head -1)
  done_count=$(wc -l < "$ckpt" 2>/dev/null || echo 0)
  retry=$(grep -c '\[Retry' "$f" 2>/dev/null)
  fb=$(grep -c '\[FB\]' "$f" 2>/dev/null)
  rate=$(grep -oE '[0-9]+\.[0-9]+s/it' "$f" 2>/dev/null | tail -1)
  printf "%-14s %-10s %-8s %-8s %-12s\n" "$m" "${done_count}/500" "$retry" "$fb" "${rate:-?}"
done

echo
echo "Active screens:"
screen -ls | grep mmda_ | sed 's/^/  /'

echo
echo "Latest sample completions (last 3 per model):"
for m in gpt52 qwen36_27b qwen35_9b qwen35_4b; do
  f="${ROOT}/outputs/logs/${m}.log"
  echo "  -- ${m} --"
  grep -E 'verdict pred' "$f" 2>/dev/null | tail -3 | sed 's/^/    /'
done
