#!/usr/bin/env bash
# Snapshot the progress of all four parallel ReMMDBench evaluation runs.
#
# Usage:
#   bash scripts/monitor.sh           # one-shot snapshot
#   bash scripts/monitor.sh --watch   # refresh every 30s
#   bash scripts/monitor.sh --watch 60  # refresh every 60s

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

print_status() {
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] T2-Agent ReMMDBench progress"
    echo "============================================================"
    echo
    echo "--- screen sessions ---"
    screen -ls 2>/dev/null | grep "mmd_" || echo "(no mmd_ screens found)"
    echo

    for PROV in gpt qwen3_6_27b qwen3_5_9b qwen3_5_4b; do
        RUN_DIR=$(ls -td "${PROJECT_ROOT}/artifacts/runs/${PROV}_"*/ 2>/dev/null | head -1)
        if [ -z "$RUN_DIR" ]; then
            echo "[$PROV] no run directory yet"
            echo
            continue
        fi
        RUN_NAME=$(basename "$RUN_DIR")
        DONE=$(ls "${RUN_DIR}/details/" 2>/dev/null | wc -l)
        CALL_LOG="${PROJECT_ROOT}/records/${RUN_NAME}/llm_calls.jsonl"
        CALLS=$(wc -l < "$CALL_LOG" 2>/dev/null || echo 0)
        LATEST_LOG=$(ls -t "${PROJECT_ROOT}/logs/${PROV}_"*.log 2>/dev/null | head -1)
        LAST=""
        if [ -n "$LATEST_LOG" ]; then
            LAST=$(grep -E "DONE|FATAL|FINISHED" "$LATEST_LOG" 2>/dev/null | tail -1)
        fi
        printf "[%-12s] %s\n" "$PROV" "$RUN_NAME"
        printf "    samples done: %s/500   |  llm/tool events: %s\n" "$DONE" "$CALLS"
        [ -n "$LAST" ] && printf "    last: %s\n" "$LAST"
        echo
    done
}

WATCH=false
INTERVAL=30
while [ $# -gt 0 ]; do
    case "$1" in
        --watch)
            WATCH=true
            shift
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                INTERVAL="$1"
                shift
            fi
            ;;
        *) shift ;;
    esac
done

if $WATCH; then
    while true; do
        clear
        print_status
        sleep "$INTERVAL"
    done
else
    print_status
fi
