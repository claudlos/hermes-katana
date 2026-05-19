#!/usr/bin/env bash
# Thermal watchdog — pauses llama-server when GPU temp hits the ceiling,
# resumes when it drops back to a safe floor. Prevents the local qwen
# orchestrator from cooking the 3050 Ti during overnight runs.
#
# Uses SIGSTOP/SIGCONT (pause, not kill). llama-server's connections to
# run_shard.py stall until we resume, so no data loss — just a pause.
#
# Usage:
#   scripts/thermal_watchdog.sh [ceiling_c] [floor_c] [poll_sec]
#   scripts/thermal_watchdog.sh 81 74 5           # defaults
#
# Log: /tmp/thermal_watchdog.log

set -u

CEILING="${1:-81}"      # pause if current temp >= this
FLOOR="${2:-74}"        # resume when current temp <= this
POLL="${3:-5}"          # seconds between samples

LOG=/tmp/thermal_watchdog.log
STATE="ok"              # "ok" | "paused"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "thermal_watchdog started — ceiling=${CEILING}C floor=${FLOOR}C poll=${POLL}s"

cleanup() {
    # On exit, always SIGCONT — don't leave llama-server paused.
    if [[ "$STATE" == "paused" ]]; then
        log "exiting — sending SIGCONT to resume llama-server"
        pkill -CONT -f llama-server 2>/dev/null || true
    fi
    log "thermal_watchdog stopped"
    exit 0
}
trap cleanup INT TERM

while true; do
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$temp" ]] || ! [[ "$temp" =~ ^[0-9]+$ ]]; then
        log "couldn't read GPU temp — backing off 30s"
        sleep 30
        continue
    fi

    if [[ "$STATE" == "ok" && "$temp" -ge "$CEILING" ]]; then
        log "PAUSE llama-server — temp=${temp}C >= ceiling=${CEILING}C"
        pkill -STOP -f llama-server 2>/dev/null && STATE="paused"
    elif [[ "$STATE" == "paused" && "$temp" -le "$FLOOR" ]]; then
        log "RESUME llama-server — temp=${temp}C <= floor=${FLOOR}C"
        pkill -CONT -f llama-server 2>/dev/null && STATE="ok"
    fi
    sleep "$POLL"
done
