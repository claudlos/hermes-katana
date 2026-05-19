#!/usr/bin/env bash
# CPU thermal watchdog — pauses the fleet supervisor when x86_pkg_temp
# crosses the ceiling, resumes when it drops to the floor. Prevents the
# kind of sustained 95C+ cooking that killed the laptop last time.
#
# Companion to scripts/thermal_watchdog.sh (which guards the GPU only).
#
# Usage:
#   scripts/cpu_watchdog.sh [ceiling_c] [floor_c] [poll_sec]
#   scripts/cpu_watchdog.sh 90 80 5    # defaults (below 95°C TJMax)
#
# Log: /tmp/cpu_watchdog.log

set -u

CEILING="${1:-80}"      # SIGSTOP supervisor if current temp >= this
FLOOR="${2:-70}"        # SIGCONT when temp <= this
POLL="${3:-5}"

LOG=/tmp/cpu_watchdog.log
STATE="ok"
SUP_PID=""

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

read_pkg_temp() {
    # x86_pkg_temp is zone3 on this box; fall back to any zone with the right type.
    for z in /sys/class/thermal/thermal_zone*; do
        [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ] || continue
        v=$(cat "$z/temp" 2>/dev/null)
        echo $(( v / 1000 ))
        return
    done
    echo "?"
}

find_supervisor() {
    # Target only the Python fleet.py process. /tmp/fleet/fleet.pid is the
    # canonical source — fleet.py writes its own pid there on launch. Fall
    # back to /proc/<pid>/comm matching "python3" so we never pause a bash
    # wrapper (which leaves the Python supervisor running unprotected).
    if [ -f /tmp/fleet/fleet.pid ]; then
        pid=$(cat /tmp/fleet/fleet.pid 2>/dev/null)
        # Verify pid exists AND is a python process (not a stale pidfile).
        if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
            comm=$(cat "/proc/$pid/comm" 2>/dev/null)
            if [[ "$comm" == python* ]]; then
                echo "$pid"
                return
            fi
        fi
    fi
    # Fallback: scan for a process whose /proc/<pid>/comm starts with
    # "python" AND has scripts/fleet.py in its cmdline.
    for pid in $(pgrep -f 'scripts/fleet\.py'); do
        comm=$(cat "/proc/$pid/comm" 2>/dev/null)
        if [[ "$comm" == python* ]]; then
            echo "$pid"
            return
        fi
    done
}

log "cpu_watchdog started — ceiling=${CEILING}C floor=${FLOOR}C poll=${POLL}s"

# Ground truth for "is the supervisor currently stopped" — check /proc/$PID/status
# rather than a shell variable. Fixes the stale-STATE bug where a missed resume
# left supervisor SIGSTOPped for 22 min (v8d incident, 2026-04-20).
is_stopped() {
    local pid="$1"
    [ -z "$pid" ] && return 1
    local st
    st=$(awk '/^State:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    [ "$st" = "T" ]
}

cleanup() {
    if [ -n "$SUP_PID" ] && is_stopped "$SUP_PID"; then
        log "exiting — SIGCONT supervisor pid=$SUP_PID"
        kill -CONT "$SUP_PID" 2>/dev/null || true
    fi
    log "cpu_watchdog stopped"
    exit 0
}
trap cleanup INT TERM

# Heartbeat every N polls so we can tell the watchdog is actually running
HEARTBEAT_EVERY=12  # 12 × 5s = 1 min
ticks=0
last_pause_ts=0

while true; do
    ticks=$((ticks + 1))
    temp=$(read_pkg_temp)
    if [ "$temp" = "?" ]; then
        log "couldn't read pkg temp — backing off"
        sleep 30; continue
    fi

    SUP_PID=$(find_supervisor)
    if [ -z "$SUP_PID" ]; then
        sleep "$POLL"; continue
    fi

    stopped=0; is_stopped "$SUP_PID" && stopped=1
    now=$(date +%s)

    if [ "$stopped" = 0 ] && [ "$temp" -ge "$CEILING" ]; then
        log "PAUSE supervisor pid=$SUP_PID — temp=${temp}C >= ceiling=${CEILING}C"
        kill -STOP "$SUP_PID" 2>/dev/null
        last_pause_ts=$now
        for p in $(pgrep -f run_agent_shard.py); do
            kill -STOP "$p" 2>/dev/null || true
        done
    elif [ "$stopped" = 1 ] && [ "$temp" -le "$FLOOR" ]; then
        log "RESUME supervisor pid=$SUP_PID — temp=${temp}C <= floor=${FLOOR}C (normal)"
        kill -CONT "$SUP_PID" 2>/dev/null
        for p in $(pgrep -f run_agent_shard.py); do
            kill -CONT "$p" 2>/dev/null || true
        done
        last_pause_ts=0
    elif [ "$stopped" = 1 ] && [ "$last_pause_ts" -gt 0 ] \
         && [ $((now - last_pause_ts)) -ge 120 ] \
         && [ "$temp" -le $((FLOOR + 5)) ]; then
        # Safety net: if still STOPPED after 2 min and temp is within 5°C
        # of floor (i.e. plenty cool), force-resume. Covers the case where
        # a previous resume was missed (stale shell STATE, signal lost).
        log "FORCE-RESUME pid=$SUP_PID — stopped ${now}-${last_pause_ts}s with temp=${temp}C (<=floor+5)"
        kill -CONT "$SUP_PID" 2>/dev/null
        for p in $(pgrep -f run_agent_shard.py); do
            kill -CONT "$p" 2>/dev/null || true
        done
        last_pause_ts=0
    fi

    if [ $((ticks % HEARTBEAT_EVERY)) = 0 ]; then
        log "heartbeat — sup=$SUP_PID stopped=$stopped temp=${temp}C"
    fi

    sleep "$POLL"
done
