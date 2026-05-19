#!/usr/bin/env bash
# fleet_monitor.sh — health-check + auto-heal for an active fleet.py supervisor.
# Intended to run every 30 min via user cron. Idempotent; safe to run even when
# no fleet is active.
#
# Responsibilities:
#  1. If /tmp/fleet/fleet.pid is stale and the supervisor.log shows an ongoing
#     run (last "fleet launch" has no matching "fleet exit"), relaunch fleet.py
#     against whatever spec was last launched.
#  2. If supervisor is T (SIGSTOPped) for >2 min AND CPU temp is safe,
#     SIGCONT it. Also ensures cpu_watchdog is alive.
#  3. SIGKILL any run_agent_shard.py worker running >20 min (likely hung on
#     an unresponsive provider API).
#  4. Scan /tmp/fleet/atk_*.log for rate-limit markers in the last 30 min.
#     If claude_cli_* hits ≥3, log a ROTATE_HINT so a human can swap providers.
#  5. Append a one-line status (done/queued/rate/temp) to /tmp/fleet_monitor.log.
#
# Usage from cron (every 30 min):
#   */30 * * * * /home/carlos/Documents/Code/katana-proving-ground/scripts/fleet_monitor.sh

set -u

ROOT=/home/carlos/Documents/Code/katana-proving-ground
VENV=$ROOT/.venv/bin/python
PID_FILE=/tmp/fleet/fleet.pid
SUP_LOG=/tmp/fleet/supervisor.log
WD_LOG=/tmp/cpu_watchdog.log
MON_LOG=/tmp/fleet_monitor.log
THERMAL=/sys/class/thermal/thermal_zone3/temp
WORKER_MAX_AGE_SEC=1200   # 20 min
PAUSED_MAX_AGE_SEC=120    # 2 min
TEMP_SAFE_C=75

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MON_LOG"; }

# ---------------------------------------------------------------------------
# 0. Require the basic dirs / files — if supervisor.log doesn't exist, there's
# nothing to monitor yet. Silent no-op.
[ -f "$SUP_LOG" ] || exit 0

# ---------------------------------------------------------------------------
# 1. Supervisor liveness + auto-relaunch
last_launch=$(grep -n 'fleet launch' "$SUP_LOG" | tail -1 | cut -d: -f1)
last_exit=$(grep -n 'fleet exit'   "$SUP_LOG" | tail -1 | cut -d: -f1)

# "should be running" = last launch is later than last exit (or no exit yet)
should_run=0
if [ -n "$last_launch" ] && { [ -z "$last_exit" ] || [ "$last_launch" -gt "$last_exit" ]; }; then
    should_run=1
fi

sup_alive=0
sup_pid=""
if [ -f "$PID_FILE" ]; then
    sup_pid=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$sup_pid" ] && kill -0 "$sup_pid" 2>/dev/null; then
        sup_alive=1
    fi
fi

if [ "$should_run" = 1 ] && [ "$sup_alive" = 0 ]; then
    # Recover which spec was last launched from supervisor.log — fleet.py's
    # own stdout writes `launch --spec` into a wrapper log file per-launch;
    # we infer from the most recent v11_launch.stdout / fleet_v<N>.json pair.
    spec=$(ls -t /tmp/fleet/v*_launch.stdout 2>/dev/null | head -1 \
            | sed -E 's#.*/v(.+)_launch.stdout#\1#')
    spec_path="$ROOT/scripts/fleet_v${spec}.json"
    if [ -f "$spec_path" ]; then
        say "AUTO_RELAUNCH spec=$spec_path (supervisor gone but last run was still active)"
        nohup "$VENV" "$ROOT/scripts/fleet.py" launch --spec "$spec_path" \
              >"/tmp/fleet/auto_relaunch_$(date +%s).stdout" 2>&1 &
        disown
    else
        say "AUTO_RELAUNCH_SKIPPED no spec found (looked for $spec_path)"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Stuck-SIGSTOP recovery + watchdog liveness
if [ "$sup_alive" = 1 ]; then
    state=$(awk '/^State:/{print $2}' "/proc/$sup_pid/status" 2>/dev/null)
    temp=$(( $(cat "$THERMAL" 2>/dev/null || echo 0) / 1000 ))
    if [ "$state" = "T" ]; then
        # How long has supervisor been in T? Use process start + CPU time
        # as a crude proxy — but simpler: look at last watchdog PAUSE line.
        last_pause_ts=$(grep 'PAUSE supervisor' "$WD_LOG" 2>/dev/null | tail -1 \
                        | grep -oE '\[[0-9-]+ [0-9:]+\]' | tr -d '[]')
        now=$(date +%s)
        if [ -n "$last_pause_ts" ]; then
            pause_epoch=$(date -d "$last_pause_ts" +%s 2>/dev/null || echo 0)
            age=$(( now - pause_epoch ))
            if [ "$age" -gt "$PAUSED_MAX_AGE_SEC" ] && [ "$temp" -le "$TEMP_SAFE_C" ]; then
                say "FORCE_RESUME pid=$sup_pid paused=${age}s temp=${temp}C"
                kill -CONT "$sup_pid" 2>/dev/null
                for p in $(pgrep -f run_agent_shard.py); do kill -CONT "$p" 2>/dev/null; done
            fi
        fi
    fi
fi

# Watchdog heartbeat — restart if silent >2 min
if [ -f "$WD_LOG" ]; then
    last_hb=$(grep heartbeat "$WD_LOG" | tail -1 | grep -oE '\[[0-9-]+ [0-9:]+\]' | tr -d '[]')
    if [ -n "$last_hb" ]; then
        hb_epoch=$(date -d "$last_hb" +%s 2>/dev/null || echo 0)
        age=$(( $(date +%s) - hb_epoch ))
        if [ "$age" -gt 300 ] && ! pgrep -f 'scripts/cpu_watchdog.sh' >/dev/null; then
            say "WATCHDOG_RESTART last heartbeat ${age}s old"
            nohup bash "$ROOT/scripts/cpu_watchdog.sh" 80 70 5 \
                  >/tmp/cpu_watchdog.stdout 2>&1 &
            disown
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 3. Reap workers stuck >20 min
killed=0
for pid in $(pgrep -f run_agent_shard.py); do
    # etime in seconds via elapsed
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$age" ] && continue
    if [ "$age" -gt "$WORKER_MAX_AGE_SEC" ]; then
        say "REAP_WORKER pid=$pid age=${age}s (likely API hung)"
        kill -KILL "$pid" 2>/dev/null
        killed=$((killed + 1))
    fi
done

# ---------------------------------------------------------------------------
# 4. Rate-limit scan — last 30 min of atk_*.log
# Patterns tight enough NOT to match attack IDs like "atk_dc429..." — require
# the 429 to be in an HTTP/error context, or a word like "rate_limit_error".
RL_RE='HTTP[ /]4[0-9][0-9]|status.?code.?4[0-9][0-9]|code[:= ]4[0-9][0-9]|: 429 |rate[_ -]?limit[_ -]?(error|reached|exceeded)|rate_limit_error|RESOURCE_EXHAUSTED|usage limit|too many requests|TooManyRequests|429 Too Many|quota exceeded|insufficient_quota'

declare -A rl_count
while IFS= read -r f; do
    agent=$(basename "$f" | sed -E 's/atk_(.+)_s[0-9]+_.+\.log/\1/')
    n=$(grep -cE "$RL_RE" "$f" 2>/dev/null | head -1)
    n=${n:-0}
    n=$(echo "$n" | tr -d '[:space:]')
    if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
        rl_count[$agent]=$(( ${rl_count[$agent]:-0} + n ))
    fi
done < <(find /tmp/fleet -maxdepth 1 -name 'atk_*.log' -mmin -30 2>/dev/null)

for agent in "${!rl_count[@]}"; do
    n=${rl_count[$agent]}
    if [ "$n" -ge 3 ]; then
        say "ROTATE_HINT agent=$agent rate_limit_hits=$n in last 30min"
        # AUTO-ROTATE: if a claude_cli_* agent hit the limit, swap to minimax.
        # rotate_to_minimax.sh is idempotent (no-op if already rotated).
        case "$agent" in
            claude_cli*)
                say "AUTO_ROTATE invoking rotate_to_minimax.sh"
                bash "$ROOT/scripts/rotate_to_minimax.sh" >> /tmp/fleet/rotation.log 2>&1
                ;;
        esac
    fi
done

# ---------------------------------------------------------------------------
# 5. Status summary
if [ "$sup_alive" = 1 ]; then
    # Count finishes since last launch
    done=$(sed -n "${last_launch},\$p" "$SUP_LOG" 2>/dev/null \
           | grep -c 'finished atk:.*rc=0')
    # rate: last 30 min — avoid gawk-only match(); use python for portability
    cutoff=$(date -d '30 min ago' +%s 2>/dev/null || echo 0)
    recent=$(sed -n "${last_launch},\$p" "$SUP_LOG" 2>/dev/null \
            | grep 'finished atk:.*rc=0' \
            | "$VENV" -c '
import sys, re, datetime
cutoff = int(sys.argv[1])
n = 0
for line in sys.stdin:
    m = re.match(r"\[(\S+ \S+)\]", line)
    if m:
        try:
            ts = int(datetime.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S").timestamp())
            if ts >= cutoff: n += 1
        except Exception: pass
print(n)' "$cutoff")
    recent=${recent:-0}
    rate=$("$VENV" -c "print(f'{$recent/30:.1f}')" 2>/dev/null || echo "0.0")
    temp=$(( $(cat "$THERMAL" 2>/dev/null || echo 0) / 1000 ))
    say "STATUS sup=$sup_pid done=$done rate=${rate}/min temp=${temp}C killed=$killed"
else
    say "STATUS sup=DEAD should_run=$should_run killed=$killed"
fi
