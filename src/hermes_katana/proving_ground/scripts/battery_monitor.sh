#!/usr/bin/env bash
shopt -s nullglob
# Battery health monitor — runs periodically (via cron), does three things:
#
#   1. Snapshot worker state: running processes, per-worker progress from
#      status.json files, totals, any outputs that haven't updated in a while.
#   2. Auto-remediate the easy failures:
#        - kill orphan llama-server processes (no orchestrator parent)
#        - delete empty/stale session workspace dirs (>2h old, 0 bytes)
#        - rotate the monitor log if it gets too large
#   3. Append a dated snapshot to /tmp/battery_monitor.log, prepend a one-line
#      summary to /tmp/battery_update.txt so the user can `tail -n 1` for
#      the latest.
#
# Intended to be invoked every 5-10 minutes by cron.
# Safe to run manually any time: `scripts/battery_monitor.sh`

set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${KATANA_PROVING_GROUND_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT" || exit 1

LOG=/tmp/battery_monitor.log
ONELINER=/tmp/battery_update.txt
STAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Rotate log if it grows past 2 MB.
if [[ -f "$LOG" ]] && [[ $(stat -c%s "$LOG" 2>/dev/null || echo 0) -gt 2097152 ]]; then
    mv "$LOG" "${LOG}.1" 2>/dev/null
fi

echo "=================================================" >> "$LOG"
echo "[$STAMP] battery monitor tick" >> "$LOG"
echo "=================================================" >> "$LOG"

# --- Worker inventory ---
n_api_orch=$(pgrep -c -f "orchestrate_battery.sh" 2>/dev/null || echo 0)
n_run_shard=$(pgrep -c -f "run_shard.py" 2>/dev/null || echo 0)
n_agent_shard=$(pgrep -c -f "run_agent_shard.py" 2>/dev/null || echo 0)
n_llama=$(pgrep -c -f "llama-server" 2>/dev/null || echo 0)

echo "processes: api_orch=$n_api_orch run_shard=$n_run_shard agent_shard=$n_agent_shard llama_server=$n_llama" >> "$LOG"

# --- Per-worker progress from status files ---
api_done=0; api_eff=0; api_col=0
agent_done=0; agent_eff=0; agent_canary=0

for s in "$ROOT"/results/shard_runs/*.status.json; do
    [[ -f "$s" ]] || continue
    line=$(python3 -c "
import json,sys
try:
    d=json.load(open('$s'))
    print(f\"{d.get('done',0)} {d.get('effective',0)} {d.get('collapsed',0)}\")
except Exception:
    print('0 0 0')
")
    d=$(echo "$line" | cut -d' ' -f1)
    e=$(echo "$line" | cut -d' ' -f2)
    c=$(echo "$line" | cut -d' ' -f3)
    api_done=$((api_done + d))
    api_eff=$((api_eff + e))
    api_col=$((api_col + c))
done

for s in "$ROOT"/results/agent_shard_runs/*.status.json; do
    [[ -f "$s" ]] || continue
    line=$(python3 -c "
import json,sys
try:
    d=json.load(open('$s'))
    print(f\"{d.get('done',0)} {d.get('effective',0)} {d.get('canary_leaks',0)}\")
except Exception:
    print('0 0 0')
")
    d=$(echo "$line" | cut -d' ' -f1)
    e=$(echo "$line" | cut -d' ' -f2)
    c=$(echo "$line" | cut -d' ' -f3)
    agent_done=$((agent_done + d))
    agent_eff=$((agent_eff + e))
    agent_canary=$((agent_canary + c))
done

echo "API sessions: done=$api_done effective=$api_eff collapsed=$api_col" >> "$LOG"
echo "Agent sessions: done=$agent_done effective=$agent_eff canary_leaks=$agent_canary" >> "$LOG"

# --- GPU state ---
gpu_state=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $gpu_state" >> "$LOG"

# --- Disk state ---
disk_home=$(df -h /home | tail -1 | awk '{print $5 " used, " $4 " free"}')
disk_tmp=$(df -h /tmp  | tail -1 | awk '{print $5 " used, " $4 " free"}')
echo "Disk /home: $disk_home" >> "$LOG"
echo "Disk /tmp : $disk_tmp" >> "$LOG"

# Pressure check: if /home usage >95%, log LOUD warning.
pct=$(df /home | tail -1 | awk '{print $5}' | tr -d '%')
if [[ -n "$pct" && "$pct" -ge 95 ]]; then
    echo "!!! CRITICAL: /home disk usage ${pct}% — scanner artifacts + session logs will fail to write" >> "$LOG"
fi

# --- Stale workers: status files that haven't updated in >15 min but
#     worker process is still alive. Usually fine (slow models like arcee
#     at 190s/attack); flag if over 20 min.
now=$(date +%s)
stale_list=""
for s in "$ROOT"/results/shard_runs/*.status.json "$ROOT"/results/agent_shard_runs/*.status.json; do
    [[ -f "$s" ]] || continue
    age=$(( now - $(stat -c %Y "$s") ))
    if [[ $age -gt 1200 ]]; then
        stale_list="$stale_list $(basename ${s%.status.json})(${age}s)"
    fi
done
if [[ -n "$stale_list" ]]; then
    echo "STALE (no progress >20min): $stale_list" >> "$LOG"
fi

# --- Auto-remediation ---

# 1. Kill orphan llama-server processes (no orchestrator parent alive).
#    Our orchestrator pkill's llama-server when switching models, but a
#    crashed orchestrator can leave one behind blocking VRAM.
if [[ $n_api_orch -eq 0 && $n_llama -gt 0 ]]; then
    echo "REMEDIATE: $n_llama orphan llama-server process(es) with no orchestrator — killing" >> "$LOG"
    pkill -f "llama-server" 2>/dev/null
fi

# 2. Empty + stale session workspace dirs (>2 h old, 0 bytes of content).
#    Our worker creates a fresh workspace per session; if the process
#    crashed before the agent wrote anything, we get empty dirs.
stale_ws=$(find "$ROOT/sessions" -maxdepth 1 -type d -empty -mmin +120 2>/dev/null | wc -l)
if [[ $stale_ws -gt 0 ]]; then
    find "$ROOT/sessions" -maxdepth 1 -type d -empty -mmin +120 -delete 2>/dev/null
    echo "REMEDIATE: deleted $stale_ws empty stale workspace dirs" >> "$LOG"
fi

# 3. Session workspaces older than 6 h — keep the JSONL outputs in
#    results/ but wipe the workspace files. Saves disk; workspace can be
#    regenerated if we need to re-run.
old_ws_bytes=$(find "$ROOT/sessions" -maxdepth 1 -type d -mmin +360 2>/dev/null | wc -l)
if [[ $old_ws_bytes -gt 50 ]]; then
    # Only remediate if we're above 50 old dirs (threshold avoids churn).
    find "$ROOT/sessions" -maxdepth 1 -type d -mmin +360 -exec rm -rf {} + 2>/dev/null
    echo "REMEDIATE: wiped $old_ws_bytes session workspace dirs >6h old" >> "$LOG"
fi

# --- Recent effective hits (last 10 lines from most active streams) ---
echo "Recent agent signal:" >> "$LOG"
for f in "$ROOT"/results/agent_shard_runs/*.jsonl; do
    [[ -f "$f" ]] || continue
    name=$(basename "${f%.jsonl}")
    # Pull last row; report only if effective or has a reason tag.
    python3 -c "
import json
try:
    with open('$f') as fh:
        lines = fh.readlines()
    if lines:
        d = json.loads(lines[-1])
        if d.get('effective') or d.get('canary_leaked'):
            tags = d.get('reasons') or []
            print(f\"    {d['agent_id']:<30} shard={d.get('shard')} ch={d.get('channel','file_content')} : {'|'.join(tags)}\")
except Exception:
    pass
" >> "$LOG"
done 2>/dev/null

# --- One-line summary (newest first) ---
ONELINE="[$STAMP] workers api_orch=$n_api_orch agents=$n_agent_shard llama=$n_llama | API_done=$api_done eff=$api_eff col=$api_col | AGENT_done=$agent_done eff=$agent_eff canary=$agent_canary | GPU:$gpu_state"
echo "$ONELINE" > "$ONELINER"
echo "$ONELINE" >> "${ONELINER}.history"

# Keep history small.
tail -n 200 "${ONELINER}.history" > "${ONELINER}.history.tmp" 2>/dev/null
mv "${ONELINER}.history.tmp" "${ONELINER}.history" 2>/dev/null

echo "" >> "$LOG"
