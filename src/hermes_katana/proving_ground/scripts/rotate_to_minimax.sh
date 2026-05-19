#!/usr/bin/env bash
# rotate_to_minimax.sh — swap claude_cli_* workers OUT and hermes_minimax_m2_7 IN
# when Claude Max hits its rate limit. Called by fleet_monitor.sh when it detects
# ≥3 rate-limit markers on claude_cli_*, OR invoke manually.
#
# Strategy:
#   1. Stop the supervisor (SIGINT) and wait for clean exit.
#   2. Edit scripts/fleet_v11.json: disable claude_cli_sonnet + claude_cli_haiku,
#      enable hermes_minimax_m2_7 with same shard range and channels.
#      Uses jq to keep the edit structural-not-string so it's safe.
#   3. Relaunch fleet.py against the same spec.
#   4. Write /tmp/fleet/rotation.state with timestamp + "to_minimax" so
#      rotate_to_claude.sh can swap back.
#
# Safe to call multiple times — if already on minimax, it's a no-op.

set -u
ROOT=/home/carlos/Documents/Code/katana-proving-ground
VENV=$ROOT/.venv/bin/python
SPEC=$ROOT/scripts/fleet_v11.json
LOG=/tmp/fleet/rotation.log
STATE=/tmp/fleet/rotation.state
PID_FILE=/tmp/fleet/fleet.pid

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

current_state=""
[ -f "$STATE" ] && current_state=$(awk '{print $2}' "$STATE" | tail -1)
if [ "$current_state" = "to_minimax" ]; then
    say "already rotated to_minimax — no-op"
    exit 0
fi

say "ROTATE_TO_MINIMAX start"

# Stop supervisor
if [ -f "$PID_FILE" ]; then
    sup_pid=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$sup_pid" ] && kill -0 "$sup_pid" 2>/dev/null; then
        say "stopping supervisor pid=$sup_pid"
        "$VENV" "$ROOT/scripts/fleet.py" stop 2>&1 | tail -1 >> "$LOG"
        for i in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$sup_pid" 2>/dev/null || break
            sleep 2
        done
    fi
fi
# reap any leftover workers
for p in $(pgrep -f run_agent_shard.py); do kill -TERM "$p" 2>/dev/null; done
sleep 2
for p in $(pgrep -f run_agent_shard.py); do kill -KILL "$p" 2>/dev/null; done

# Edit spec — remove claude_cli_* workers, add hermes_minimax_m2_7
"$VENV" <<PY
import json, pathlib
p = pathlib.Path("$SPEC")
spec = json.loads(p.read_text())
workers = spec["workers"]
kept = [w for w in workers if not w["agent"].startswith("claude_cli")]
# derive shard range from the claude_cli entry we removed (use sonnet if present)
claude = next((w for w in workers if w["agent"] == "claude_cli_sonnet"), None)
shards = claude["shards"] if claude else workers[0]["shards"]
channels = claude["channels"] if claude else workers[0]["channels"]
max_attacks = claude.get("max_attacks", 100) if claude else 100
# add minimax (or update if present already)
if any(w["agent"] == "hermes_minimax_m2_7" for w in kept):
    pass
else:
    kept.append({
        "_note": "AUTO-ADDED by rotate_to_minimax.sh — claude_cli dropped because of Max rate-limit.",
        "agent": "hermes_minimax_m2_7",
        "shards": shards,
        "channels": channels,
        "max_attacks": max_attacks,
        "instances": 1
    })
spec["workers"] = kept
spec["_comment"] = spec.get("_comment","") + " | ROTATED to minimax at $(date -u +%FT%TZ)"
p.write_text(json.dumps(spec, indent=2) + "\n")
print("spec updated; workers:", [w["agent"] for w in kept])
PY
echo "$(date +%s) to_minimax" > "$STATE"
say "spec rewritten; relaunching"

# Relaunch
nohup "$VENV" "$ROOT/scripts/fleet.py" launch --spec "$SPEC" \
      >"/tmp/fleet/post_rotate_$(date +%s).stdout" 2>&1 &
disown
sleep 3
say "relaunch done — supervisor pid=$(cat $PID_FILE 2>/dev/null)"
