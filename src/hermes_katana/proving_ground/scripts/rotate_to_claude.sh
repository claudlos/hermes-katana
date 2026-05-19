#!/usr/bin/env bash
# rotate_to_claude.sh — reverse of rotate_to_minimax.sh. Invoke after the
# Claude Max 5-hour window resets, to put claude_cli_sonnet + claude_cli_haiku
# back into the fleet spec and drop the auto-added minimax worker.

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
if [ "$current_state" != "to_minimax" ]; then
    say "not in to_minimax state — nothing to rotate back"
    exit 0
fi

say "ROTATE_TO_CLAUDE start"

if [ -f "$PID_FILE" ]; then
    sup_pid=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$sup_pid" ] && kill -0 "$sup_pid" 2>/dev/null; then
        "$VENV" "$ROOT/scripts/fleet.py" stop 2>&1 | tail -1 >> "$LOG"
        for i in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$sup_pid" 2>/dev/null || break
            sleep 2
        done
    fi
fi
for p in $(pgrep -f run_agent_shard.py); do kill -TERM "$p" 2>/dev/null; done
sleep 2
for p in $(pgrep -f run_agent_shard.py); do kill -KILL "$p" 2>/dev/null; done

"$VENV" <<PY
import json, pathlib
p = pathlib.Path("$SPEC")
spec = json.loads(p.read_text())
workers = spec["workers"]
# Drop auto-added minimax
kept = [w for w in workers if not (w["agent"] == "hermes_minimax_m2_7" and w.get("_note","").startswith("AUTO-ADDED"))]
# Use first remaining worker as the shard/channel/max_attacks template
tmpl = kept[0] if kept else workers[0]
shards = tmpl["shards"]
channels = tmpl["channels"]
max_attacks = tmpl.get("max_attacks", 100)
# Re-add claude_cli_sonnet + claude_cli_haiku if missing
have = {w["agent"] for w in kept}
if "claude_cli_sonnet" not in have:
    kept.append({"_note": "Re-added by rotate_to_claude.sh after Max reset.",
                 "agent": "claude_cli_sonnet", "shards": shards, "channels": channels,
                 "max_attacks": max_attacks, "instances": 1})
if "claude_cli_haiku" not in have:
    kept.append({"_note": "Re-added by rotate_to_claude.sh after Max reset.",
                 "agent": "claude_cli_haiku", "shards": shards, "channels": channels,
                 "max_attacks": max_attacks, "instances": 1})
spec["workers"] = kept
spec["_comment"] = spec.get("_comment","") + " | ROTATED back to claude at $(date -u +%FT%TZ)"
p.write_text(json.dumps(spec, indent=2) + "\n")
print("spec updated; workers:", [w["agent"] for w in kept])
PY
echo "$(date +%s) to_claude" > "$STATE"
say "spec rewritten; relaunching"

nohup "$VENV" "$ROOT/scripts/fleet.py" launch --spec "$SPEC" \
      >"/tmp/fleet/post_rotate_$(date +%s).stdout" 2>&1 &
disown
sleep 3
say "relaunch done — supervisor pid=$(cat $PID_FILE 2>/dev/null)"
