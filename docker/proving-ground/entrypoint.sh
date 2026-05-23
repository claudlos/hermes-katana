#!/bin/bash
set -euo pipefail

export HERMES_HOME="${HERMES_HOME:-/opt/hermes_data}"
mkdir -p "$HERMES_HOME" results sessions

case "${1:-bash}" in
  healthcheck)
    exec /usr/local/bin/healthcheck.sh
    ;;
  shard)
    shift
    exec python -m hermes_katana.proving_ground.run_agent_shard "$@"
    ;;
  fleet)
    shift
    exec python -m hermes_katana.proving_ground.scripts.fleet "$@"
    ;;
  proving-ground)
    shift
    exec katana proving-ground "$@"
    ;;
  bash|sh)
    exec "$1"
    ;;
  *)
    exec "$@"
    ;;
esac
