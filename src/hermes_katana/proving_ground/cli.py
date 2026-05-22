#!/usr/bin/env python3
"""Entry point for katana-proving-ground.

Also exposed as the ``proving-ground`` console script via pyproject.toml.

Dispatches README-documented subcommands to the underlying modules so users
have a single command surface. Specialized shard runners remain callable with
``python -m hermes_katana.proving_ground.run_shard`` and
``python -m hermes_katana.proving_ground.run_agent_shard`` because their
argparse interfaces are tied to fleet-execution concerns.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is importable when invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    pass


USAGE = """\
Katana Proving Ground

Workflows:
  proving-ground run [options]              Single sandbox session
  proving-ground batch [options]            Batch of sandbox sessions
  proving-ground analyze <session-id>       Analyze an existing session
  proving-ground list-sessions              List tracked sessions
  proving-ground list-tasks                 Show available workspace tasks
  proving-ground synthesize [options]       Generate synthetic attack variants

Direct module entry points:
  python -m hermes_katana.proving_ground.run_shard [...]        API fleet runner
  python -m hermes_katana.proving_ground.run_agent_shard [...]  CLI-agent fleet runner

Environment overrides:
  HERMES_KATANA_ROOT          path to sibling hermes-katana repo
  KATANA_PROVING_GROUND_ROOT  path to this repo (rarely needed)

See README.md for full options.
"""


_SANDBOX_SUBCOMMANDS = {"run", "batch", "analyze", "list-sessions", "list-tasks"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    cmd = args[0]

    if cmd in _SANDBOX_SUBCOMMANDS:
        from hermes_katana.proving_ground.sandbox_cli import main as sandbox_main

        sys.argv = ["proving-ground", *args]
        sandbox_main()
        return 0

    if cmd == "synthesize":
        from hermes_katana.proving_ground.generate_variants import main as synthesize_main

        sys.argv = ["proving-ground", *args[1:]]
        rc = synthesize_main()
        return int(rc) if isinstance(rc, int) else 0

    print(f"Unknown command: {cmd}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
