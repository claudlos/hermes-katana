"""Backward-compatible module alias for hermes_katana.proving_ground.sandbox_cli."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.sandbox_cli")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    raise SystemExit(_mod.main())
