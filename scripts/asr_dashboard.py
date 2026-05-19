"""Module alias for hermes_katana.proving_ground.scripts.asr_dashboard."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.asr_dashboard")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.asr_dashboard")
    raise SystemExit(main())
