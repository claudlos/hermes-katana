"""Module alias for hermes_katana.proving_ground.scripts.v8_freeze_analysis_20260514."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.v8_freeze_analysis_20260514")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.v8_freeze_analysis_20260514")
    raise SystemExit(main())
