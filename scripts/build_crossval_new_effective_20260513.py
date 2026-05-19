"""Module alias for hermes_katana.proving_ground.scripts.build_crossval_new_effective_20260513."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.build_crossval_new_effective_20260513")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.build_crossval_new_effective_20260513")
    raise SystemExit(main())
