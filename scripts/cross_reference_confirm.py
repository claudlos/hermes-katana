"""Module alias for hermes_katana.proving_ground.scripts.cross_reference_confirm."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.cross_reference_confirm")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.cross_reference_confirm")
    raise SystemExit(main())
