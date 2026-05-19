"""Module alias for hermes_katana.proving_ground.scripts.factorial_decompose."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.factorial_decompose")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.factorial_decompose")
    raise SystemExit(main())
