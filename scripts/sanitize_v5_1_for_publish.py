"""Module alias for hermes_katana.proving_ground.scripts.sanitize_v5_1_for_publish."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.sanitize_v5_1_for_publish")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.sanitize_v5_1_for_publish")
    raise SystemExit(main())
