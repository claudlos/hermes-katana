"""Module alias for hermes_katana.proving_ground.scripts.export_channel_weights."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.export_channel_weights")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.export_channel_weights")
    raise SystemExit(main())
