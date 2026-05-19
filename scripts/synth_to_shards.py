"""Module alias for hermes_katana.proving_ground.scripts.synth_to_shards."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.synth_to_shards")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.synth_to_shards")
    raise SystemExit(main())
