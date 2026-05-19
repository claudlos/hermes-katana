"""Module alias for hermes_katana.proving_ground.scripts.quota_mix_status_20260506."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.quota_mix_status_20260506")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.quota_mix_status_20260506")
    raise SystemExit(main())
