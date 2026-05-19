"""Module alias for hermes_katana.proving_ground.scripts.audit_v8_for_hermes_katana_20260515."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.scripts.audit_v8_for_hermes_katana_20260515")
sys.modules[__name__] = _mod

if __name__ == "__main__":
    main = getattr(_mod, "main", None)
    if main is None:
        raise SystemExit("No main() entry point for scripts.audit_v8_for_hermes_katana_20260515")
    raise SystemExit(main())
