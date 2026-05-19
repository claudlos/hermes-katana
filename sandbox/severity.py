"""Legacy alias for hermes_katana.proving_ground.sandbox.severity."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.sandbox.severity")
sys.modules[__name__] = _mod
