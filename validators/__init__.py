"""Legacy alias for hermes_katana.proving_ground.validators."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.validators")
sys.modules[__name__] = _mod
