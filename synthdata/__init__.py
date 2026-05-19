"""Legacy alias for hermes_katana.proving_ground.synthdata."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata")
sys.modules[__name__] = _mod
