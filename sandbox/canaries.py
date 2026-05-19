"""Legacy alias for hermes_katana.proving_ground.sandbox.canaries."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.sandbox.canaries")
sys.modules[__name__] = _mod
