"""Legacy alias for hermes_katana.proving_ground.sandbox.parsers."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.sandbox.parsers")
sys.modules[__name__] = _mod
