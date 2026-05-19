"""Legacy alias for hermes_katana.proving_ground.synthdata.critics."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata.critics")
sys.modules[__name__] = _mod
