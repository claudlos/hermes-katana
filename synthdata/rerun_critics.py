"""Legacy alias for hermes_katana.proving_ground.synthdata.rerun_critics."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata.rerun_critics")
sys.modules[__name__] = _mod
