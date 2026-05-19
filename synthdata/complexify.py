"""Legacy alias for hermes_katana.proving_ground.synthdata.complexify."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata.complexify")
sys.modules[__name__] = _mod
