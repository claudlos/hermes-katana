"""Legacy alias for hermes_katana.proving_ground.synthdata.schema."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata.schema")
sys.modules[__name__] = _mod
