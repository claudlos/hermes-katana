"""Legacy alias for hermes_katana.proving_ground.synthdata.expand_gap_labels."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata.expand_gap_labels")
sys.modules[__name__] = _mod
