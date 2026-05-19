"""Legacy alias for hermes_katana.proving_ground.synthdata.meta_prompt."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.synthdata.meta_prompt")
sys.modules[__name__] = _mod
