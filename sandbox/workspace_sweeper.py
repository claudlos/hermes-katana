"""Legacy alias for hermes_katana.proving_ground.sandbox.workspace_sweeper."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.sandbox.workspace_sweeper")
sys.modules[__name__] = _mod
