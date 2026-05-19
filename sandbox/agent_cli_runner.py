"""Legacy alias for hermes_katana.proving_ground.sandbox.agent_cli_runner."""

from importlib import import_module
import sys

_mod = import_module("hermes_katana.proving_ground.sandbox.agent_cli_runner")
sys.modules[__name__] = _mod
