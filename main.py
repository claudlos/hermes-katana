"""Backward-compatible Proving Ground CLI shim."""

from hermes_katana.proving_ground.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
