"""Backward-compatibility shim -- use :mod:`hermes_katana.scabbard.scabbard`."""

from hermes_katana.scabbard.config import ScabbardConfig
from hermes_katana.scabbard.scabbard import ScabbardClassifier

__all__ = ["ScabbardClassifier", "ScabbardConfig"]
