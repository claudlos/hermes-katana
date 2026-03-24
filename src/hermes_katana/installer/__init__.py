"""
HermesKatana Installer — patch-based integration into Hermes checkouts.

The installer detects a local Hermes source tree, applies source patches
to wire in Katana's middleware, proxy, and banner, and manages the
configuration and CA certificate lifecycle.

Usage::

    from hermes_katana.installer import KatanaInstaller

    installer = KatanaInstaller()
    if installer.detect_hermes("/path/to/hermes"):
        installer.install("/path/to/hermes")
"""

from hermes_katana.installer.installer import KatanaInstaller

__all__ = [
    "KatanaInstaller",
]
