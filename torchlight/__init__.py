"""Compatibility entry point for the bundled torchlight package.

The upstream project keeps the installable package under
``torchlight/torchlight``. When commands run from the repository root, Python
sees the outer ``torchlight`` directory first, so this module forwards public
imports to the inner package.
"""

from .torchlight import *  # noqa: F401,F403
