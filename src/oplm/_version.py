"""Read the version recorded when this distribution was built or installed."""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("oplm")
