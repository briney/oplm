"""`oplm sweep` command surface."""

from __future__ import annotations

from oplm.sweep.coord_check import main as coord_check_main
from oplm.sweep.phases import app

app.command("coord-check")(coord_check_main)

__all__ = ["app"]
