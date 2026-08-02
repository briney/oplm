"""Shared helper for normalizing CLI output before asserting on it in tests.

Why this exists: CI's terminal detection makes `rich` (which Typer/oplm's CLIs use for
console output) emit ANSI escape codes, while a plain local run typically does not. Rich
also auto-highlights things like bare numbers, so a code sequence can land *inside* a
substring a test is matching on -- e.g. ``--array=1,2,3`` renders as
``--array=\x1b[1;36m1\x1b[0m,\x1b[1;36m2\x1b[0m,...`` -- breaking a plain ``in`` check that
passes locally but fails in CI. Typer's error panels similarly interleave escape codes
mid-word (``-\x1b[0m\x1b[1;2;34m-help``) and wrap long lines with box-drawing borders.

`plain()` strips ANSI CSI escape sequences, replaces the box-drawing vertical bar rich uses
to pad wrapped panel/table lines (present regardless of color -- it is plain formatting, not
an SGR code) with whitespace, and then collapses all whitespace runs (including the newlines
introduced by panel/table wrapping) to single spaces. That combination is what makes substring
assertions match the same rendered text regardless of whether color/highlighting is active. Do
not delete this as "redundant" with a raw string comparison -- without it, tests that pass
locally can fail in any CI environment whose color/terminal detection differs (this is
exactly what broke `tests/sweep/test_phases.py` and friends under `FORCE_COLOR=1`).
"""

from __future__ import annotations

import re

# General CSI (Control Sequence Introducer) form: ESC '[' followed by parameter bytes
# (0x30-0x3F), intermediate bytes (0x20-0x2F), and a single final byte (0x40-0x7E). This
# covers SGR color/style codes (`\x1b[1;36m`) as well as other CSI sequences rich may emit,
# not just the `m`-terminated color case.
_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Rich pads wrapped lines inside error panels/tables with a vertical bar (e.g. a long error
# message wraps as `...same number │\n│ of values...`), splitting a contiguous phrase
# across lines even with color off. Whitespace-collapsing alone won't rejoin that, since
# `│` is not whitespace, so it is replaced with a space first.
_BOX_VERTICAL = "│"


def plain(text: str) -> str:
    """Strip ANSI escape codes and rich panel borders from ``text``, then collapse whitespace.

    Args:
        text: Raw CLI output (e.g. ``result.stdout`` / ``result.output`` from Typer's
            `CliRunner`), which may contain ANSI escape codes and rich's panel/table
            line-wrapping (box-drawing borders included).

    Returns:
        The text with escape codes and box-drawing borders removed and whitespace runs
        collapsed, safe for substring assertions regardless of whether the run had color
        enabled.
    """
    stripped = _CSI_RE.sub("", text).replace(_BOX_VERTICAL, " ")
    return " ".join(stripped.split())
