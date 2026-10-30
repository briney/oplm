"""Thin MMseqs2 subprocess helpers shared by the data build and fold eval sets."""

from __future__ import annotations

import subprocess

__all__ = ["mmseqs_version", "run_mmseqs_cmd"]


def run_mmseqs_cmd(mmseqs: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run mmseqs with args, raising clear RuntimeError on missing or failing binary.

    Args:
        mmseqs: Path or name of mmseqs executable.
        args: Command and arguments, e.g. ["version"] or ["easy-cluster", ...].

    Returns:
        CompletedProcess with stdout and stderr as strings.

    Raises:
        RuntimeError: mmseqs binary not found or exited non-zero.
    """
    try:
        return subprocess.run([mmseqs, *args], check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"MMseqs2 executable not found: {mmseqs}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{mmseqs} {args[0]} failed:\n{exc.stderr}") from exc


def mmseqs_version(mmseqs: str = "mmseqs") -> str:
    """The ``mmseqs version`` string, recorded in the README."""
    return run_mmseqs_cmd(mmseqs, ["version"]).stdout.strip()
