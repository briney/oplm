"""Link check for repository markdown docs.

Guards against the silent-doc-rot failure mode from TECHNICAL_ANALYSIS.md
Finding 8: a doc referencing a file that does not exist (e.g. the
``EVAL_HARNESS.md`` / ``DATA_TOOLING.md`` / ``TESTING_E2E.md`` links that were
dangling before Phase 4). Every relative markdown link to a repo path must
resolve to an existing file or directory.

External links (``http(s)://``, ``mailto:``), pure in-page anchors (``#frag``),
and links inside fenced code blocks are ignored — only on-disk repo links are
checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Inline markdown link: [text](target). The target may carry a #fragment and an
# optional "title" after whitespace; both are stripped before resolution.
_LINK_RE = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:")


def _markdown_files() -> list[Path]:
    """Every tracked-looking markdown file under the repo (excluding VCS dirs)."""
    return sorted(
        p
        for p in _REPO_ROOT.rglob("*.md")
        if ".git" not in p.parts and "node_modules" not in p.parts
    )


def _repo_links(md_file: Path) -> list[str]:
    """Relative on-disk link targets in `md_file` (code blocks/externals removed)."""
    text = _FENCE_RE.sub("", md_file.read_text(encoding="utf-8"))
    links: list[str] = []
    for raw in _LINK_RE.findall(text):
        target = raw.strip().split()[0]  # drop any optional "title"
        if target.lower().startswith(_EXTERNAL_PREFIXES):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:  # pure in-page anchor
            continue
        links.append(path_part)
    return links


def _resolve(md_file: Path, link: str) -> Path:
    """Resolve a link relative to the repo root (``/``-prefixed) or the doc's dir."""
    if link.startswith("/"):
        return (_REPO_ROOT / link.lstrip("/")).resolve()
    return (md_file.parent / link).resolve()


_CASES = [
    pytest.param(md, link, id=f"{md.relative_to(_REPO_ROOT)}->{link}")
    for md in _markdown_files()
    for link in _repo_links(md)
]


@pytest.mark.parametrize(("md_file", "link"), _CASES)
def test_markdown_repo_link_resolves(md_file: Path, link: str) -> None:
    """Every relative markdown link points to a file or directory that exists."""
    target = _resolve(md_file, link)
    assert target.exists(), (
        f"{md_file.relative_to(_REPO_ROOT)} links to {link!r}, "
        f"which resolves to a missing path: {target}"
    )


def test_link_check_found_cases() -> None:
    """Sanity guard: the scanner actually found links (e.g. didn't silently no-op)."""
    assert _CASES, "no markdown repo links were discovered — the scanner is broken"
