"""Docs guard: links resolve, retired doc names are gone, documented CLI/config exist."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import DataConfig, FoldRunConfig, TrainConfig
from oplm.model import OplmConfig

REPO = Path(__file__).resolve().parents[1]
RETIRED: list[str] = [
    "DATA_TOOLING.md",
    "DATA_MANIFEST.md",
    "MODEL_ARCHITECTURE.md",
    "CONFIG.md",
    "EVAL_HARNESS.md",
    "TRAIN.md",
    "SLURM.md",
    "MUP.md",
    "LR_SWEEP.md",
    "LATE_LOSS_INVESTIGATION.md",
    "docs/late-loss",
    "TESTING_E2E.md",
    "RECOVERY_TESTS.md",
    "NOTES.md",
    "yagni_assessment",
    "docs/perf/",
    "TODOS.md",
    "OVERVIEW.md",
]  # tasks append retired doc names as they delete them
# New docs are lowercase; the old uppercase docs (about to be deleted) are never checked.
TREE_DOCS = [
    p
    for p in [
        REPO / "README.md",
        REPO / "AGENTS.md",
        REPO / "docs" / "FOLD.md",
        REPO / "docs" / "README.md",
        *sorted(p for p in (REPO / "docs").glob("*.md") if p.name == p.name.lower()),
    ]
    if p.exists()
]
_SCAN = [
    *TREE_DOCS,
    *REPO.glob("src/**/*.py"),
    *REPO.glob("src/**/*.yaml"),
    *REPO.glob("tests/**/*.py"),
    *REPO.glob("configs/**/*.yaml"),
    REPO / "pyproject.toml",
]
_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)[^)]*\)")


def _slug(heading: str) -> str:  # GitHub anchor slug
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower().replace("`", ""))
    return s.replace(" ", "-")


def _anchors(md: Path) -> set[str]:
    text = _FENCE.sub("", md.read_text(encoding="utf-8"))
    return {_slug(m) for m in re.findall(r"^#+\s+(.*)$", text, re.MULTILINE)}


def test_relative_links_resolve() -> None:
    failures = []
    checked = 0
    for doc in TREE_DOCS:
        text = _FENCE.sub("", doc.read_text(encoding="utf-8"))
        for target in _LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            checked += 1
            path_part, _, anchor = target.partition("#")
            base = REPO if path_part.startswith("/") else doc.parent  # "/x" = repo root
            dest = (base / path_part.lstrip("/")).resolve() if path_part else doc
            if not dest.exists():
                failures.append(f"{doc.relative_to(REPO)}: {target}")
            elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
                failures.append(f"{doc.relative_to(REPO)}: {target} (no such heading)")
    assert checked > 0, "scanner found no links"
    assert not failures, "\n".join(failures)


def test_no_references_to_retired_docs() -> None:
    hits = []
    for f in _SCAN:
        if f == Path(__file__).resolve() or not f.exists():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "/blob/" in line:
                continue
            hits += [f"{f.relative_to(REPO)}:{n}: {r}" for r in RETIRED if r in line]
    assert not hits, "\n".join(hits)


def test_documented_cli_commands_exist() -> None:
    runner = CliRunner()
    failures = []
    checked = 0
    for doc in TREE_DOCS:
        for block in re.findall(r"^```.*?^```", doc.read_text(encoding="utf-8"), re.M | re.S):
            for line in block.splitlines():
                m = re.match(r"(?:\$ )?oplm\s+(\S+)(?:\s+(\S+))?", line.strip())
                if not m:
                    continue
                argv = [m.group(1)]
                if argv[0] in {"sweep", "slurm", "fold"} and m.group(2) and m.group(2)[0] != "-":
                    argv.append(m.group(2))
                checked += 1
                if runner.invoke(app, [*argv, "--help"]).exit_code != 0:
                    failures.append(f"{doc.relative_to(REPO)}: oplm {' '.join(argv)}")
    assert checked > 0, "scanner found no documented oplm commands"
    assert not failures, "\n".join(failures)


def test_documented_config_fields_exist() -> None:
    classes = {"train": TrainConfig, "data": DataConfig, "fold": FoldRunConfig}
    fields = {k: {f.name for f in dataclasses.fields(c)} for k, c in classes.items()}
    model = OplmConfig()
    pat = re.compile(r"(?:`|\s)(model|train|data|fold)\.([a-z_][a-z0-9_]*)(?=[`=.\[])")
    failures = []
    checked = 0
    for doc in TREE_DOCS:
        for section, name in pat.findall(doc.read_text(encoding="utf-8")):
            if name in {"py", "yaml", "md", "json"}:
                continue
            checked += 1
            ok = hasattr(model, name) if section == "model" else name in fields[section]
            if not ok:
                failures.append(f"{doc.relative_to(REPO)}: {section}.{name}")
    assert checked > 0, "scanner found no documented config fields"
    assert not failures, "\n".join(failures)


PENDING = "> **Pending:**"


def test_pending_markers_present() -> None:
    expect = {  # doc → issue numbers whose Pending markers must appear in it
        "docs/data.md": ["#61"],
        "docs/training.md": ["#62", "#63"],
        "docs/mup-lr.md": ["#55"],
        "docs/inference.md": ["#62"],
        "docs/evaluation.md": ["#57"],
        "docs/reproduce.md": ["#55"],  # decay recipe + expected loss / P@L ranges
    }
    for doc, issues in expect.items():
        lines = [ln for ln in (REPO / doc).read_text().splitlines() if ln.startswith(PENDING)]
        for issue in issues:
            assert any(issue in ln for ln in lines), f"{doc}: no Pending marker for {issue}"
