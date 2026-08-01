from __future__ import annotations

import ast
from pathlib import Path

import oplm.slurm


def _static_sweep_imports(tree: ast.AST, module_name: str) -> list[str]:
    """Return static import statements anywhere in ``tree`` that reach into ``oplm.sweep``.

    ``ast.walk`` traverses every descendant node, so this already catches imports nested
    inside a function body, an ``if TYPE_CHECKING:`` block, or a ``try/except`` -- not just
    module-level statements. It also recognizes two forms a naive ``node.module.startswith(...)``
    check would miss: the package-attribute form (``from oplm import sweep``) and relative
    imports (``from ..sweep import X``, ``from .. import sweep``), both of which are realistic
    ways ``oplm.sweep`` vocabulary could sneak into ``oplm.slurm`` without ever writing the
    literal string ``"oplm.sweep"``.
    """
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "oplm.sweep" or alias.name.startswith("oplm.sweep."):
                    offenders.append(f"{module_name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                full_module = node.module or ""
                if full_module == "oplm.sweep" or full_module.startswith("oplm.sweep."):
                    offenders.append(f"{module_name}:{node.lineno} from {full_module} import ...")
                elif full_module == "oplm" and any(a.name == "sweep" for a in node.names):
                    offenders.append(f"{module_name}:{node.lineno} from oplm import sweep")
            elif node.level >= 2:
                # oplm.slurm.<module> sits two packages deep (oplm, slurm), so level>=2
                # resolves at or above the `oplm` package itself: `from ..sweep import X` or
                # `from .. import sweep`.
                resolved = node.module or ""
                hits_sweep_module = resolved == "sweep" or resolved.startswith("sweep.")
                hits_bare_sweep_name = resolved == "" and any(a.name == "sweep" for a in node.names)
                if hits_sweep_module or hits_bare_sweep_name:
                    offenders.append(f"{module_name}:{node.lineno} relative import of sweep")
    return offenders


def _dynamic_sweep_imports(tree: ast.AST, module_name: str) -> list[str]:
    """Return ``importlib.import_module(...)`` / ``__import__(...)`` calls naming ``oplm.sweep``.

    A static ``ast.Import`` / ``ast.ImportFrom`` scan alone cannot see a deferred,
    string-driven import like ``importlib.import_module("oplm.sweep.common")``; this checks
    call sites separately. It only catches a string *literal* first argument -- a computed or
    concatenated string is undecidable by static analysis and is out of scope here.
    """
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_import_module = isinstance(func, ast.Attribute) and func.attr == "import_module"
        is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
        if not (is_import_module or is_dunder_import) or not node.args:
            continue
        target = node.args[0]
        if (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value.startswith("oplm.sweep")
        ):
            offenders.append(f"{module_name}:{node.lineno} dynamic import {target.value!r}")
    return offenders


def test_slurm_layer_does_not_import_sweep() -> None:
    """oplm.slurm is general-purpose; oplm.sweep depends on it, never the reverse."""
    package = Path(oplm.slurm.__file__).parent
    offenders = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text())
        offenders += _static_sweep_imports(tree, module.name)
    assert offenders == [], f"oplm.slurm must not import oplm.sweep: {offenders}"


def test_slurm_layer_does_not_dynamically_import_sweep() -> None:
    """A deferred `importlib.import_module("oplm.sweep...")` must be caught too."""
    package = Path(oplm.slurm.__file__).parent
    offenders = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text())
        offenders += _dynamic_sweep_imports(tree, module.name)
    assert offenders == [], f"oplm.slurm must not dynamically import oplm.sweep: {offenders}"
