"""Eval-task registry tests — pure, no model. See docs/EVAL_HARNESS.md §10."""

from __future__ import annotations

import pytest

import oplm.eval.tasks  # noqa: F401  -- import triggers @register_eval_task for shipped types
from oplm.eval import EvalTask, register_eval_task
from oplm.eval.registry import get_eval_task_class

_REAL_TYPES = ("sequence", "structure", "proteingym", "tape", "proteinglue", "everest")


def test_duplicate_registration_raises() -> None:
    """Re-registering an existing type fails and names the conflict."""

    class _Clash(EvalTask):
        def evaluate(self, model: object, accelerator: object) -> dict[str, float]:
            return {}

    with pytest.raises(ValueError, match="sequence"):
        register_eval_task("sequence")(_Clash)  # type: ignore[arg-type]


def test_unknown_type_raises_with_known_list() -> None:
    """An unknown type is rejected with an actionable list of registered types."""
    with pytest.raises(ValueError, match="sequence") as exc_info:
        get_eval_task_class("does_not_exist")
    assert "does_not_exist" in str(exc_info.value)


@pytest.mark.parametrize("type_name", _REAL_TYPES)
def test_real_types_resolve_to_eval_task_subclasses(type_name: str) -> None:
    """Every shipped type resolves to an ``EvalTask`` subclass after import."""
    cls = get_eval_task_class(type_name)
    assert issubclass(cls, EvalTask)


def test_registration_via_fixture_is_isolated(dummy_task_type: str) -> None:
    """The dummy fixture registers a resolvable type and cleans it up afterward."""
    from oplm.eval.registry import EVAL_TASK_REGISTRY

    assert issubclass(get_eval_task_class(dummy_task_type), EvalTask)
    assert dummy_task_type in EVAL_TASK_REGISTRY
