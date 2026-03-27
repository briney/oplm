"""Decorator-based registry mapping type strings to EvalTask subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.eval.tasks.base import EvalTask

EVAL_TASK_REGISTRY: dict[str, type[EvalTask]] = {}


def register_eval_task(type_name: str) -> Callable[[type[EvalTask]], type[EvalTask]]:
    """Decorator to register an EvalTask subclass for a dataset type.

    Args:
        type_name: The eval type string (e.g. "sequence", "structure").

    Returns:
        Decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[EvalTask]) -> type[EvalTask]:
        if type_name in EVAL_TASK_REGISTRY:
            raise ValueError(
                f"Eval task type {type_name!r} is already registered "
                f"to {EVAL_TASK_REGISTRY[type_name].__name__}"
            )
        EVAL_TASK_REGISTRY[type_name] = cls
        return cls

    return decorator


def get_eval_task_class(type_name: str) -> type[EvalTask]:
    """Look up a registered EvalTask class by type string.

    Args:
        type_name: The eval type string.

    Returns:
        The registered EvalTask subclass.

    Raises:
        ValueError: If the type is not registered.
    """
    if type_name not in EVAL_TASK_REGISTRY:
        available = ", ".join(sorted(EVAL_TASK_REGISTRY))
        raise ValueError(f"Unknown eval type {type_name!r}. Available: {available}")
    return EVAL_TASK_REGISTRY[type_name]
