"""Structure prediction head: kernels (milestone 0) and the ESMFold2 inference port (milestone 1).

Importing the model classes registers ``oplm_fold`` with ``AutoConfig``/``AutoModel``.
Exports are lazy so the CLI can import ``oplm.fold.cli`` without loading torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oplm.fold.configuration_fold import FoldConfig
    from oplm.fold.data.featurize import ChainSpec, FoldFeatures, featurize
    from oplm.fold.modeling_fold import FoldOutput, OplmForFolding

__all__ = ["ChainSpec", "FoldConfig", "FoldFeatures", "FoldOutput", "OplmForFolding", "featurize"]

_LAZY = {
    "FoldConfig": ("oplm.fold.configuration_fold", "FoldConfig"),
    "ChainSpec": ("oplm.fold.data.featurize", "ChainSpec"),
    "FoldFeatures": ("oplm.fold.data.featurize", "FoldFeatures"),
    "featurize": ("oplm.fold.data.featurize", "featurize"),
    "FoldOutput": ("oplm.fold.modeling_fold", "FoldOutput"),
    "OplmForFolding": ("oplm.fold.modeling_fold", "OplmForFolding"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'oplm.fold' has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)
