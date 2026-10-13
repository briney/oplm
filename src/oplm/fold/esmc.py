"""Frozen ESM-C behind the hidden-state contract (addendum §3).

``load_esmc`` wraps the ``esm`` package's HF-style ``EsmcModel`` (``esm.models.esmc.model``,
``esm==3.4.1.post1``) in :class:`FrozenEsmC`, which presents the call
:func:`oplm.fold.lm_shim.frozen_lm_hidden_states` makes (``input_ids``, ``attention_mask``,
``output_hidden_states``, ``return_dict``) and returns the package's output, whose
``hidden_states`` is one ``(num_hidden_layers + 1, C, T, D)`` tensor: the input of every block
(index 0 is the embedding output) and the final post-LayerNorm output. ESM-C 600M therefore has
37 states of width 1152. The package needs Python >= 3.12 (extra ``oplm[esmc]``); it is imported
inside :func:`load_esmc` only, and CPU tests use a fake encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["ESMC_PREFIX", "ESMC_REPOS", "FrozenEsmC", "load_esmc", "parse_esmc_name"]

ESMC_PREFIX = "esmc:"
ESMC_REPOS = {
    "esmc_300m": "biohub/ESMC-300M",
    "esmc_600m": "biohub/ESMC-600M",
    "esmc_6b": "biohub/ESMC-6B",  # non-commercial license; a config choice, never the default
}


def parse_esmc_name(name_or_path: str) -> str | None:
    """The model name after ``esmc:``, or ``None`` when ``name_or_path`` has no such prefix."""
    text = str(name_or_path)
    return text[len(ESMC_PREFIX) :] if text.startswith(ESMC_PREFIX) else None


class FrozenEsmC(nn.Module):
    """A frozen, eval-only ESM-C encoder with the shape facts the shim needs.

    ``config`` is the wrapped model's config (``hidden_size``, ``num_hidden_layers``,
    ``pad_token_id``, ``mask_token_id``); ``num_hidden_states`` is ``num_hidden_layers + 1``.
    ``train()`` never leaves eval mode, so the head's ``train()`` cannot switch it.
    """

    def __init__(self, model: nn.Module, *, name: str, revision: str | None = None) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.revision = revision
        self.config: Any = getattr(model, "config")  # noqa: B009  # nn.Module attrs type as Tensor | Module
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @property
    def num_hidden_states(self) -> int:
        return int(self.config.num_hidden_layers) + 1

    def train(self, mode: bool = True) -> FrozenEsmC:
        super().train(False)
        return self

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Any:
        """Run the encoder; the result's ``hidden_states`` is ``(num_hidden_states, C, T, D)``.

        Raises:
            RuntimeError: The encoder returned no hidden states or a different count.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        states = getattr(out, "hidden_states", None)
        if states is None or states.shape[0] != self.num_hidden_states:
            got = None if states is None else int(states.shape[0])
            raise RuntimeError(
                f"ESM-C returned {got} hidden states; expected {self.num_hidden_states} "
                "(block inputs plus the final output)"
            )
        return out


def load_esmc(
    name: str,
    *,
    revision: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FrozenEsmC:
    """Load ``esmc_300m`` / ``esmc_600m`` / ``esmc_6b`` (or any ESM-C repo id or local dir).

    The package selects its kernels from ``device`` at load time, so the device is passed
    explicitly (CPU when ``None``). The Hub weights are gated behind the Cambrian license; a
    logged-in ``huggingface_hub`` token is required once per machine.

    Raises:
        ImportError: The ``esm`` package is missing (install ``oplm[esmc]`` on Python >= 3.12).
    """
    try:
        from esm.models.esmc.model import EsmcModel  # ty: ignore[unresolved-import]  # optional extra
    except ImportError as exc:
        raise ImportError(
            f"{ESMC_PREFIX}{name} needs the `esm` package: install `oplm[esmc]` (Python >= 3.12)"
        ) from exc
    target = torch.device("cpu") if device is None else torch.device(device)
    model = EsmcModel.from_pretrained(
        ESMC_REPOS.get(name, name), device=target, dtype=dtype, revision=revision
    )
    return FrozenEsmC(model, name=name, revision=revision)
