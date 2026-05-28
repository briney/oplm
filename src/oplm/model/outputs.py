"""LogitsConfig and LogitsOutput dataclasses for the ESM-C-style API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["LogitsConfig", "LogitsOutput"]


@dataclass
class LogitsConfig:
    """Knobs that control what `OplmFor*.logits()` returns.

    Attributes:
        sequence: When True, populate `LogitsOutput.sequence_logits` from the
            model's output head.
        return_embeddings: When True, populate `LogitsOutput.embeddings` with
            the model's last hidden state.
        return_hidden_states: When True, populate `LogitsOutput.hidden_states`
            with the full per-layer hidden-state tuple (post-embedding plus
            one tensor per `OplmBlock`).
        return_attentions: When True, populate `LogitsOutput.attentions` with
            the per-layer attention weights. Forces the manual fallback
            attention kernel (see `docs/MODEL_ARCHITECTURE.md` §6.5).
    """

    sequence: bool = True
    return_embeddings: bool = False
    return_hidden_states: bool = False
    return_attentions: bool = False


@dataclass
class LogitsOutput:
    """Structured return type for the ESM-C-style `logits()` convenience API.

    Each field is `None` unless the corresponding `LogitsConfig` flag was set.

    Attributes:
        sequence_logits: `(B, T, vocab_size)` token logits, or `None`.
        embeddings: `(B, T, D)` last hidden state, or `None`.
        hidden_states: Tuple of `(B, T, D)` tensors — the post-embedding state
            followed by one entry per block — or `None`.
        attentions: Tuple of `(B, H, T, T)` fp32 attention-weight tensors, one
            per layer, or `None`.
    """

    sequence_logits: torch.Tensor | None
    embeddings: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
