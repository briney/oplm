"""All public Oplm* model classes — single file for trust_remote_code compatibility.

The architecture doc keeps every public model class in one module so that
`auto_map` + `trust_remote_code` loading works without chasing imports across
files. Internal building blocks live in their own modules and are imported here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

# A few of these relative imports (`attention`, `conv`, `ffn`, `masking`, `rope`)
# are not used directly here — they are reached transitively via `transformer`.
# They are imported anyway so that loading custom code from a local directory
# with trust_remote_code=True bundles every helper file: HuggingFace copies only
# a module's *direct* relative imports (depth-1; see
# transformers.dynamic_module_utils.get_cached_module_file). `_REMOTE_CODE_DEPS`
# references the otherwise-unused names so linters don't flag them.
from .attention import OplmAttention
from .configuration_oplm import _VALID_GRADIENT_CHECKPOINTING_MODES, OplmConfig
from .conv import CanonConv
from .embedding import cls_pool, mean_pool
from .ffn import SwiGLU
from .masking import prepare_attention_mask
from .norm import OplmLayerNorm, OplmRMSNorm, make_norm
from .outputs import LogitsConfig, LogitsOutput
from .rope import RotaryEmbedding
from .transformer import OplmBlock, OplmStack

_REMOTE_CODE_DEPS = (OplmAttention, CanonConv, SwiGLU, prepare_attention_mask, RotaryEmbedding)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

__all__ = [
    "OplmPreTrainedModel",
    "OplmModel",
    "OplmForMaskedLM",
    "OplmForSequenceClassification",
    "OplmForTokenClassification",
    "OplmMLMHead",
    "EsmcCompatMixin",
]


_MLM_HEAD_ACTIVATIONS = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}


class OplmPreTrainedModel(PreTrainedModel):
    """Abstract base for every OPLM model: weight init, gradient checkpointing, tokenizer attach."""

    config_class = OplmConfig
    base_model_prefix = "oplm"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OplmBlock"]
    _supports_sdpa = True
    _supports_flash_attn_2 = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights per §15.1 of the architecture doc.

        Residual-stream-writing projections (attention `o_proj`, FFN
        `down_proj`) are marked `_is_residual_writer = True` in their parent
        modules; when `config.init_scale_output_projections` is set, their std
        is shrunk by `1/sqrt(2 * num_hidden_layers)` to keep the residual
        stream variance stable with depth.
        """
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module_std = std
            if getattr(module, "_is_residual_writer", False) and (
                self.config.init_scale_output_projections
            ):
                module_std = std / math.sqrt(2 * self.config.num_hidden_layers)
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=module_std, a=-2 * module_std, b=2 * module_std
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            # HF convention: do NOT zero the <pad> row.
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (OplmLayerNorm, OplmRMSNorm)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                # nn.Module.__getattr__ types `.bias` as Tensor | Module; it is a Tensor here.
                nn.init.zeros_(module.bias)  # ty: ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Gradient checkpointing — propagate the toggle to every OplmBlock.
    # ------------------------------------------------------------------

    def _set_gradient_checkpointing(self, value: bool, mode: str | None = None) -> None:
        self.gradient_checkpointing = value
        for module in self.modules():
            if isinstance(module, OplmBlock | OplmStack):
                module.gradient_checkpointing = value
                if mode is not None:
                    module.gradient_checkpointing_mode = mode

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable activation checkpointing on every transformer block.

        Honors an optional ``{"mode": "full" | "selective"}`` entry in
        ``gradient_checkpointing_kwargs``; when absent, falls back to
        ``config.gradient_checkpointing_mode`` (default ``"full"``). ``"selective"``
        keeps matmul/SDPA outputs resident and recomputes only cheap ops, trading
        less memory savings for substantially less recompute than ``"full"``.
        """
        mode = (gradient_checkpointing_kwargs or {}).get("mode")
        if mode is None:
            mode = getattr(self.config, "gradient_checkpointing_mode", "full")
        if mode not in _VALID_GRADIENT_CHECKPOINTING_MODES:
            raise ValueError(
                f"gradient_checkpointing mode must be one of "
                f"{_VALID_GRADIENT_CHECKPOINTING_MODES}; got {mode!r}."
            )
        self._set_gradient_checkpointing(True, mode=mode)

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing on every transformer block."""
        self._set_gradient_checkpointing(False)

    # ------------------------------------------------------------------
    # Tokenizer auto-attachment.
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load the model and best-effort attach the saved tokenizer.

        If no tokenizer files sit next to the weights (scratch models, offline
        workflows, tests), `model.tokenizer` stays `None` and any call to
        `tokenize`/`encode`/`logits` raises with an actionable message.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        try:
            from transformers import AutoTokenizer

            model.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
        except (OSError, ValueError):
            model.tokenizer = None
        return model


class EsmcCompatMixin:
    """ESM-C-style convenience API (`tokenize`, `encode`, `logits`).

    Holds an optional `self.tokenizer`. There is deliberately no lazy
    `AutoTokenizer.from_pretrained` fallback — that pattern silently breaks for
    scratch models, offline workflows, and tests where tokenizer files do not
    exist next to the config. Attach a tokenizer either by loading through
    `Oplm*.from_pretrained(<path-with-tokenizer-files>)` or by assigning
    `model.tokenizer = AutoTokenizer.from_pretrained(...)`.
    """

    # NOTE: no class-level type annotation on `tokenizer` — transformers'
    # `PreTrainedModel.__init_subclass__` eagerly resolves class-body
    # annotations via `get_type_hints`, which would force a runtime import of
    # `PreTrainedTokenizerBase`. The plain default keeps that import deferrable.
    tokenizer = None

    def tokenize(self, seqs: list[str], **tokenizer_kwargs) -> BatchEncoding:
        """Tokenize `seqs` with the attached tokenizer and move to the model device."""
        tok = self._require_tokenizer()
        defaults = {"return_tensors": "pt", "padding": True}
        defaults.update(tokenizer_kwargs)
        batch = tok(seqs, **defaults)
        try:
            # EsmcCompatMixin is always mixed into an nn.Module subclass.
            device = next(self.parameters()).device  # ty: ignore[unresolved-attribute]
        except StopIteration:
            device = torch.device("cpu")
        return batch.to(device)

    def encode(self, seqs: list[str], **tokenizer_kwargs) -> torch.Tensor:
        """ESM-C-compatible: return the padded `input_ids` only.

        Warning:
            This returns input IDs **without** the attention mask. If you feed
            the result into `forward()` without also passing `attention_mask`,
            pad tokens are treated as real input. Prefer `tokenize()` (returns
            the full `BatchEncoding`) or `logits()` (handles the mask plumbing
            internally).
        """
        return self.tokenize(seqs, **tokenizer_kwargs).input_ids

    def logits(self, seqs: list[str], config: LogitsConfig | None = None) -> LogitsOutput:
        """Run the model on `seqs` and return a structured `LogitsOutput`."""
        cfg = config or LogitsConfig()
        batch = self.tokenize(seqs)
        out = self(  # ty: ignore[call-non-callable]  # mixin is always mixed into nn.Module
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=(cfg.return_hidden_states or cfg.return_embeddings),
            output_attentions=cfg.return_attentions,
            return_dict=True,
        )
        embeddings = None
        if cfg.return_embeddings:
            # `OplmModel` exposes the post-final-norm state directly; task heads
            # only carry the per-layer tuple, so fall back to its last entry.
            embeddings = getattr(out, "last_hidden_state", None)
            if embeddings is None:
                embeddings = out.hidden_states[-1]
        return LogitsOutput(
            sequence_logits=getattr(out, "logits", None) if cfg.sequence else None,
            embeddings=embeddings,
            hidden_states=out.hidden_states if cfg.return_hidden_states else None,
            attentions=out.attentions if cfg.return_attentions else None,
        )

    def _require_tokenizer(self) -> PreTrainedTokenizerBase:
        if getattr(self, "tokenizer", None) is None:
            raise RuntimeError(
                "No tokenizer attached. Either load via "
                "`Oplm*.from_pretrained(<path-with-tokenizer-files>)` "
                "or assign one manually with "
                "`model.tokenizer = AutoTokenizer.from_pretrained(...)`."
            )
        # Narrowed non-None by the guard above; ty does not track it through getattr.
        return self.tokenizer  # ty: ignore[invalid-return-type]


class OplmModel(OplmPreTrainedModel, EsmcCompatMixin):
    """The OPLM encoder backbone — token embedding, N × OplmBlock, final norm.

    Returns a `BaseModelOutput`. Has no task head, so `logits()` returns
    `sequence_logits=None` (there is no `logits` attribute on the output).
    """

    def __init__(self, config: OplmConfig) -> None:
        super().__init__(config)
        self.backbone = OplmStack(config)
        self.tokenizer = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.embed_tokens.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.embed_tokens.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        last_hidden, hidden_states, attentions = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if not return_dict:
            return tuple(v for v in (last_hidden, hidden_states, attentions) if v is not None)
        return BaseModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class OplmMLMHead(nn.Module):
    """Masked-language-model head: dense -> activation -> norm -> decoder."""

    def __init__(self, config: OplmConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.act = _MLM_HEAD_ACTIVATIONS[config.mlm_head_activation]
        self.norm = make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        # The decoder writes to vocab space, NOT the residual stream: no 1/sqrt(2L) scaling.
        self.decoder._is_residual_writer = (
            False  # ty: ignore[unresolved-attribute]  # nn.Module setattr
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(x))))


class OplmForMaskedLM(OplmPreTrainedModel, EsmcCompatMixin):
    """OPLM with a masked-language-model head."""

    _tied_weights_keys = {
        "lm_head.decoder.weight": "oplm.backbone.embed_tokens.embed_tokens.weight"
    }

    def __init__(self, config: OplmConfig) -> None:
        super().__init__(config)
        self.oplm = OplmModel(config)
        self.lm_head = OplmMLMHead(config)
        self.tokenizer = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.oplm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.oplm.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> MaskedLMOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.oplm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class OplmForSequenceClassification(OplmPreTrainedModel, EsmcCompatMixin):
    """OPLM with a pooled sequence-classification head."""

    def __init__(self, config: OplmConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.oplm = OplmModel(config)
        if config.pre_head_norm:
            self.pre_head_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
        else:
            self.pre_head_norm = nn.Identity()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.tokenizer = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.oplm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.oplm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.oplm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        if self.config.classifier_pool == "cls":
            pooled = cls_pool(last_hidden)
        else:
            pooled_mask = self._pooling_mask(attention_mask, last_hidden)
            pooled = mean_pool(last_hidden, pooled_mask)

        pooled = self.pre_head_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).float()

        loss = self._compute_loss(logits, labels)

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _pooling_mask(attention_mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
        if attention_mask is not None:
            return attention_mask
        return hidden.new_ones(hidden.shape[:2], dtype=torch.long)

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Loss with HF-style `problem_type` inference (regression / classification)."""
        if labels is None:
            return None

        problem_type = self.config.problem_type
        if problem_type is None:
            if self.num_labels == 1:
                problem_type = "regression"
            elif labels.dtype in (torch.long, torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
            self.config.problem_type = problem_type

        if problem_type == "regression":
            if self.num_labels == 1:
                return F.mse_loss(logits.squeeze(-1), labels.squeeze(-1))
            return F.mse_loss(logits, labels)
        if problem_type == "single_label_classification":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class OplmForTokenClassification(OplmPreTrainedModel, EsmcCompatMixin):
    """OPLM with a per-token classification head."""

    def __init__(self, config: OplmConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.oplm = OplmModel(config)
        if config.pre_head_norm:
            self.pre_head_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
        else:
            self.pre_head_norm = nn.Identity()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.tokenizer = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.oplm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.oplm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.oplm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = self.pre_head_norm(outputs.last_hidden_state)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
