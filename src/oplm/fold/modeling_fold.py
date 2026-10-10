"""OplmForFolding — the HF model that owns the heads and a frozen, unregistered language model.

Composition and data flow follow Biohub's ESMFold2 ``EsmFold2Model.forward``
(esm/models/esmfold2/model.py, Apache-2.0; see THIRD_PARTY_NOTICES.md); parameter names are
the released checkpoint's (``input_embedder``, ``language_model``, ``lm_encoder``,
``folding_trunk``, ``parcae``, ``distogram_head``, ``structure_head``, ``confidence_head``).
Modifications: the LM is any ``OplmModel`` (or precomputed hidden states) held *outside* the
module tree (spec §6.6); loop count, gradient loops, initial state and the sampler RNG are
explicit; per-loop LM-pair dropout is training-only; the generic HF init respects the port's
zero/identity/−2 initialisations through module tags (spec §5.4).

Loading a fold checkpoint requires ``oplm`` to be installed (``import oplm.fold`` registers
``oplm_fold`` with the Auto classes); ``trust_remote_code`` bundling is not provided in this
milestone.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel

from oplm.fold.atoms import InputsEmbedder
from oplm.fold.attention import (
    ensure_flex_recompile_limit,
    resolve_attention_backend,
    sliding_window_block_mask,
)
from oplm.fold.confidence import ConfidenceHead, ConfidenceOutput, symmetrized_distogram
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.diffusion import StructureHead
from oplm.fold.lm_shim import LanguageModelShim, frozen_lm_hidden_states, lm_state_count
from oplm.fold.trunk import PairStack, Recurrence, pair_stack_kwargs
from oplm.model import OplmModel

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from oplm.fold.data.featurize import FoldFeatures

__all__ = ["FoldOutput", "OplmFoldPreTrainedModel", "OplmForFolding", "load_frozen_lm"]


class OplmFoldPreTrainedModel(PreTrainedModel):
    """HF plumbing for fold models: config class, init policy, checkpointing hooks."""

    config_class = FoldConfig
    base_model_prefix = "oplm_fold"
    main_input_name = "lm_input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PairUpdateBlock", "DiffusionBlock", "AtomBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        """Generic init plus the port's tagged exceptions; never touches loaded tensors."""
        if isinstance(module, nn.Linear):
            if getattr(module, "_init_zero", False):
                nn.init.zeros_(module.weight)
            elif not getattr(module, "_init_identity", False):  # identity is applied by its owner
                nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.constant_(module.bias, float(getattr(module, "_init_bias", 0.0)))
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
        elif isinstance(module, Recurrence):
            module.reset_parameters()
        elif isinstance(module, LanguageModelShim):
            nn.init.zeros_(module.layer_weights)
        elif isinstance(module, ConfidenceHead):
            nn.init.zeros_(module.plddt_weight)
            nn.init.zeros_(module.resolved_weight)


@dataclass
class FoldOutput:
    """What one forward pass produces (``coords``/``confidence`` have ``B · num_samples`` rows)."""

    distogram_logits: Tensor  # (B, L, L, bins) fp32, symmetric
    coords: Tensor  # (B·S, A, 3) Å, padded atoms meaningless
    confidence: ConfidenceOutput
    s_inputs: Tensor  # (B, L, single_inputs_width)
    pair: Tensor  # (B, L, L, pair) fp32 final pair (after readout + coda)
    intermediates: dict[str, Any] | None = None


def load_frozen_lm(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> OplmModel:
    """Load the bare OPLM encoder (local dir, ``<ckpt>/hf``, or Hub id) frozen in eval mode.

    Raises:
        ValueError: ``name_or_path`` has the ``<repo>#esmc`` form written by
            ``fold_config_from_upstream``: the head was trained against the ESMC bundled in
            that repo, which is not an OPLM; pass ``lm_hidden_states`` to ``forward`` instead.
    """
    if str(name_or_path).endswith("#esmc"):
        raise ValueError(
            f"{name_or_path!r} names a language model bundled in an upstream checkpoint, not an "
            "OPLM; run the head with precomputed lm_hidden_states (see docs/FOLD.md §7)"
        )
    lm = OplmModel.from_pretrained(str(name_or_path), revision=revision)
    if dtype is not None:
        lm = lm.to(dtype)
    if device is not None:
        lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm


class OplmForFolding(OplmFoldPreTrainedModel):
    """Frozen LM -> shim -> recurrent pair trunk -> distogram, diffusion and confidence heads."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__(config)
        kw = pair_stack_kwargs(config)
        self.input_embedder = InputsEmbedder(config)
        self.language_model = LanguageModelShim(config)
        self.lm_encoder = PairStack(config.lm_encoder_blocks, config.pair_width, **kw)
        self.folding_trunk = PairStack(config.trunk_blocks, config.pair_width, **kw)
        self.parcae = Recurrence(config.pair_width, coda_blocks=config.coda_blocks, **kw)
        self.distogram_head = nn.Linear(config.pair_width, config.distogram_bins, bias=True)
        self.structure_head = StructureHead(config)
        self.confidence_head = ConfidenceHead(config)
        object.__setattr__(self, "_lm", None)  # bypass nn.Module registration (spec §6.6)
        self.post_init()

    # --- frozen LM ownership -------------------------------------------------------------

    @property
    def lm(self) -> nn.Module | None:
        """The attached frozen LM, or ``None``; its device and dtype are the caller's."""
        return self._lm

    def attach_lm(
        self, lm: nn.Module, *, name_or_path: str | None = None, revision: str | None = None
    ) -> None:
        """Attach a frozen LM after checking it against the config's shape contract.

        The caller owns the LM's device and dtype: it is frozen and put in eval mode but never
        moved or cast (``forward`` runs it where it lives). Only the auto-resolution in
        :meth:`from_pretrained` and ``oplm fold predict`` cast it to bf16 on CUDA.

        Raises:
            ValueError: ``lm.config.hidden_size`` or its hidden-state count disagrees with
                ``lm_hidden_size`` / ``lm_num_hidden_states``.
        """
        hidden = int(lm.config.hidden_size)  # ty: ignore[unresolved-attribute, invalid-argument-type]  # nn.Module attrs are Tensor | Module
        if hidden != self.config.lm_hidden_size:
            raise ValueError(
                f"lm_hidden_size mismatch: config expects {self.config.lm_hidden_size}, "
                f"LM has {hidden}"
            )
        states = lm_state_count(lm)
        if states != self.config.lm_num_hidden_states:
            raise ValueError(
                "lm_num_hidden_states mismatch: config expects "
                f"{self.config.lm_num_hidden_states}, LM produces {states}"
            )
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_lm", lm)
        if name_or_path is not None:
            self.config.lm_name_or_path = name_or_path
        if revision is not None:
            self.config.lm_revision = revision

    def train(self, mode: bool = True) -> OplmForFolding:
        super().train(mode)
        if self._lm is not None:
            self._lm.eval()
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> OplmForFolding:
        """Deep-copy the head; share the frozen LM (``build_ema`` must not duplicate it)."""
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            object.__setattr__(clone, name, value if name == "_lm" else copy.deepcopy(value, memo))
        return clone

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path | None,
        *args: Any,
        lm: nn.Module | None = None,
        lm_name_or_path: str | Path | None = None,
        lm_revision: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load the head, then attach ``lm`` or resolve the LM from the config (override wins).

        No LM is attached when neither is given and ``config.lm_name_or_path`` is unset or has
        the ``<repo>#esmc`` form (a head trained against upstream's bundled ESMC); call
        :meth:`attach_lm` or pass ``lm_hidden_states`` to ``forward``. An auto-resolved LM goes
        to the head's device, in bf16 when that is CUDA; an ``lm`` passed in keeps the caller's
        device and dtype.
        """
        result = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        source = lm_name_or_path or model.config.lm_name_or_path
        if lm_name_or_path is None and str(source).endswith("#esmc"):
            source = None  # not an OPLM: run the head with precomputed lm_hidden_states
        if lm is not None:
            model.attach_lm(lm)
        elif source is not None:
            revision = lm_revision or (
                model.config.lm_revision if lm_name_or_path is None else None
            )
            dtype = torch.bfloat16 if model.device.type == "cuda" else None
            model.attach_lm(
                load_frozen_lm(source, revision=revision, device=model.device, dtype=dtype)
            )
        return result

    # --- forward -------------------------------------------------------------------------

    def forward(
        self,
        features: FoldFeatures,
        *,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_samples: int | None = None,
        num_steps: int | None = None,
        z0: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
    ) -> FoldOutput:
        """Predict a structure for ``features`` (batch size 1).

        Args:
            features: Output of :func:`oplm.fold.featurize`.
            lm_hidden_states: ``(1, L, K, D)`` precomputed LM states (skips the attached LM).
            num_loops: Recurrence iterations; default ``config.inference_num_loops``.
            num_samples: Diffusion samples; default ``config.inference_num_samples``.
            num_steps: Sampler steps before the sigma cap; default ``config.inference_num_steps``.
            z0: Initial pair state; drawn from ``generator`` when ``None``.
            generator: RNG for the initial state and the sampler (deterministic when set);
                must live on the model's device (a CPU generator fails with CUDA tensors).
            return_intermediates: Also return the stage tensors used by the parity tests.

        Raises:
            RuntimeError: No LM is attached and ``lm_hidden_states`` is ``None``.
        """
        cfg = self.config
        f = features.to(self.device)
        if lm_hidden_states is None:
            if self._lm is None:
                raise RuntimeError(
                    "no language model attached: call attach_lm(), load with lm_name_or_path, "
                    "or pass lm_hidden_states"
                )
            lm_device = next(self._lm.parameters()).device
            lm_dtype = torch.bfloat16 if lm_device.type == "cuda" else None
            lm_hidden_states = frozen_lm_hidden_states(self._lm, f.to(lm_device), dtype=lm_dtype)
        lm_hidden_states = lm_hidden_states.to(self.device)
        loops = cfg.inference_num_loops if num_loops is None else num_loops
        samples = cfg.inference_num_samples if num_samples is None else num_samples
        on_cuda = self.device.type == "cuda"
        if on_cuda:
            ensure_flex_recompile_limit()
        flex = on_cuda and resolve_attention_backend(f.ref_pos, cfg.attention_backend) == "flex"
        atom_block_mask = (
            sliding_window_block_mask(f.atom_mask, cfg.atom_window // 2) if flex else None
        )
        pair_mask = f.token_mask[:, :, None].float() * f.token_mask[:, None, :].float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on_cuda):
            emb = self.input_embedder(f, block_mask=atom_block_mask)
            lm_z = self.language_model(lm_hidden_states)
            if z0 is None:
                z0 = self.parcae.init_state(emb.z_init, generator=generator)
            drop = cfg.pair_dropout if self.training else 0.0

            def inject(_t: int) -> Tensor:
                lm_t = F.dropout(lm_z, drop, training=True) if drop > 0 else lm_z
                return emb.z_init + self.lm_encoder(lm_t.to(emb.z_init.dtype), pair_mask)

            z, states = self.parcae.run(
                self.folding_trunk,
                inject,
                z0=z0.to(emb.z_init.dtype),
                pair_mask=pair_mask,
                num_loops=loops,
                grad_loops=cfg.recurrence_grad_loops if self.training else None,
                return_states=return_intermediates,
            )
            z = self.parcae.readout(z, pair_mask)
        z = z.float()
        distogram_logits = symmetrized_distogram(self.distogram_head, z)
        inp = self.structure_head.prepare(
            s_inputs=emb.s_inputs,
            z_trunk=z,
            relpos=emb.relpos,
            atom_features=emb.atom_features,
            rope=emb.rope,
            atom_mask=f.atom_mask,
            atom_to_token=f.atom_to_token,
            token_mask=f.token_mask,
            num_samples=samples,
        )
        coords = self.structure_head.sample(inp, num_steps=num_steps, generator=generator)
        confidence = self.confidence_head(
            s_inputs=emb.s_inputs,
            z=z,
            relpos=emb.relpos,
            bonds=emb.bonds,
            coords=coords,
            distogram_atom_idx=f.distogram_atom_idx,
            token_mask=f.token_mask,
            atom_to_token=f.atom_to_token,
            atom_mask=f.atom_mask,
            asym_id=f.asym_id,
        )
        intermediates = None
        if return_intermediates:
            intermediates = {
                "s_inputs": emb.s_inputs,
                "z_init": emb.z_init,
                "relpos": emb.relpos,
                "bonds": emb.bonds,
                "lm_z": lm_z,
                "z0": z0,
                "states": states,
                "pair": z,
                "conditioned_pair": inp.z,
            }
        return FoldOutput(
            distogram_logits=distogram_logits,
            coords=coords,
            confidence=confidence,
            s_inputs=emb.s_inputs,
            pair=z,
            intermediates=intermediates,
        )


AutoConfig.register("oplm_fold", FoldConfig, exist_ok=True)
AutoModel.register(FoldConfig, OplmForFolding, exist_ok=True)
