"""OplmForFolding — the HF model that owns the heads and a frozen, unregistered language model.

Composition and data flow follow Biohub's ESMFold2 ``EsmFold2Model.forward``
(esm/models/esmfold2/model.py, Apache-2.0; see THIRD_PARTY_NOTICES.md); parameter names are
the released checkpoint's (``input_embedder``, ``language_model``, ``lm_encoder``,
``folding_trunk``, ``parcae``, ``distogram_head``, ``structure_head``, ``confidence_head``).
Modifications: the LM is any ``OplmModel`` (or precomputed hidden states) held *outside* the
module tree (spec §6.6); loop count, gradient loops, initial state and the sampler RNG are
explicit; per-loop LM-pair dropout is training-only; the generic HF init respects the port's
zero/identity/−2 initialisations through module tags (spec §5.4).

Training branch (spec §6.1): ``forward(features, targets=..., train=TrainSettings(...))`` runs
the same prefix as inference (steps 1-3: in-model LM-input masking of ``lm_input_mask_fraction``
inner residues, a per-rank clamped Poisson loop count drawn from the global RNG, the recurrent
trunk with truncated gradients, the distogram head) and then, instead of sampling: (4) denoises
``diffusion_samples`` noised copies of the ground truth per example in microbatches; when
``diffusion_microbatch < diffusion_samples`` each denoiser call runs under activation
checkpointing so only one chunk is live at a time; (5) rolls the sampler out under ``no_grad``
from the shared conditioning and runs the confidence head on detached trunk outputs and the
rollout; (6) returns the weighted sum of the diffusion, smooth-lDDT, distogram and confidence
losses with per-term metrics. The branch runs whenever ``targets`` is given; train-mode
behaviour (dropout, masking, the Poisson draw) still follows ``self.training``.

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
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel, PreTrainedModel

from oplm.fold.atoms import InputsEmbedder
from oplm.fold.attention import (
    ensure_flex_recompile_limit,
    resolve_attention_backend,
    sliding_window_block_mask,
)
from oplm.fold.confidence import ConfidenceHead, ConfidenceOutput, symmetrized_distogram
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.diffusion import StructureHead, center_random_augmentation
from oplm.fold.esmc import parse_esmc_name
from oplm.fold.lm_shim import (
    LanguageModelShim,
    frozen_lm_hidden_states,
    lm_state_count,
    mask_lm_inputs,
)
from oplm.fold.losses import (
    LossWeights,
    diffusion_mse,
    distogram_loss,
    edm_weight,
    pae_loss,
    pde_loss,
    plddt_loss,
    resolved_loss,
    smooth_lddt,
)
from oplm.fold.trunk import PairStack, Recurrence, fp32_autocast_off, pair_stack_kwargs
from oplm.model import OplmModel

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from oplm.fold.atoms import InputsEmbedding
    from oplm.fold.data.featurize import FoldFeatures, FoldTargets
    from oplm.fold.diffusion import DenoiserInputs

__all__ = [
    "FoldOutput",
    "FoldTrainOutput",
    "OplmFoldPreTrainedModel",
    "OplmForFolding",
    "TrainSettings",
    "draw_num_loops",
    "load_frozen_lm",
]


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


@dataclass(frozen=True)
class TrainSettings:
    """Per-run training knobs the forward pass needs (the ``fold:`` block; addendum §5)."""

    diffusion_samples: int = 8
    diffusion_microbatch: int | None = None  # samples per denoiser call; None = all at once
    confidence_rollout_steps: int = 20
    weights: LossWeights = LossWeights()

    def __post_init__(self) -> None:
        if self.diffusion_samples < 1 or self.confidence_rollout_steps < 1:
            raise ValueError("diffusion_samples and confidence_rollout_steps must be >= 1")
        if self.diffusion_microbatch is not None and self.diffusion_microbatch < 1:
            raise ValueError("diffusion_microbatch must be >= 1 or None")


@dataclass
class FoldTrainOutput:
    """What the training branch returns: the loss to backpropagate and the Trainer's accounting."""

    loss: Tensor
    metrics: dict[str, float]
    tokens: int
    chains: int


def draw_num_loops(mean: float, min_loops: int, max_loops: int) -> int:
    """One clamped Poisson draw from the global RNG (spec §5.1: mean 3, clamped to 1-6)."""
    draw = int(torch.poisson(torch.tensor(float(mean))).item())
    return max(min_loops, min(max_loops, draw))


def load_frozen_lm(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Load a frozen encoder: ``esmc:<name>`` loads the frozen ESM-C adapter; OPLM paths and Hub ids
    load the bare encoder; the ``<repo>#esmc`` form raises.

    Raises:
        ValueError: ``name_or_path`` has the ``<repo>#esmc`` form written by
            ``fold_config_from_upstream``: the head was trained against the ESMC bundled in
            that repo, which is not an OPLM; pass ``lm_hidden_states`` to ``forward`` instead.
        ImportError: ``name_or_path`` has the ``esmc:`` prefix but the ``esm`` package is
            missing (install ``oplm[esmc]`` on Python >= 3.12).
    """
    esmc_name = parse_esmc_name(str(name_or_path))
    if esmc_name is not None:
        from oplm.fold.esmc import load_esmc

        return load_esmc(esmc_name, revision=revision, device=device, dtype=dtype)
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
        targets: FoldTargets | None = None,
        train: TrainSettings | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_samples: int | None = None,
        num_steps: int | None = None,
        z0: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
    ) -> FoldOutput | FoldTrainOutput:
        """Predict a structure, or with ``targets`` run the training branch (spec §6.1).

        Args:
            features: Output of :func:`oplm.fold.featurize` or a collated batch.
            targets: Supervision aligned with ``features``; selects the training branch, which
                returns :class:`FoldTrainOutput`. ``train`` supplies its settings (defaults
                otherwise) and is ignored without ``targets``.
            train: Sample count, microbatch, rollout steps and loss weights.
            lm_hidden_states: ``(B, L, K, D)`` precomputed LM states (skips the attached LM; in
                train mode the LM-input masking still consumes the global RNG).
            num_loops: Recurrence iterations; default ``config.inference_num_loops`` in eval mode
                and a clamped Poisson draw in train mode.
            num_samples: Diffusion samples (inference); default ``config.inference_num_samples``.
            num_steps: Sampler steps before the sigma cap; default ``config.inference_num_steps``.
                Ignored when ``targets`` is given (``train.confidence_rollout_steps`` applies).
            z0: Initial pair state; drawn from ``generator`` when ``None``.
            generator: RNG for the initial state and the samplers (deterministic when set);
                must live on the model's device (a CPU generator fails with CUDA tensors).
            return_intermediates: Also return the stage tensors used by the parity tests;
                ignored when ``targets`` is given.

        Raises:
            RuntimeError: No LM is attached and ``lm_hidden_states`` is ``None``.
        """
        cfg = self.config
        f = features.to(self.device)
        if self.training:
            f = mask_lm_inputs(f, cfg.lm_input_mask_fraction, cfg.lm_mask_token_id)
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
        if num_loops is not None:
            loops = num_loops
        elif self.training:
            loops = draw_num_loops(
                cfg.recurrence_poisson_mean, cfg.recurrence_min_loops, cfg.recurrence_max_loops
            )
        else:
            loops = cfg.inference_num_loops
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
        # accelerate's outer autocast would otherwise make these logits bf16 (spec §5.4: fp32)
        with fp32_autocast_off(self.device):
            distogram_logits = symmetrized_distogram(self.distogram_head, z)
        if targets is not None:
            return self._train_branch(
                f, targets, train or TrainSettings(), emb, z, distogram_logits, loops, generator
            )
        samples = cfg.inference_num_samples if num_samples is None else num_samples
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

    def _train_branch(
        self,
        f: FoldFeatures,
        targets: FoldTargets,
        s: TrainSettings,
        emb: InputsEmbedding,
        z: Tensor,
        distogram_logits: Tensor,
        loops: int,
        generator: torch.Generator | None,
    ) -> FoldTrainOutput:
        """Spec §6.1 steps 3-6 on the shared prefix (see the module docstring for the design).

        The pLDDT target scores each atom against the polymer tokens' centre atoms (CA; AF3 SI
        4.3.1, Boltz-2 ``r_set_to_rep_atom``), taken from ``frame_atom_idx[..., 1]``; the PDE
        and distogram targets use the representative atom (CB, CA for glycine).
        """
        cfg, w = self.config, s.weights
        device = z.device
        B = z.shape[0]
        t = targets.to(device)
        with fp32_autocast_off(device):
            resolved = t.resolved & f.atom_mask
            disto = distogram_loss(
                distogram_logits,
                t.coords,
                resolved,
                f.token_mask,
                f.distogram_atom_idx,
                min_dist=cfg.distogram_min_dist,
                max_dist=cfg.distogram_max_dist,
            )
            base = self.structure_head.prepare(
                s_inputs=emb.s_inputs,
                z_trunk=z,
                relpos=emb.relpos,
                atom_features=emb.atom_features,
                rope=emb.rope,
                atom_mask=f.atom_mask,
                atom_to_token=f.atom_to_token,
                token_mask=f.token_mask,
                num_samples=1,
            )
            # Diffusion: S samples per example, rows b·S + j; one denoiser call per microbatch
            # of samples, checkpointed so only one chunk's activations are live at a time.
            S = s.diffusion_samples
            micro = S if s.diffusion_microbatch is None else min(s.diffusion_microbatch, S)
            x_gt = t.coords.float().repeat_interleave(S, 0)
            res_s = resolved.repeat_interleave(S, 0)
            w_s = res_s.float()
            x_aug = center_random_augmentation(x_gt, w_s)
            sigma = cfg.sigma_data * torch.exp(
                cfg.train_noise_log_mean
                + cfg.train_noise_log_std * torch.randn(B * S, device=device)
            )
            x_noisy = x_aug + sigma[:, None, None] * torch.randn_like(x_aug)
            x_den = torch.zeros_like(x_noisy)
            use_ckpt = self.training and torch.is_grad_enabled() and micro < S
            for start in range(0, S, micro):
                m = min(micro, S - start)
                inp = self.structure_head.expand(base, m)
                index = (
                    torch.arange(B, device=device)[:, None] * S
                    + torch.arange(start, start + m, device=device)[None]
                ).reshape(-1)

                def run(x: Tensor, tt: Tensor, inp: DenoiserInputs = inp) -> Tensor:
                    return self.structure_head.denoise(x, tt, inp)

                pred = (
                    checkpoint(run, x_noisy[index], sigma[index], use_reentrant=False)
                    if use_ckpt
                    else run(x_noisy[index], sigma[index])
                )
                x_den = x_den.index_put((index,), pred)
            mse = diffusion_mse(x_den, x_aug, w_s)
            lddt = smooth_lddt(x_den, x_aug, res_s)
            diffusion_term = (edm_weight(sigma, cfg.sigma_data) * mse).mean()
            lddt_term = lddt.mean()
            # Confidence: a no-grad rollout, then the head on detached trunk outputs.
            with torch.no_grad():
                rollout = self.structure_head.sample(
                    base, num_steps=s.confidence_rollout_steps, generator=generator
                )
            conf = self.confidence_head(
                s_inputs=emb.s_inputs.detach(),
                z=z.detach(),
                relpos=emb.relpos.detach(),
                bonds=emb.bonds.detach(),
                coords=rollout,
                distogram_atom_idx=f.distogram_atom_idx,
                token_mask=f.token_mask,
                atom_to_token=f.atom_to_token,
                atom_mask=f.atom_mask,
                asym_id=f.asym_id,
            )
            plddt = plddt_loss(
                conf.plddt_logits,
                rollout,
                t.coords,
                resolved,
                f.atom_mask,
                f.token_mask,
                t.frame_atom_idx[..., 1],
            )
            pde = pde_loss(
                conf.pde_logits,
                rollout,
                t.coords,
                resolved,
                f.token_mask,
                f.distogram_atom_idx,
                max_dist=cfg.pae_max_dist,
            )
            pae = pae_loss(
                conf.pae_logits,
                rollout,
                t.coords,
                resolved,
                f.token_mask,
                t.frame_atom_idx,
                max_dist=cfg.pae_max_dist,
            )
            res = resolved_loss(conf.resolved_logits, resolved, f.atom_mask)
            confidence = (plddt + pde + res + w.pae * pae).mean()
            loss = (
                w.diffusion_mse * diffusion_term
                + w.smooth_lddt * lddt_term
                + w.distogram * disto.mean()
                + w.confidence * confidence
            )
        metrics = {
            "diffusion_mse": diffusion_term.item(),
            "smooth_lddt": lddt_term.item(),
            "distogram": disto.mean().item(),
            "confidence": confidence.item(),
            "plddt": plddt.mean().item(),
            "pde": pde.mean().item(),
            "pae": pae.mean().item(),
            "resolved": res.mean().item(),
            "loops": float(loops),
            "sigma_mean": sigma.mean().item(),
        }
        # chains == B: milestone-2 examples are single-chain, so one example is one chain
        return FoldTrainOutput(loss=loss, metrics=metrics, tokens=int(f.token_mask.sum()), chains=B)


AutoConfig.register("oplm_fold", FoldConfig, exist_ok=True)
AutoModel.register(FoldConfig, OplmForFolding, exist_ok=True)
