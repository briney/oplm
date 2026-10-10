"""FoldConfig — PretrainedConfig for the OPLM structure prediction head.

Defaults are the released ``biohub/ESMFold2-Fast`` configuration so its weights load for
parity; training experiments override explicitly. The frozen language model is not part of
the fold model's parameters; this config records which LM the head was built against
(spec §6.6) and the shape facts the shim needs.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

__all__ = ["FoldConfig"]

_VALID_TRIMUL_BACKENDS = ("auto", "fused", "fused_forward_reference_backward", "reference")
_VALID_ATTENTION_BACKENDS = ("auto", "dense", "flex")


class FoldConfig(PretrainedConfig):
    """Configuration for :class:`~oplm.fold.modeling_fold.OplmForFolding`.

    Widths and block counts follow ESMFold2-Fast (docs/FOLD.md §2). ``lm_*`` fields describe
    the frozen LM: ``lm_num_hidden_states`` is the number of per-layer states the shim mixes
    (OPLM: ``len(layer_execution_order) + 1``; ESMC-6B: 81). ``pair_dropout`` is the per-loop
    elementwise dropout on the LM pair (upstream ``lm_dropout``, training only in this port);
    ``trunk_dropout`` is the row-shared residual dropout inside every pair-update block
    (upstream ``DropoutResidual``, 0 in the release). ``inference_num_loops`` counts iterations
    executed (upstream's ``num_loops + 1``; spec default 10, the release runs 21).
    """

    model_type = "oplm_fold"

    def __init__(
        self,
        *,
        lm_name_or_path: str | None = None,
        lm_revision: str | None = None,
        lm_hidden_size: int = 768,
        lm_num_hidden_states: int = 25,
        lm_max_chain_length: int | None = None,
        lm_pad_token_id: int = 1,
        lm_bos_token_id: int = 0,
        lm_eos_token_id: int = 2,
        lm_mask_token_id: int = 32,
        lm_input_mask_fraction: float = 0.1,
        num_res_types: int = 33,
        max_atomic_number: int = 128,
        atom_name_chars: int = 4,
        atom_name_vocab: int = 64,
        atom_pad_multiple: int = 32,
        max_atoms_per_token: int = 23,
        pair_width: int = 256,
        token_width: int = 768,
        atom_width: int = 128,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        atom_window: int = 128,
        spatial_rope_base: float = 20.0,
        spatial_rope_pairs_per_axis: int = 2,
        uid_rope_pairs: int = 10,
        uid_rope_base: float = 10000.0,
        relpos_r_max: int = 32,
        relpos_s_max: int = 2,
        trunk_blocks: int = 24,
        lm_encoder_blocks: int = 4,
        coda_blocks: int = 2,
        transition_expansion: int = 4,
        pair_dropout: float = 0.25,
        trunk_dropout: float = 0.0,
        recurrence_poisson_mean: float = 3.0,
        recurrence_min_loops: int = 1,
        recurrence_max_loops: int = 6,
        recurrence_grad_loops: int = 2,
        inference_num_loops: int = 10,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        diffusion_blocks: int = 12,
        diffusion_heads: int = 16,
        diffusion_transition_multiplier: int = 2,
        diffusion_atom_blocks: int = 3,
        diffusion_atom_heads: int = 4,
        train_noise_log_mean: float = -1.2,
        train_noise_log_std: float = 1.5,
        inference_num_steps: int = 14,
        inference_sigma_max: float = 160.0,
        inference_sigma_min: float = 4e-4,
        inference_rho: float = 7.0,
        inference_sigma_cap: float = 256.0,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        inference_num_samples: int = 1,
        distogram_bins: int = 64,
        confidence_blocks: int = 4,
        plddt_bins: int = 50,
        pae_bins: int = 64,
        pde_bins: int = 64,
        pae_max_dist: float = 32.0,
        confidence_dist_bins: int = 39,
        confidence_min_dist: float = 3.25,
        confidence_max_dist: float = 50.75,
        trimul_backend: str = "auto",
        trimul_chunk_size: int | None = 64,
        attention_backend: str = "auto",
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        self.lm_name_or_path = lm_name_or_path
        self.lm_revision = lm_revision
        self.lm_hidden_size = int(lm_hidden_size)
        self.lm_num_hidden_states = int(lm_num_hidden_states)
        self.lm_max_chain_length = None if lm_max_chain_length is None else int(lm_max_chain_length)
        self.lm_pad_token_id = int(lm_pad_token_id)
        self.lm_bos_token_id = int(lm_bos_token_id)
        self.lm_eos_token_id = int(lm_eos_token_id)
        self.lm_mask_token_id = int(lm_mask_token_id)
        self.lm_input_mask_fraction = float(lm_input_mask_fraction)
        self.num_res_types = int(num_res_types)
        self.max_atomic_number = int(max_atomic_number)
        self.atom_name_chars = int(atom_name_chars)
        self.atom_name_vocab = int(atom_name_vocab)
        self.atom_pad_multiple = int(atom_pad_multiple)
        self.max_atoms_per_token = int(max_atoms_per_token)
        self.pair_width = int(pair_width)
        self.token_width = int(token_width)
        self.atom_width = int(atom_width)
        self.atom_encoder_blocks = int(atom_encoder_blocks)
        self.atom_encoder_heads = int(atom_encoder_heads)
        self.atom_window = int(atom_window)
        self.spatial_rope_base = float(spatial_rope_base)
        self.spatial_rope_pairs_per_axis = int(spatial_rope_pairs_per_axis)
        self.uid_rope_pairs = int(uid_rope_pairs)
        self.uid_rope_base = float(uid_rope_base)
        self.relpos_r_max = int(relpos_r_max)
        self.relpos_s_max = int(relpos_s_max)
        self.trunk_blocks = int(trunk_blocks)
        self.lm_encoder_blocks = int(lm_encoder_blocks)
        self.coda_blocks = int(coda_blocks)
        self.transition_expansion = int(transition_expansion)
        self.pair_dropout = float(pair_dropout)
        self.trunk_dropout = float(trunk_dropout)
        self.recurrence_poisson_mean = float(recurrence_poisson_mean)
        self.recurrence_min_loops = int(recurrence_min_loops)
        self.recurrence_max_loops = int(recurrence_max_loops)
        self.recurrence_grad_loops = int(recurrence_grad_loops)
        self.inference_num_loops = int(inference_num_loops)
        self.sigma_data = float(sigma_data)
        self.fourier_dim = int(fourier_dim)
        self.diffusion_blocks = int(diffusion_blocks)
        self.diffusion_heads = int(diffusion_heads)
        self.diffusion_transition_multiplier = int(diffusion_transition_multiplier)
        self.diffusion_atom_blocks = int(diffusion_atom_blocks)
        self.diffusion_atom_heads = int(diffusion_atom_heads)
        self.train_noise_log_mean = float(train_noise_log_mean)
        self.train_noise_log_std = float(train_noise_log_std)
        self.inference_num_steps = int(inference_num_steps)
        self.inference_sigma_max = float(inference_sigma_max)
        self.inference_sigma_min = float(inference_sigma_min)
        self.inference_rho = float(inference_rho)
        self.inference_sigma_cap = float(inference_sigma_cap)
        self.gamma_0 = float(gamma_0)
        self.gamma_min = float(gamma_min)
        self.noise_scale = float(noise_scale)
        self.step_scale = float(step_scale)
        self.inference_num_samples = int(inference_num_samples)
        self.distogram_bins = int(distogram_bins)
        self.confidence_blocks = int(confidence_blocks)
        self.plddt_bins = int(plddt_bins)
        self.pae_bins = int(pae_bins)
        self.pde_bins = int(pde_bins)
        self.pae_max_dist = float(pae_max_dist)
        self.confidence_dist_bins = int(confidence_dist_bins)
        self.confidence_min_dist = float(confidence_min_dist)
        self.confidence_max_dist = float(confidence_max_dist)
        self.trimul_backend = str(trimul_backend)
        self.trimul_chunk_size = None if trimul_chunk_size is None else int(trimul_chunk_size)
        self.attention_backend = str(attention_backend)
        self.layer_norm_eps = float(layer_norm_eps)
        self._validate()
        super().__init__(**kwargs)

    # Derived widths (ESMFold2: 768 // 2 + 33 + 33 + 1 = 451; 3 + 1 + 1 + 128 + 4 * 64 = 389;
    # 2 * (2 * 32 + 2) + 1 + (2 * 2 + 2) = 139).
    @property
    def inputs_token_width(self) -> int:
        """Width of the inputs-embedder atom aggregation (half the diffusion token width)."""
        return self.token_width // 2

    @property
    def single_inputs_width(self) -> int:
        """Width of ``s_inputs``: atom aggregation + res-type one-hot + profile + deletion mean."""
        return self.inputs_token_width + 2 * self.num_res_types + 1

    @property
    def atom_feature_dim(self) -> int:
        """Width of the per-atom reference features."""
        return 3 + 1 + 1 + self.max_atomic_number + self.atom_name_chars * self.atom_name_vocab

    @property
    def relpos_feature_dim(self) -> int:
        """Width of the relative-position one-hot features."""
        return 2 * (2 * self.relpos_r_max + 2) + 1 + (2 * self.relpos_s_max + 2)

    def _validate(self) -> None:
        if self.pair_width <= 0 or self.pair_width % 32:
            raise ValueError(
                f"pair_width must be a positive multiple of 32; got {self.pair_width!r}."
            )
        if self.token_width % 2 or self.token_width % self.diffusion_heads:
            raise ValueError(
                "token_width must be even and divisible by diffusion_heads; "
                f"got {self.token_width!r} / {self.diffusion_heads!r}."
            )
        if self.atom_width % self.atom_encoder_heads or self.atom_width % self.diffusion_atom_heads:
            raise ValueError(
                "atom_width must be divisible by atom_encoder_heads and diffusion_atom_heads; "
                f"got {self.atom_width!r}."
            )
        head_dim = self.atom_width // self.atom_encoder_heads
        if 3 * self.spatial_rope_pairs_per_axis + self.uid_rope_pairs > head_dim // 2:
            raise ValueError(
                "3 * spatial_rope_pairs_per_axis + uid_rope_pairs must fit head_dim // 2."
            )
        for name in (
            "lm_hidden_size",
            "lm_num_hidden_states",
            "atom_window",
            "trunk_blocks",
            "lm_encoder_blocks",
            "coda_blocks",
            "transition_expansion",
            "recurrence_min_loops",
            "recurrence_max_loops",
            "recurrence_grad_loops",
            "inference_num_loops",
            "fourier_dim",
            "diffusion_blocks",
            "diffusion_atom_blocks",
            "inference_num_steps",
            "inference_num_samples",
            "distogram_bins",
            "confidence_blocks",
            "plddt_bins",
            "pae_bins",
            "pde_bins",
            "confidence_dist_bins",
            "atom_pad_multiple",
            "max_atoms_per_token",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1; got {getattr(self, name)!r}.")
        if self.recurrence_max_loops < self.recurrence_min_loops:
            raise ValueError("recurrence_max_loops must be >= recurrence_min_loops.")
        for name in ("pair_dropout", "trunk_dropout", "lm_input_mask_fraction"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1); got {getattr(self, name)!r}.")
        for name in (
            "sigma_data",
            "inference_sigma_max",
            "inference_sigma_min",
            "inference_rho",
            "inference_sigma_cap",
            "step_scale",
            "pae_max_dist",
            "layer_norm_eps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0; got {getattr(self, name)!r}.")
        for name in ("gamma_0", "gamma_min", "noise_scale", "recurrence_poisson_mean"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0; got {getattr(self, name)!r}.")
        if not 0 < self.confidence_min_dist < self.confidence_max_dist:
            raise ValueError("confidence_min_dist must be positive and below confidence_max_dist.")
        if self.trimul_backend not in _VALID_TRIMUL_BACKENDS:
            raise ValueError(
                f"trimul_backend must be one of {_VALID_TRIMUL_BACKENDS}; "
                f"got {self.trimul_backend!r}."
            )
        if self.attention_backend not in _VALID_ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {_VALID_ATTENTION_BACKENDS}; "
                f"got {self.attention_backend!r}."
            )
