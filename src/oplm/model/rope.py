"""RoPE / partial RoPE applied to Q and K post-QK-norm."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding with optional partial-RoPE / NoPE split.

    For each head and token position `p`, pair the first `rope_dim` channels
    into `rope_dim / 2` 2-D pairs and rotate each pair by an angle `p · θ_i`,
    where `θ_i = base^(-2i / rope_dim)`. The trailing `nope_dim = head_dim -
    rope_dim` channels of each head are position-invariant and pass through
    untouched.

    `cos`/`sin` are precomputed buffers up to `max_position_embeddings`. If a
    longer sequence arrives at inference, the buffers are extended on the fly
    (no parameter update — `inv_freq`, `cos`, and `sin` are all non-persistent
    buffers).

    Rotations are computed in fp32 then cast back to the input dtype.
    """

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        rope_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if rope_dim < 0 or rope_dim > head_dim:
            raise ValueError(
                f"rope_dim must satisfy 0 <= rope_dim <= head_dim; got rope_dim={rope_dim}, "
                f"head_dim={head_dim}."
            )
        if rope_dim % 2 != 0:
            raise ValueError(
                f"rope_dim must be even (each rotation consumes a pair); got {rope_dim}."
            )
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)

        if rope_dim == 0:
            # NoPE-only path: still register zero-size buffers so the module's
            # state dict shape is stable across configurations.
            self.register_buffer("inv_freq", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer(
                "cos_cached",
                torch.empty(max_position_embeddings, 0, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "sin_cached",
                torch.empty(max_position_embeddings, 0, dtype=torch.float32),
                persistent=False,
            )
            # No rotation -> nothing to (re)build.
            self._cache_initialized = True
            return

        inv_freq = self._compute_inv_freq(device=None)  # (rope_dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._compute_cos_sin(max_position_embeddings, device=inv_freq.device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        # Under HF's meta-device fast init these buffers are computed on `meta`
        # and re-materialized later as uninitialized memory (not in the
        # checkpoint, since they're non-persistent). Flag that so the first real
        # forward rebuilds them; a normal CPU/GPU construction is already valid.
        self._cache_initialized = not self.cos_cached.is_meta

    def _compute_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        """Derive `inv_freq` from `base`/`rope_dim` (stored as plain attributes).

        Recomputed rather than read from the buffer so a meta-device-corrupted
        buffer can never leak into the rotation tables.
        """
        return self.base ** (
            -torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device) / self.rope_dim
        )

    def _compute_cos_sin(
        self, seq_len: int, device: torch.device | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fp32 cos/sin tables of shape `(seq_len, rope_dim/2)`."""
        inv_freq = self._compute_inv_freq(device=device)
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # (T,)
        freqs = positions[:, None] * inv_freq[None, :]  # (T, rope_dim/2)
        return freqs.cos(), freqs.sin()

    def _maybe_extend_cache(self, seq_len: int, device: torch.device) -> None:
        """Rebuild the cos/sin tables if `seq_len` exceeds the cached length.

        Also rebuilds when the cached buffers live on a different device than
        the incoming tensors (e.g., first call after `.to(cuda)`) or when the
        cache has not yet been built on a real device (post meta-init load).
        """
        if self.rope_dim == 0:
            return
        cached_len = self.cos_cached.shape[0]
        same_device = self.cos_cached.device == device
        if self._cache_initialized and seq_len <= cached_len and same_device:
            return
        new_len = max(seq_len, cached_len)
        cos, sin = self._compute_cos_sin(new_len, device=device)
        self.inv_freq = self._compute_inv_freq(device=device)
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_position_embeddings = new_len
        self._cache_initialized = True

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the first `rope_dim` channels of `q` and `k` in fp32.

        Args:
            q: `(B, H, T, head_dim)` query tensor in any floating dtype.
            k: `(B, H, T, head_dim)` key tensor in any floating dtype.

        Returns:
            Rotated `(q, k)` with the same shape and dtype as the inputs. When
            `rope_dim == 0`, inputs are returned unchanged.
        """
        if self.rope_dim == 0:
            return q, k

        seq_len = q.shape[-2]
        self._maybe_extend_cache(seq_len, device=q.device)

        cos = self.cos_cached[:seq_len]  # (T, rope_dim/2)
        sin = self.sin_cached[:seq_len]
        # Broadcast against (B, H, T, rope_dim/2).
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply paired-channel rotation to the first `rope_dim` channels.

        `x: (B, H, T, head_dim)`, `cos`/`sin: (1, 1, T, rope_dim/2)`.
        """
        input_dtype = x.dtype
        if self.nope_dim == 0:
            x_rope, x_pass = x, None
        else:
            x_rope = x[..., : self.rope_dim]
            x_pass = x[..., self.rope_dim :]

        x_rope_fp32 = x_rope.to(torch.float32)
        x_even = x_rope_fp32[..., 0::2]  # (B, H, T, rope_dim/2)
        x_odd = x_rope_fp32[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # Re-interleave: (..., rope_dim/2, 2) -> (..., rope_dim).
        rotated = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        rotated = rotated.to(input_dtype)

        if x_pass is None:
            return rotated
        return torch.cat([rotated, x_pass], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, rope_dim={self.rope_dim}, "
            f"nope_dim={self.nope_dim}, max_position_embeddings={self.max_position_embeddings}, "
            f"base={self.base}"
        )
