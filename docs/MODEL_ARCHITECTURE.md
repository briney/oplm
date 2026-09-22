# OPLM Model Architecture

This document is the source of truth for model semantics that are not obvious
from tensor shapes alone; it carries the intended architecture contract.

## Canon Semantics

Canon is controlled by `canon_enabled`, `canon_positions`,
`canon_kernel_sizes`, `canon_activation`, and `canon_residual`. There is no
legacy Canon mode: the paper-exact encoder implementation replaced the earlier
Canon-inspired tensor paths in place, rather than retaining old behavior behind a
compatibility switch.

The intended encoder target keeps the Canon paper's layer insertion semantics
and adapts only the local convolution window for an encoder MLM. Canon is
supported under `norm_strategy` in `{"pre", "sandwich", "post_sdpa"}`: all three
have an outer attention pre-norm and FFN pre-norm, and their additional
post-norms act on the sublayer *outputs* (downstream of every Canon insertion
point), so A and C land identically to plain pre-norm. `norm_strategy="hybrid"`
is rejected at config validation: it suppresses the outer attention pre-norm
(QK/V-norm moves inside attention, which sees raw `x`), so Canon-A has no
normalized stream to act on.

Paper-exact encoder Canon uses:

- `canon_residual=True`;
- `canon_activation="none"` for paper-comparable experiments;
- centered bidirectional same-length depthwise convolution;
- no causal masking inside Canon convolutions.

### Convolution Window

The encoder adaptation uses a bidirectional same-length depthwise convolution
over sequence positions. Pad-token rows are zeroed before convolution so pad
content cannot contribute to real tokens.

The library (bare-config) default kernel size is `4`, the paper's choice; the
production training default (`configs/model/base.yaml`) uses `7`.
For even kernels, exact token-centered symmetry is impossible in a same-length
discrete convolution. OPLM uses the existing half-token alignment:

```text
k = 4: pad left by 2, right by 1
output i sees input positions [i-2, i-1, i, i+1]
```

Use odd kernels for experiments that require exact symmetry around each token.

### Insertion Sites

For `norm_strategy="pre"`, one transformer block with paper-exact encoder Canon
is (sandwich/post_sdpa are identical on the Canon insertion points, adding only
post-norms on the sublayer outputs):

```text
attn_base = AttnNorm(x)
attn_in   = attn_base + CanonA(attn_base)
attn_out  = Attention(attn_in)
h         = x + alpha * attn_out

mlp_base  = FfnNorm(h)
mlp_in    = mlp_base + CanonC(mlp_base)
ffn_out   = FFN_with_CanonD(mlp_in)
y         = h + alpha * ffn_out
```

Position definitions:

- **A** acts on the attention pre-norm stream passed to Q/K/V projection.
- **B** acts inside attention after Q/K/V projection and after OPLM's QK/V
  normalization, before value residual, RoPE, and the attention kernel.
- **C** acts on the FFN pre-norm stream passed into the FFN.
- **D** acts inside the FFN at intermediate width before the activation.

The current target keeps Canon-B after OPLM's QK/V normalization to preserve the
stability assumptions of the existing attention stack:

```text
QKV projection -> QK/V norm -> residual Canon-B -> value residual -> RoPE
```

### Residual Form

Paper-exact encoder Canon uses residual Canon at every enabled position:

```text
z = z + Canon(z, attention_mask)
```

This applies to A, B, C, and D. Replacement-style convolution belongs only to
future explicitly named non-paper research knobs.

### Canon-D Tensor Space

Canon-D belongs inside the FFN before the activation, at `intermediate_size`
channels:

```text
SwiGLU: hidden = silu(gate + CanonD(gate)) * up
GEGLU:  hidden = gelu(gate + CanonD(gate)) * up
relu2:  hidden = relu(up + CanonD(up)) ** 2
```

It must not be applied to the hidden-size FFN input before the FFN projections.

## Shared-layer looping

`num_hidden_layers` is physical depth L. `num_loops` (default 1) selects the total
number of invocations of each block in `[loop_start, loop_end)`, a zero-based,
half-open range whose default is the whole stack. Blocks outside that range run
once. For eight physical layers and the range `[2, 6)`, two loops execute:

- `stack`: `1 2 → 3 4 5 6 → 3 4 5 6 → 7 8` (one-based labels).
- `interleave`: `1 2 → 3 3 4 4 5 5 6 6 → 7 8`.

Each physical block is registered once; all its attention, FFN, norm, gate, and
Canon parameters are reused. Canon kernels retain their physical indices.
Embeddings run once before the schedule, and final normalization/task prediction
runs once after it. Residue positions and padding masks stay the same. Training
uses the existing final-output loss and differentiates through every occurrence.
Dropout samples normally per invocation, with checkpoint recomputation preserving
that invocation's RNG state.

The effective depth is `D = L + (num_loops - 1) * (loop_end - loop_start)` after
resolving a null end to L. Requested hidden states contain D+1 tensors (embedding
plus each pre-final-norm block output); attentions contain D tensors.
`model.oplm.backbone.layer_execution_order` maps occurrences to physical blocks
(`model.backbone` for `OplmModel`). One loop reproduces the ordinary traversal.

Residual alpha and projection initialization continue to scale with physical
L, not D. Converting a checkpoint does not alter its weights or alpha buffers.
When value residuals are enabled, the first execution of physical block zero
provides the global reference for all later physical blocks. Every invocation of
block zero bypasses mixing; revisiting it never replaces or detaches the reference.
Changing loop settings requires constructing/loading a model with the new config.

Training compute estimates use effective depth `D` for the block projections and
count the MLM head once. Unique parameter counts use physical blocks and remain
unchanged by looping. Activation memory grows with executed depth unless reduced
through activation checkpointing; parameter sharing alone does not reduce it.
