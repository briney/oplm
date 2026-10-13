# Boltz-2 loss oracles

Frozen copies of the Boltz-2 loss functions `src/oplm/fold/losses.py` is checked against
(spec §6.2, §9). Source: the `boltz==2.2.1` wheel from PyPI, sha256
`b8c62bbdede1922931d9203118f62c858f11aa699bf91fd4c05a5ed6a6d8b4fc`, upstream
<https://github.com/jwohlwend/boltz> (MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele
Corso, Saro Passaro; full text in `THIRD_PARTY_NOTICES.md`). These modules are test code only.

| module | upstream file | functions |
|---|---|---|
| `boltz2_diffusion.py` | `boltz/model/loss/diffusionv2.py` | `weighted_rigid_align`, `smooth_lddt_loss` |
| `boltz2_distogram.py` | `boltz/model/loss/distogramv2.py` | `distogram_loss` |
| `boltz2_confidence.py` | `boltz/model/loss/confidencev2.py`, `boltz/model/layers/confidence_utils.py` | `resolved_loss`, `get_target_lddt`, `plddt_loss`, `lddt_dist`, `express_coordinate_in_frame`, `get_target_pae`, `pae_loss`, `get_target_pde`, `pde_loss`, `compute_collinear_mask`, `compute_frame_pred` |

Modifications: `einx`/`einops` calls rewritten as `torch.einsum` (no new dependency);
`boltz.data.const.chain_type_ids` inlined as `CHAIN_TYPE_IDS`; `torch.cuda.amp.autocast` replaced by
`torch.autocast("cuda", ...)`; unused wrappers (`confidence_loss`, `tm_function`, `compute_ptms`)
omitted. Everything else is verbatim, including the eps constants and reductions the parity tests
in `tests/fold/test_losses.py` match.

Conventions the oracles encode (and OPLM reproduces): the reference is aligned onto the
prediction; smooth lDDT scores pairs under 15 Å (30 Å for nucleotides) excluding `i == j`;
pLDDT targets are per-atom lDDT against representative atoms within 15 Å; PAE expresses token `j`'s
frame center in token `i`'s (N, CA, C) frame; PDE is the representative-atom distance error;
`resolved` class 0 means resolved.
