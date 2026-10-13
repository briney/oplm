"""Loss terms against the vendored Boltz-2 oracles, rigid-transform recovery, masks, edge cases."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.data.collate import collate_examples
from oplm.fold.data.featurize import featurize_complex
from oplm.fold.diffusion import random_rotations
from oplm.fold.losses import (
    LossWeights,
    diffusion_mse,
    distogram_loss,
    edm_weight,
    express_in_frames,
    pae_loss,
    pair_distances,
    pde_loss,
    plddt_loss,
    resolved_loss,
    smooth_lddt,
    token_frames,
)
from tests.fold.oracles import boltz2_confidence, boltz2_diffusion, boltz2_distogram

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.collate import FoldBatch
    from oplm.fold.data.complex import MolecularComplex

# The bin counts (BINS = 8 for distogram/PAE/PDE, 10 for pLDDT below) are kept small on purpose:
# parity at the production 50/64 bins fails on ~1-2 % of seeds, because the oracle's `cdist`
# matmul path tips distances across bin thresholds.
BINS = 8
MAX_DIST = 32.0


@pytest.fixture(scope="module")
def batch(parsed: Callable[[str], MolecularComplex]) -> FoldBatch:
    """1crn and 1ejg (46 residues each, 1ejg has unresolved atoms) collated at crop 64."""
    return collate_examples(
        [featurize_complex(parsed("1crn")), featurize_complex(parsed("1ejg"))], crop_size=64
    )


def _pred(batch: FoldBatch, noise: float = 0.5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return batch.targets.coords + noise * torch.randn(batch.targets.coords.shape, generator=g)


def _boltz_feats(batch: FoldBatch) -> dict[str, torch.Tensor]:
    f, t = batch.features, batch.targets
    B, L, A = f.token_mask.shape[0], f.num_tokens, f.num_atoms
    atom_to_token = F.one_hot(f.atom_to_token, L).float() * f.atom_mask[..., None].float()
    token_to_rep = torch.zeros(B, L, A)
    token_to_rep[torch.arange(B)[:, None], torch.arange(L)[None], f.distogram_atom_idx] = 1.0
    token_to_rep = token_to_rep * f.token_mask[..., None].float()
    rep_resolved = torch.gather(t.resolved, 1, f.distogram_atom_idx) & f.token_mask
    frame_ok = f.token_mask.clone()
    for k in range(3):
        frame_ok &= torch.gather(t.resolved, 1, t.frame_atom_idx[..., k])
    return {
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep,
        "r_set_to_rep_atom": token_to_rep,
        "token_pad_mask": f.token_mask.float(),
        "atom_pad_mask": f.atom_mask.float(),
        "mol_type": torch.zeros(B, L, dtype=torch.long),
        "asym_id": f.asym_id,
        "frames_idx": t.frame_atom_idx,
        "frame_resolved_mask": frame_ok.float(),
        "token_disto_mask": rep_resolved.float(),
        "atom_resolved_mask": t.resolved.float(),
    }


def test_edm_weight_matches_the_boltz_formula() -> None:
    sigma = torch.tensor([1.0, 16.0, 100.0])
    torch.testing.assert_close(edm_weight(sigma, 16.0), (sigma**2 + 256.0) / (sigma * 16.0) ** 2)


def test_diffusion_mse_matches_boltz(batch: FoldBatch) -> None:
    pred, true, res = _pred(batch), batch.targets.coords, batch.targets.resolved.float()
    ours = diffusion_mse(pred, true, res)
    aligned = boltz2_diffusion.weighted_rigid_align(true, pred, torch.ones_like(res), res)
    err = ((pred - aligned) ** 2).sum(-1)
    expected = (err * res).sum(-1) / ((3 * res).sum(-1) + 1e-5)
    torch.testing.assert_close(ours, expected, atol=1e-4, rtol=1e-4)


def test_diffusion_mse_recovers_a_rigid_transform(batch: FoldBatch) -> None:
    true = batch.targets.coords
    rot = random_rotations(
        2, device=true.device, dtype=true.dtype, generator=torch.Generator().manual_seed(1)
    )
    moved = torch.einsum("bad,bds->bas", true, rot) + torch.tensor([[[3.0, -2.0, 7.0]]])
    loss = diffusion_mse(moved, true, batch.targets.resolved.float())
    assert loss.shape == (2,) and (loss < 1e-6).all()


def test_rigid_align_with_fewer_than_three_points_is_finite(batch: FoldBatch) -> None:
    """Review Focus 3."""
    pred, true = _pred(batch), batch.targets.coords
    for n_points in (0, 1, 2):
        w = torch.zeros_like(batch.targets.resolved.float())
        w[:, :n_points] = 1.0
        loss = diffusion_mse(pred.clone().requires_grad_(True), true, w)
        assert torch.isfinite(loss).all()
        loss.sum().backward()


def test_smooth_lddt_matches_boltz(batch: FoldBatch) -> None:
    pred, true = _pred(batch), batch.targets.coords
    ours = smooth_lddt(pred, true, batch.targets.resolved)
    expected = boltz2_diffusion.smooth_lddt_loss(
        pred, true, torch.zeros(2, pred.shape[1]), coords_mask=batch.targets.resolved.float()
    )
    torch.testing.assert_close(ours.mean(), expected, atol=1e-4, rtol=1e-4)


def test_smooth_lddt_of_a_rigid_transform_is_the_sigmoid_constant(batch: FoldBatch) -> None:
    true = batch.targets.coords
    rot = random_rotations(
        2, device=true.device, dtype=true.dtype, generator=torch.Generator().manual_seed(2)
    )
    moved = torch.einsum("bad,bds->bas", true, rot) + 1.0
    expected = 1 - sum(torch.sigmoid(torch.tensor(t)) for t in (0.5, 1.0, 2.0, 4.0)) / 4
    torch.testing.assert_close(
        smooth_lddt(moved, true, batch.targets.resolved), expected.expand(2), atol=1e-4, rtol=0
    )


def test_distogram_matches_boltz(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    torch.manual_seed(0)
    logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    ours = distogram_loss(
        logits,
        t.coords,
        t.resolved,
        f.token_mask,
        f.distogram_atom_idx,
        min_dist=2.0,
        max_dist=22.0,
    )
    reps = torch.gather(t.coords, 1, f.distogram_atom_idx[..., None].expand(-1, -1, 3))
    bins = torch.bucketize(torch.cdist(reps, reps), torch.linspace(2.0, 22.0, BINS - 1))
    feats = _boltz_feats(batch)
    feats["disto_target"] = F.one_hot(bins, BINS).float()[:, :, :, None, :]
    expected, _ = boltz2_distogram.distogram_loss({"pdistogram": logits[:, :, :, None, :]}, feats)
    torch.testing.assert_close(ours.mean(), expected, atol=1e-5, rtol=1e-5)


def test_distogram_perfect_logits_give_near_zero_loss(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    reps = torch.gather(t.coords, 1, f.distogram_atom_idx[..., None].expand(-1, -1, 3))
    bins = torch.bucketize(pair_distances(reps), torch.linspace(2.0, 22.0, BINS - 1))
    logits = 50.0 * F.one_hot(bins, BINS).float()
    loss = distogram_loss(
        logits,
        t.coords,
        t.resolved,
        f.token_mask,
        f.distogram_atom_idx,
        min_dist=2.0,
        max_dist=22.0,
    )
    assert (loss < 1e-4).all()


def test_token_frames_and_pae_match_boltz(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    pred = _pred(batch, noise=1.0)
    torch.manual_seed(1)
    logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    ours = pae_loss(
        logits, pred, t.coords, t.resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST
    )
    expected, _ = boltz2_confidence.pae_loss(
        logits, pred, _boltz_feats(batch), t.coords, t.resolved.float(), max_dist=MAX_DIST
    )
    torch.testing.assert_close(ours.mean(), expected, atol=1e-5, rtol=1e-5)
    frames, centers, ok = token_frames(t.coords, t.frame_atom_idx, t.resolved, f.token_mask)
    assert frames.shape == (2, f.num_tokens, 3, 3) and ok.dtype == torch.bool
    torch.testing.assert_close(
        torch.einsum("blij,blkj->blik", frames, frames)[ok],
        torch.eye(3).expand(int(ok.sum()), 3, 3),
        atol=1e-4,
        rtol=0,
    )
    assert express_in_frames(frames, centers).shape == (2, f.num_tokens, f.num_tokens, 3)


def test_pae_mask_drops_unresolved_and_collinear_frames(batch: FoldBatch) -> None:
    """Review Focus 2: a frame missing CA, and a collinear frame, drop out of the pair mask."""
    f, t = batch.features, batch.targets
    coords, resolved = t.coords.clone(), t.resolved.clone()
    resolved[0, t.frame_atom_idx[0, 3, 1]] = False  # token 3 of 1crn loses its CA
    n_idx, ca_idx, c_idx = (t.frame_atom_idx[0, 5, k] for k in range(3))
    coords[0, c_idx] = coords[0, ca_idx] + (
        coords[0, ca_idx] - coords[0, n_idx]
    )  # token 5: collinear
    _, _, ok = token_frames(coords, t.frame_atom_idx, resolved, f.token_mask)
    assert not ok[0, 3] and not ok[0, 5] and ok[0, 4]
    pred = _pred(batch, noise=1.0)
    torch.manual_seed(2)
    logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)

    def loss_of(x: torch.Tensor) -> torch.Tensor:
        return pae_loss(
            x, pred, coords, resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST
        )

    base = loss_of(logits)
    # Bump one bin (a constant shift over all bins leaves cross-entropy unchanged).
    rows = logits.clone()
    rows[0, 3, :, 0] += 10.0  # unresolved CA frame row
    rows[0, 5, :, 0] += 10.0  # collinear frame row
    torch.testing.assert_close(loss_of(rows), base)
    column = logits.clone()
    column[0, :, 3, 0] += 10.0  # token 3's CA is unresolved: its column is masked too
    torch.testing.assert_close(loss_of(column), base)
    control = logits.clone()
    control[0, 4, :, 0] += 10.0  # a valid frame row must move the loss
    assert not torch.allclose(loss_of(control), base, atol=1e-3)


def test_pde_and_plddt_and_resolved_match_boltz(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    pred = _pred(batch, noise=1.0)
    feats = _boltz_feats(batch)
    torch.manual_seed(3)
    pde_logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    plddt_logits = torch.randn(2, f.num_atoms, 10)  # 10 bins: see the BINS note
    resolved_logits = torch.randn(2, f.num_atoms, 2)
    ours_pde = pde_loss(
        pde_logits,
        pred,
        t.coords,
        t.resolved,
        f.token_mask,
        f.distogram_atom_idx,
        max_dist=MAX_DIST,
    )
    exp_pde, _ = boltz2_confidence.pde_loss(
        pde_logits, pred, feats, t.coords, t.resolved.float(), max_dist=MAX_DIST
    )
    torch.testing.assert_close(ours_pde.mean(), exp_pde, atol=1e-5, rtol=1e-5)
    ours_plddt = plddt_loss(
        plddt_logits, pred, t.coords, t.resolved, f.atom_mask, f.token_mask, f.distogram_atom_idx
    )
    exp_plddt, _ = boltz2_confidence.plddt_loss(
        plddt_logits, pred, feats, t.coords, t.resolved.float(), token_level_confidence=False
    )
    torch.testing.assert_close(ours_plddt.mean(), exp_plddt, atol=1e-5, rtol=1e-5)
    ours_res = resolved_loss(resolved_logits, t.resolved, f.atom_mask)
    exp_res = boltz2_confidence.resolved_loss(resolved_logits, feats, t.resolved.float())
    torch.testing.assert_close(ours_res.mean(), exp_res, atol=1e-5, rtol=1e-5)


def test_confidence_targets_of_a_rigid_transform_are_bin_zero(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    rot = random_rotations(
        2, device=t.coords.device, dtype=t.coords.dtype, generator=torch.Generator().manual_seed(4)
    )
    moved = torch.einsum("bad,bds->bas", t.coords, rot) - 5.0
    zero_bin = torch.zeros(2, f.num_tokens, f.num_tokens, BINS)
    zero_bin[..., 0] = 50.0
    pae = pae_loss(
        zero_bin, moved, t.coords, t.resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST
    )
    assert (pae < 1e-4).all()
    pde = pde_loss(
        zero_bin, moved, t.coords, t.resolved, f.token_mask, f.distogram_atom_idx, max_dist=MAX_DIST
    )
    assert (pde < 1e-4).all()
    top_bin = torch.zeros(2, f.num_atoms, 10)
    top_bin[..., -1] = 50.0  # lDDT 1.0 -> the last bin
    plddt = plddt_loss(
        top_bin, moved, t.coords, t.resolved, f.atom_mask, f.token_mask, f.distogram_atom_idx
    )
    assert (plddt < 1e-4).all()


def test_rollout_multiplicity_repeats_the_base_batch(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    # Two rollouts per example; rollout j of example b sits at row b·2 + j.
    logits = torch.randn(4, f.num_atoms, 2)
    losses = resolved_loss(logits, t.resolved, f.atom_mask)
    single = torch.cat(
        [
            resolved_loss(logits[2 * b : 2 * b + 2], t.resolved[b : b + 1], f.atom_mask[b : b + 1])
            for b in range(2)
        ]
    )
    torch.testing.assert_close(losses, single)


def test_losses_are_finite_with_no_resolved_atoms(batch: FoldBatch) -> None:
    """Review Focus 1."""
    f, t = batch.features, batch.targets
    none = torch.zeros_like(t.resolved)
    pred = _pred(batch).requires_grad_(True)
    torch.manual_seed(5)
    pair_logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS, requires_grad=True)
    atom_logits = torch.randn(2, f.num_atoms, 10, requires_grad=True)
    res_logits = torch.randn(2, f.num_atoms, 2, requires_grad=True)
    terms = [
        diffusion_mse(pred, t.coords, none.float()),
        smooth_lddt(pred, t.coords, none),
        distogram_loss(
            pair_logits,
            t.coords,
            none,
            f.token_mask,
            f.distogram_atom_idx,
            min_dist=2.0,
            max_dist=22.0,
        ),
        pae_loss(
            pair_logits, pred, t.coords, none, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST
        ),
        pde_loss(
            pair_logits, pred, t.coords, none, f.token_mask, f.distogram_atom_idx, max_dist=MAX_DIST
        ),
        plddt_loss(
            atom_logits, pred, t.coords, none, f.atom_mask, f.token_mask, f.distogram_atom_idx
        ),
        resolved_loss(res_logits, none, torch.zeros_like(f.atom_mask)),
    ]
    total = sum(term.sum() for term in terms)
    assert torch.isfinite(total) and float(total.detach()) == 0.0
    total.backward()
    for leaf in (pred, pair_logits, atom_logits, res_logits):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_loss_weights_defaults() -> None:
    w = LossWeights()
    assert (w.diffusion_mse, w.smooth_lddt, w.distogram, w.confidence, w.pae) == (
        4.0,
        1.0,
        0.03,
        1e-4,
        1.0,
    )
    assert math.isclose(LossWeights(pae=0.0).pae, 0.0)
