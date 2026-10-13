"""The vendored Boltz-2 oracles import and reproduce their closed-form values on trivial inputs."""

from __future__ import annotations

import torch

from tests.fold.oracles import boltz2_confidence, boltz2_diffusion, boltz2_distogram


def test_weighted_rigid_align_is_the_identity_on_matching_clouds() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 7, 3)
    aligned = boltz2_diffusion.weighted_rigid_align(x, x, torch.ones(2, 7), torch.ones(2, 7))
    torch.testing.assert_close(aligned, x, atol=1e-5, rtol=0)


def test_smooth_lddt_of_identical_coordinates_is_the_sigmoid_constant() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 9, 3) * 3
    loss = boltz2_diffusion.smooth_lddt_loss(x, x, torch.zeros(1, 9), torch.ones(1, 9))
    expected = 1 - sum(torch.sigmoid(torch.tensor(t)) for t in (0.5, 1.0, 2.0, 4.0)) / 4
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=0)


def test_lddt_dist_of_identical_matrices_is_one() -> None:
    d = torch.rand(1, 5, 5) * 10
    mask = torch.ones(1, 5, 5) - torch.eye(5)[None]
    score, _ = boltz2_confidence.lddt_dist(d, d, mask, cutoff=15.0, per_atom=True)
    torch.testing.assert_close(score, torch.ones(1, 5), atol=1e-6, rtol=0)


def test_distogram_oracle_runs_on_a_one_hot_target() -> None:
    logits = torch.zeros(1, 3, 3, 1, 4)
    target = torch.nn.functional.one_hot(torch.zeros(1, 3, 3, 1, dtype=torch.long), 4).float()
    feats = {"disto_target": target, "token_disto_mask": torch.ones(1, 3)}
    loss, per_example = boltz2_distogram.distogram_loss({"pdistogram": logits}, feats)
    torch.testing.assert_close(loss, torch.log(torch.tensor(4.0)), atol=1e-5, rtol=0)
    assert per_example.shape == (1,)
    assert boltz2_confidence.CHAIN_TYPE_IDS["PROTEIN"] == 0
