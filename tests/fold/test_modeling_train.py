"""The training branch of OplmForFolding.forward: losses, gradients, microbatching, masking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.fold.data.collate import collate_examples
from oplm.fold.data.featurize import featurize_complex
from oplm.fold.losses import LossWeights
from oplm.fold.modeling_fold import (
    FoldOutput,
    FoldTrainOutput,
    OplmForFolding,
    TrainSettings,
    draw_num_loops,
)
from tests.fold.helpers import tiny_fold_config, tiny_lm

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.collate import FoldBatch
    from oplm.fold.data.complex import MolecularComplex

_TINY: dict[str, Any] = dict(
    plddt_bins=10,
    pae_bins=8,
    pde_bins=8,
    confidence_dist_bins=5,
    distogram_bins=8,
    inference_num_steps=2,
    inference_num_loops=2,
    lm_hidden_size=32,
    lm_num_hidden_states=3,
)
_SETTINGS = TrainSettings(diffusion_samples=3, diffusion_microbatch=2, confidence_rollout_steps=2)


@pytest.fixture(scope="module")
def batch(parsed: Callable[[str], MolecularComplex]) -> FoldBatch:
    return collate_examples(
        [featurize_complex(parsed("1crn")), featurize_complex(parsed("1ejg"))], crop_size=64
    )


def _model(**over: Any) -> OplmForFolding:
    torch.manual_seed(0)
    model = OplmForFolding(tiny_fold_config(**{**_TINY, **over}))
    model.attach_lm(tiny_lm())
    return model.train()


def test_training_forward_returns_loss_metrics_and_counts(batch: FoldBatch) -> None:
    model = _model()
    out = model(batch.features, targets=batch.targets, train=_SETTINGS)
    assert isinstance(out, FoldTrainOutput)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and out.loss.requires_grad
    assert out.tokens == int(batch.features.token_mask.sum()) and out.chains == 2
    assert set(out.metrics) == {
        "diffusion_mse",
        "smooth_lddt",
        "distogram",
        "confidence",
        "plddt",
        "pde",
        "pae",
        "resolved",
        "loops",
        "sigma_mean",
    }
    assert 1 <= out.metrics["loops"] <= 6 and all(v == v for v in out.metrics.values())


def test_gradients_reach_every_head_parameter_and_none_of_the_lm(batch: FoldBatch) -> None:
    model = _model()
    out = model(batch.features, targets=batch.targets, train=_SETTINGS)
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.lm is not None and all(p.grad is None for p in model.lm.parameters())
    nonzero = {n for n, p in model.named_parameters() if p.grad.abs().sum() > 0}
    prefixes = (
        "language_model.",
        "folding_trunk.",
        "distogram_head.",
        "structure_head.",
        "confidence_head.pae_head",
    )
    for prefix in prefixes:
        assert any(n.startswith(prefix) for n in nonzero), prefix


def test_microbatch_does_not_change_the_loss(batch: FoldBatch) -> None:
    """Review Focus 4: chunks of 3, 3, 2 give the same loss as one chunk of 8."""
    model = _model()
    whole = TrainSettings(
        diffusion_samples=8, diffusion_microbatch=None, confidence_rollout_steps=2
    )
    chunked = TrainSettings(diffusion_samples=8, diffusion_microbatch=3, confidence_rollout_steps=2)
    torch.manual_seed(7)
    a = model(batch.features, targets=batch.targets, train=whole, num_loops=1)
    torch.manual_seed(7)
    b = model(batch.features, targets=batch.targets, train=chunked, num_loops=1)
    torch.testing.assert_close(a.loss, b.loss, atol=1e-5, rtol=1e-5)
    a.loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    b.loss.backward()
    for g, p in zip(grads, model.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad, atol=1e-5, rtol=1e-4)


def test_weights_scale_their_terms(batch: FoldBatch) -> None:
    model = _model()
    torch.manual_seed(1)
    base = model(batch.features, targets=batch.targets, train=_SETTINGS, num_loops=1)
    torch.manual_seed(1)
    only_disto = model(
        batch.features,
        targets=batch.targets,
        train=TrainSettings(
            3, 2, 2, LossWeights(diffusion_mse=0, smooth_lddt=0, distogram=1.0, confidence=0)
        ),
        num_loops=1,
    )
    torch.testing.assert_close(
        only_disto.loss, torch.tensor(base.metrics["distogram"]), atol=1e-5, rtol=1e-5
    )


def test_training_forward_with_no_resolved_atoms_is_finite(batch: FoldBatch) -> None:
    """Review Focus 1."""
    model = _model()
    targets = batch.targets
    targets = type(targets)(
        coords=targets.coords,
        resolved=torch.zeros_like(targets.resolved),
        bfactor=targets.bfactor,
        frame_atom_idx=targets.frame_atom_idx,
    )
    out = model(batch.features, targets=targets, train=_SETTINGS)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_eval_forward_ignores_train_settings(batch: FoldBatch) -> None:
    """Review Focus 5: no targets -> the inference path, whatever ``train`` says."""
    model = _model().eval()
    with torch.no_grad():
        out = model(batch.features, train=_SETTINGS, generator=torch.Generator().manual_seed(0))
        again = model(batch.features, generator=torch.Generator().manual_seed(0))
    assert isinstance(out, FoldOutput)
    torch.testing.assert_close(out.coords, again.coords)


def test_num_loops_is_poisson_clamped() -> None:
    torch.manual_seed(0)
    draws = [draw_num_loops(3.0, 1, 6) for _ in range(500)]
    assert min(draws) >= 1 and max(draws) <= 6 and 2.0 < sum(draws) / len(draws) < 4.0
    assert draw_num_loops(0.0, 1, 6) == 1


def test_train_mode_draws_loops_and_eval_uses_the_config(batch: FoldBatch) -> None:
    model = _model(recurrence_min_loops=2, recurrence_max_loops=2)
    out = model(batch.features, targets=batch.targets, train=_SETTINGS)
    assert out.metrics["loops"] == 2.0
    model.eval()
    with torch.no_grad():
        inference = model(batch.features, return_intermediates=True)
    assert len(inference.intermediates["states"]) == model.config.inference_num_loops
