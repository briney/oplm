"""TM-score and lDDT against identity, rigid motion, a hinge, TM-align and Boltz-2 (spec §9)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from oplm.fold.eval.metrics import lddt, tm_d0, tm_score
from tests.fold.oracles.boltz2_confidence import lddt_dist

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def _chain(entry: Callable[[str], MolecularComplex], name: str):
    c = entry(name)
    return c.select_tokens(c.chain_tokens(0)), c.chains[0].sequence


def _random_walk(n: int, seed: int) -> np.ndarray:
    steps = np.random.default_rng(seed).normal(size=(n, 3))
    return np.cumsum(3.8 * steps / np.linalg.norm(steps, axis=1, keepdims=True), axis=0)


def _rigid(x: np.ndarray, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    return x @ q.T + np.array([5.0, -3.0, 12.0])


def test_tm_d0_follows_the_tm_score_program() -> None:
    assert tm_d0(10) == 0.5 and tm_d0(21) == 0.5
    assert tm_d0(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    assert tm_d0(22) == pytest.approx(max(1.24 * 7 ** (1 / 3) - 1.8, 0.5))


def test_identity_and_rigid_motion_score_one() -> None:
    x = _random_walk(60, seed=0)
    moved = _rigid(x, seed=1)
    assert tm_score(x, x) == pytest.approx(1.0, abs=1e-12)
    assert tm_score(moved, x) == pytest.approx(1.0, abs=1e-9)
    assert lddt(x, x) == 1.0 and lddt(moved, x) == pytest.approx(1.0)


def test_iterative_search_beats_one_global_superposition() -> None:
    chain = _random_walk(90, seed=3)
    theta = np.pi / 2
    rz = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    hinged = chain.copy()
    hinged[60:] = (chain[60:] - chain[59]) @ rz.T + chain[59]
    # One RMSD-optimal fit over all 90 rows, scored with the same d0.
    mc, mh = chain.mean(0), hinged.mean(0)
    u, _, vt = np.linalg.svd((hinged - mh).T @ (chain - mc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    fitted = (hinged - mh) @ rot.T + mc
    single = float((1 / (1 + ((fitted - chain) ** 2).sum(1) / tm_d0(90) ** 2)).sum()) / 90
    ours = tm_score(hinged, chain)
    assert ours > single + 0.2  # 0.70 against 0.32 when planned
    assert ours > 60 / 90 - 0.05


def test_mask_drops_rows_and_sets_the_normalization_length() -> None:
    x = _random_walk(50, seed=4)
    pred = x + np.random.default_rng(5).normal(scale=1.0, size=x.shape)
    pred[30:] += 40.0  # unresolved in the reference: must not count at all
    mask = np.arange(50) < 30
    assert tm_score(pred, x, mask) == pytest.approx(tm_score(pred[:30], x[:30]))
    assert lddt(pred, x, mask) == pytest.approx(lddt(pred[:30], x[:30]))


def test_tm_score_matches_tm_align_on_identity_alignments(parsed) -> None:
    tmtools = pytest.importorskip("tmtools")
    for name in ("1crn", "2ds7", "4hhb"):
        sub, seq = _chain(parsed, name)
        ca = sub.token_center_atom
        ref = sub.atom_coords[ca].astype(np.float64)
        keep = sub.atom_resolved[ca]
        kept_seq = "".join(s for s, k in zip(seq, keep, strict=True) if k)
        for i, sigma in enumerate((0.3, 0.8, 1.5)):
            pred = ref + np.random.default_rng(i).normal(scale=sigma, size=ref.shape)
            result = tmtools.tm_align(pred[keep], ref[keep], kept_seq, kept_seq)
            assert "-" not in result.seqxA and "-" not in result.seqyA  # identity alignment
            assert tm_score(pred, ref, keep) == pytest.approx(result.tm_norm_chain2, abs=1e-6)


def test_lddt_matches_boltz2_lddt_dist(parsed) -> None:
    for name in ("1crn", "2ds7"):
        sub, _ = _chain(parsed, name)
        ref = torch.tensor(sub.atom_coords, dtype=torch.float32)
        keep = torch.tensor(sub.atom_resolved)
        res = torch.tensor(sub.atom_to_token)
        noise = torch.randn(ref.shape, generator=torch.Generator().manual_seed(1))
        pred = ref + 0.7 * noise
        pair = keep[:, None] & keep[None, :] & (res[:, None] != res[None, :])
        oracle, _ = lddt_dist(torch.cdist(pred, pred), torch.cdist(ref, ref), pair.float())
        assert lddt(pred, ref, keep, residue_index=res) == pytest.approx(float(oracle), abs=1e-6)
        assert lddt(pred, ref, keep, residue_index=res, chunk=37) == pytest.approx(
            float(oracle), abs=1e-6
        )
        ca = torch.as_tensor(sub.token_center_atom, dtype=torch.long)
        pair_ca = keep[ca][:, None] & keep[ca][None, :] & ~torch.eye(len(ca), dtype=torch.bool)
        oracle_ca, _ = lddt_dist(
            torch.cdist(pred[ca], pred[ca]), torch.cdist(ref[ca], ref[ca]), pair_ca.float()
        )
        assert lddt(pred[ca], ref[ca], keep[ca]) == pytest.approx(float(oracle_ca), abs=1e-6)


def test_invalid_inputs_raise() -> None:
    x = _random_walk(10, seed=6)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        tm_score(x, x[:9])
    with pytest.raises(ValueError, match="at least 3"):
        tm_score(x, x, np.arange(10) < 2)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        lddt(x[:, :2], x[:, :2])
    far = np.array([[0.0, 0, 0], [100.0, 0, 0], [0, 100.0, 0]])
    with pytest.raises(ValueError, match="no reference pairs"):
        lddt(far, far)
