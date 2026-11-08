"""GREMLIN pseudolikelihood fit and the Potts contact map."""

from __future__ import annotations

import torch

from oplm.fold.evalsets.gremlin import fit_gremlin, full_length_map, potts_contact_map
from oplm.fold.evalsets.msa import neff

A = 21
PLANTED = {(1, 8), (3, 10)}


def _gibbs_msa(length: int = 12, n: int = 2000, sweeps: int = 50) -> torch.Tensor:
    """``n`` sequences Gibbs-sampled from a Potts model coupled only on ``PLANTED`` pairs."""
    g = torch.Generator().manual_seed(0)
    w = torch.zeros(length, A, length, A)
    for i, j in PLANTED:
        w[i, :, j, :] = 2.0 * torch.randn(A, A, generator=g)
        w[j, :, i, :] = w[i, :, j, :].T
    seqs = torch.randint(0, A, (n, length), generator=g)
    for _ in range(sweeps):
        for i in range(length):
            onehot = torch.nn.functional.one_hot(seqs, A).float()
            onehot[:, i] = 0
            logits = torch.einsum("njb,jba->na", onehot, w[:, :, i, :])
            seqs[:, i] = torch.multinomial(logits.softmax(-1), 1, generator=g).squeeze(1)
    return seqs.to(torch.int8)


def test_recovers_planted_couplings():
    msa = _gibbs_msa()
    contacts = potts_contact_map(fit_gremlin(msa, neff(msa)[1]))
    length = contacts.shape[0]
    pairs = [(i, j) for i in range(length) for j in range(i + 3, length)]
    top2 = set(sorted(pairs, key=lambda p: -contacts[p].item())[:2])
    assert top2 == PLANTED


def test_query_only_msa_gives_finite_map():
    query_only = torch.randint(0, 20, (1, 10), generator=torch.Generator().manual_seed(1))
    w = fit_gremlin(query_only.to(torch.int8), torch.ones(1))
    assert w.shape == (10, A, 10, A) and w.dtype == torch.float32
    assert torch.isfinite(potts_contact_map(w)).all()


def test_contact_map_symmetric_zero_diagonal_deterministic():
    msa = torch.randint(0, A, (50, 9), generator=torch.Generator().manual_seed(2)).to(torch.int8)
    weights = neff(msa)[1]
    w = fit_gremlin(msa, weights)
    assert torch.equal(w, fit_gremlin(msa, weights))
    contacts = potts_contact_map(w)
    assert contacts.shape == (9, 9)
    assert torch.allclose(contacts, contacts.T)
    assert (contacts.diagonal() == 0).all()


def test_full_length_map_nans_dropped_columns():
    m = full_length_map(torch.ones(2, 2), torch.tensor([0, 2]), 3)
    assert m.shape == (3, 3)
    assert torch.isnan(m[1]).all() and torch.isnan(m[:, 1]).all()
    assert m[0, 2] == 1 and m[2, 2] == 1
