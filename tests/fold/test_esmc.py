"""The frozen ESM-C adapter: contract on a fake encoder, the ``esmc:`` prefix, the real model."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from oplm.config import FoldRunConfig
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.esmc import ESMC_PREFIX, ESMC_REPOS, FrozenEsmC, load_esmc, parse_esmc_name
from oplm.fold.lm_shim import frozen_lm_hidden_states, lm_state_count
from oplm.fold.modeling_fold import OplmForFolding, load_frozen_lm
from oplm.fold.training import build_fold_config
from tests.fold.helpers import tiny_fold_config

_HAS_ESM = importlib.util.find_spec("esm") is not None


class _FakeEsmc(nn.Module):
    """Mimics ``EsmcModel``: ``config`` fields and ``(n_layers + 1, B, T, D)`` hidden states."""

    def __init__(self, layers: int = 2, width: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=width, num_hidden_layers=layers, pad_token_id=1, mask_token_id=32
        )
        self.embed = nn.Embedding(64, width)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        x = self.embed(input_ids)
        states = torch.stack([x * (k + 1) for k in range(self.config.num_hidden_layers + 1)])
        return SimpleNamespace(hidden_states=states, last_hidden_state=states[-1])


def test_parse_esmc_name_and_repo_table() -> None:
    assert parse_esmc_name("esmc:esmc_600m") == "esmc_600m"
    assert parse_esmc_name("/some/path") is None and parse_esmc_name("biohub/x#esmc") is None
    assert ESMC_REPOS == {
        "esmc_300m": "biohub/ESMC-300M",
        "esmc_600m": "biohub/ESMC-600M",
        "esmc_6b": "biohub/ESMC-6B",
    }
    assert ESMC_PREFIX == "esmc:"


def test_frozen_wrapper_presents_the_hidden_state_contract() -> None:
    lm = FrozenEsmC(_FakeEsmc(), name="esmc_fake", revision="r0")
    assert lm.num_hidden_states == 3 and lm_state_count(lm) == 3
    assert lm.config.hidden_size == 32 and lm.name == "esmc_fake" and lm.revision == "r0"
    assert all(not p.requires_grad for p in lm.parameters())
    lm.train()
    assert not lm.training and not lm.model.training
    f = featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B")])
    states = frozen_lm_hidden_states(lm, f)
    assert states.shape == (1, 5, 3, 32)
    head = OplmForFolding(tiny_fold_config(lm_hidden_size=32, lm_num_hidden_states=3)).eval()
    head.attach_lm(lm, name_or_path="esmc:esmc_fake")
    with torch.no_grad():
        out = head(f, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out.coords).all()
    # Verify the frozen LM is not in the module tree (only the head's own modules appear)
    assert head._lm is not None
    assert not any(k.startswith("_lm") for k in head.state_dict())


def test_wrapper_rejects_a_state_count_mismatch() -> None:
    class _Short(_FakeEsmc):
        def forward(
            self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True
        ):
            out = super().forward(input_ids, attention_mask)
            out.hidden_states = out.hidden_states[:-1]
            return out

    lm = FrozenEsmC(_Short(), name="x")
    with pytest.raises(RuntimeError, match="hidden states"):
        lm(input_ids=torch.zeros(1, 4, dtype=torch.long))


@pytest.mark.skipif(_HAS_ESM, reason="esm is installed; the ImportError path is unreachable")
def test_load_frozen_lm_esmc_prefix_names_the_extra_without_the_package() -> None:
    with pytest.raises(ImportError, match=r"oplm\[esmc\]"):
        load_frozen_lm("esmc:esmc_600m")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_ESM, reason="esm package not installed (oplm[esmc], Python >= 3.12)")
def test_real_esmc_300m_states_match_its_config() -> None:
    lm = load_esmc("esmc_300m", device="cpu")
    f = featurize([ChainSpec("MKTAYIAKQRQISFVKSHFSRQ", "A")])
    states = frozen_lm_hidden_states(lm, f)
    assert states.shape == (1, 22, lm.config.num_hidden_layers + 1, lm.config.hidden_size)
    assert torch.isfinite(states).all()
    cfg = build_fold_config(FoldRunConfig(lm="esmc:esmc_300m"), lm=lm)
    assert cfg.lm_hidden_size == lm.config.hidden_size
    assert cfg.lm_num_hidden_states == lm.num_hidden_states


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_ESM, reason="esm package not installed (oplm[esmc], Python >= 3.12)")
def test_real_esmc_300m_padded_chain_rows_match_the_unpadded_chain() -> None:
    """Two chains of different lengths share one LM batch (right-padded); padding must not leak."""
    lm = load_esmc("esmc_300m", device="cpu")
    long_seq, short_seq = "MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV"
    both = featurize([ChainSpec(long_seq, "A"), ChainSpec(short_seq, "B")])
    assert both.lm_input_ids.shape[0] == 2 and both.lm_input_ids.shape[1] > len(short_seq) + 2
    padded = frozen_lm_hidden_states(lm, both)
    alone = frozen_lm_hidden_states(lm, featurize([ChainSpec(short_seq, "A")]))
    n = len(short_seq)
    torch.testing.assert_close(padded[0, len(long_seq) :], alone[0, :n], atol=1e-4, rtol=0)
