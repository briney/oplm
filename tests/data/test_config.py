"""Tests for the data-config parsing helpers and ``DataConfig`` validation (Phase 1).

Covers :func:`oplm.data.config.parse_train_configs`,
:func:`oplm.data.config.parse_eval_configs`, and the masking-split validation
added to :class:`oplm.config.DataConfig`.
"""

from __future__ import annotations

import pytest

from oplm.config import DataConfig
from oplm.data.config import parse_eval_configs, parse_train_configs

# --------------------------------------------------------------------------- #
# parse_train_configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [None, "", "   ", {}])
def test_parse_train_empty_inputs(raw: object) -> None:
    """``None``/empty string/empty mapping all yield no training entries."""
    assert parse_train_configs(raw) == []


def test_parse_train_single_path_string() -> None:
    """A single path string expands to one full-weight entry named ``train``."""
    entries = parse_train_configs("/data/uniref50/")
    assert len(entries) == 1
    only = entries[0]
    assert only.name == "train"
    assert only.path == "/data/uniref50/"
    assert only.fraction == pytest.approx(1.0)


def test_parse_train_single_mapping_entry_is_full_weight() -> None:
    """A single mapping entry normalizes to fraction 1.0 even if unspecified."""
    entries = parse_train_configs({"uniref50": {"path": "/data/uniref50/"}})
    assert len(entries) == 1
    assert entries[0].name == "uniref50"
    assert entries[0].path == "/data/uniref50/"
    assert entries[0].fraction == pytest.approx(1.0)


def test_parse_train_fractions_normalize_to_one() -> None:
    """Specified fractions are renormalized so the total is 1.0."""
    entries = parse_train_configs(
        {
            "uniref50": {"path": "/data/uniref50/", "fraction": 0.8},
            "bfd": {"path": "/data/bfd/", "fraction": 0.4},
        }
    )
    fractions = {e.name: e.fraction for e in entries}
    assert sum(fractions.values()) == pytest.approx(1.0)
    # The 2:1 input ratio is preserved after normalization.
    assert fractions["uniref50"] == pytest.approx(2 / 3)
    assert fractions["bfd"] == pytest.approx(1 / 3)


def test_parse_train_omitted_fractions_split_remainder() -> None:
    """Entries lacking ``fraction`` split the remaining mass equally."""
    entries = parse_train_configs(
        {
            "a": {"path": "/a", "fraction": 0.6},
            "b": {"path": "/b"},
            "c": {"path": "/c"},
        }
    )
    fractions = {e.name: e.fraction for e in entries}
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["a"] == pytest.approx(0.6)
    assert fractions["b"] == pytest.approx(0.2)
    assert fractions["c"] == pytest.approx(0.2)


def test_parse_train_all_omitted_is_uniform() -> None:
    """When no fractions are given, mass is split evenly across all entries."""
    entries = parse_train_configs({"a": {"path": "/a"}, "b": {"path": "/b"}})
    for e in entries:
        assert e.fraction == pytest.approx(0.5)


def test_parse_train_bare_string_value_shorthand() -> None:
    """A bare-string mapping value is treated as the dataset path."""
    entries = parse_train_configs({"a": "/a", "b": "/b"})
    assert {e.name: e.path for e in entries} == {"a": "/a", "b": "/b"}
    assert all(e.fraction == pytest.approx(0.5) for e in entries)


def test_parse_train_preserves_order() -> None:
    """Entry order follows the mapping's insertion order."""
    entries = parse_train_configs(
        {"first": {"path": "/1"}, "second": {"path": "/2"}, "third": {"path": "/3"}}
    )
    assert [e.name for e in entries] == ["first", "second", "third"]


def test_parse_train_missing_path_raises() -> None:
    """An entry mapping without ``path`` is an error."""
    with pytest.raises(ValueError, match="missing required 'path'"):
        parse_train_configs({"a": {"fraction": 0.5}})


def test_parse_train_negative_fraction_raises() -> None:
    """A negative fraction is rejected."""
    with pytest.raises(ValueError, match="must be >= 0"):
        parse_train_configs(
            {"a": {"path": "/a", "fraction": -0.1}, "b": {"path": "/b", "fraction": 0.5}}
        )


def test_parse_train_zero_total_raises() -> None:
    """Fractions that sum to zero cannot be normalized."""
    with pytest.raises(ValueError, match="must sum to > 0"):
        parse_train_configs({"a": {"path": "/a", "fraction": 0.0}})


@pytest.mark.parametrize("raw", [42, 3.14, ["/a", "/b"]])
def test_parse_train_invalid_type_raises(raw: object) -> None:
    """A train config that is neither a string nor a mapping is rejected."""
    with pytest.raises(ValueError, match="must be a path string or a"):
        parse_train_configs(raw)


# --------------------------------------------------------------------------- #
# parse_eval_configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [None, {}])
def test_parse_eval_empty_inputs(raw: object) -> None:
    """``None`` and an empty mapping yield no eval entries."""
    assert parse_eval_configs(raw, default_eval_every=1000) == []


def test_parse_eval_minimal_entry_uses_default_eval_every() -> None:
    """A minimal entry resolves ``eval_every`` to the supplied default."""
    entries = parse_eval_configs(
        {"heldout": {"path": "/data/heldout/", "type": "sequence"}},
        default_eval_every=1000,
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "heldout"
    assert entry.path == "/data/heldout/"
    assert entry.type == "sequence"
    assert entry.eval_every == 1000
    assert entry.metrics is None
    assert entry.extra == {}


def test_parse_eval_per_entry_eval_every_override() -> None:
    """A per-entry ``eval_every`` overrides the default cadence."""
    entries = parse_eval_configs(
        {"pg": {"path": "/pg", "type": "proteingym", "eval_every": 50_000}},
        default_eval_every=1000,
    )
    assert entries[0].eval_every == 50_000


def test_parse_eval_metrics_parsed() -> None:
    """A ``metrics`` list is carried through as a list of strings."""
    entries = parse_eval_configs(
        {"h": {"path": "/h", "type": "sequence", "metrics": ["perplexity", "accuracy"]}},
        default_eval_every=1000,
    )
    assert entries[0].metrics == ["perplexity", "accuracy"]


def test_parse_eval_extras_routed_into_extra() -> None:
    """Unknown top-level keys are folded into ``extra``."""
    entries = parse_eval_configs(
        {
            "casp": {
                "path": "/data/casp/",
                "type": "structure",
                "contact_threshold": 8.0,
                "min_seq_sep": 6,
            }
        },
        default_eval_every=1000,
    )
    assert entries[0].extra == {"contact_threshold": 8.0, "min_seq_sep": 6}


def test_parse_eval_missing_path_raises() -> None:
    """``path`` is required."""
    with pytest.raises(ValueError, match="missing required 'path'"):
        parse_eval_configs({"x": {"type": "sequence"}}, default_eval_every=1000)


def test_parse_eval_missing_type_raises() -> None:
    """``type`` is required."""
    with pytest.raises(ValueError, match="missing required 'type'"):
        parse_eval_configs({"x": {"path": "/x"}}, default_eval_every=1000)


def test_parse_eval_nested_extra_block_raises() -> None:
    """A nested ``extra`` block is rejected (keys go directly on the entry)."""
    with pytest.raises(ValueError, match="nested 'extra' block"):
        parse_eval_configs(
            {"x": {"path": "/x", "type": "sequence", "extra": {"a": 1}}},
            default_eval_every=1000,
        )


@pytest.mark.parametrize("raw", ["/just/a/path", 7, ["a"]])
def test_parse_eval_invalid_type_raises(raw: object) -> None:
    """A non-mapping eval config is rejected."""
    with pytest.raises(ValueError, match="must be a"):
        parse_eval_configs(raw, default_eval_every=1000)


def test_parse_eval_entry_value_not_mapping_raises() -> None:
    """An eval entry whose value is not a mapping is rejected."""
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_eval_configs({"x": "/x"}, default_eval_every=1000)


# --------------------------------------------------------------------------- #
# DataConfig masking-split validation
# --------------------------------------------------------------------------- #


def test_dataconfig_defaults_are_valid() -> None:
    """The default masking split passes validation and matches the spec."""
    cfg = DataConfig()
    assert cfg.mask_prob == pytest.approx(0.15)
    assert cfg.mask_token_prob == pytest.approx(0.8)
    assert cfg.random_token_prob == pytest.approx(0.1)
    assert cfg.weighted_masking is False


def test_dataconfig_split_sum_one_is_valid() -> None:
    """A split summing exactly to 1.0 (no keep-original mass) is allowed."""
    cfg = DataConfig(mask_token_prob=0.8, random_token_prob=0.2)
    assert cfg.mask_token_prob + cfg.random_token_prob == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_dataconfig_mask_token_prob_out_of_range_raises(value: float) -> None:
    with pytest.raises(ValueError, match="mask_token_prob must be in"):
        DataConfig(mask_token_prob=value)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_dataconfig_random_token_prob_out_of_range_raises(value: float) -> None:
    with pytest.raises(ValueError, match="random_token_prob must be in"):
        DataConfig(random_token_prob=value)


def test_dataconfig_split_sum_above_one_raises() -> None:
    with pytest.raises(ValueError, match=r"must be <= 1"):
        DataConfig(mask_token_prob=0.7, random_token_prob=0.5)
