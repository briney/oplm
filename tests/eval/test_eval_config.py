"""Tests for eval config parsing."""

from __future__ import annotations

import pytest

from oplm.config import parse_eval_configs


class TestParseEvalConfigs:
    """Test parse_eval_configs() with various input forms."""

    def test_none_returns_empty(self) -> None:
        assert parse_eval_configs(None, default_eval_every=1000) == []

    def test_empty_dict_returns_empty(self) -> None:
        assert parse_eval_configs({}, default_eval_every=1000) == []

    def test_single_dataset(self) -> None:
        raw = {"heldout": {"path": "/data/heldout", "type": "sequence"}}
        entries = parse_eval_configs(raw, default_eval_every=5000)

        assert len(entries) == 1
        assert entries[0].name == "heldout"
        assert entries[0].path == "/data/heldout"
        assert entries[0].type == "sequence"
        assert entries[0].eval_every == 5000
        assert entries[0].metrics is None

    def test_multiple_datasets(self) -> None:
        raw = {
            "heldout": {"path": "/data/heldout", "type": "sequence"},
            "structures": {"path": "/data/pdb", "type": "structure", "eval_every": 10000},
        }
        entries = parse_eval_configs(raw, default_eval_every=5000)

        assert len(entries) == 2
        by_name = {e.name: e for e in entries}
        assert by_name["heldout"].eval_every == 5000
        assert by_name["structures"].eval_every == 10000
        assert by_name["structures"].type == "structure"

    def test_custom_metrics(self) -> None:
        raw = {
            "pg": {"path": "/data/pg", "type": "proteingym", "metrics": ["spearman", "ndcg"]},
        }
        entries = parse_eval_configs(raw, default_eval_every=1000)
        assert entries[0].metrics == ["spearman", "ndcg"]

    def test_none_entry_skipped(self) -> None:
        raw = {"heldout": {"path": "/data/h", "type": "sequence"}, "disabled": None}
        entries = parse_eval_configs(raw, default_eval_every=1000)
        assert len(entries) == 1

    def test_missing_path_raises(self) -> None:
        raw = {"bad": {"type": "sequence"}}
        with pytest.raises(ValueError, match="missing required 'path'"):
            parse_eval_configs(raw, default_eval_every=1000)

    def test_missing_type_raises(self) -> None:
        raw = {"bad": {"path": "/data"}}
        with pytest.raises(ValueError, match="missing required 'type'"):
            parse_eval_configs(raw, default_eval_every=1000)

    def test_non_dict_entry_raises(self) -> None:
        raw = {"bad": "/just/a/string"}
        with pytest.raises(ValueError, match="expected dict"):
            parse_eval_configs(raw, default_eval_every=1000)

    def test_non_dict_raw_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid data.eval config type"):
            parse_eval_configs("/just/a/string", default_eval_every=1000)

    def test_nested_extra_raises_with_guidance(self) -> None:
        raw = {
            "structures": {
                "path": "/data/structures",
                "type": "structure",
                "extra": {"contact_threshold": 8.0},
            },
        }
        with pytest.raises(ValueError, match="deprecated nested 'extra' config"):
            parse_eval_configs(raw, default_eval_every=1000)

    def test_extra_keys_preserved(self) -> None:
        raw = {
            "pdb": {
                "path": "/data/structures",
                "type": "structure",
                "contact_threshold": 8.0,
                "l_divisor": 2,
                "use_cbeta": True,
                "use_categorical_jacobian": True,
                "categorical_jacobian_sample_size": 12,
            },
        }
        entries = parse_eval_configs(raw, default_eval_every=1000)
        assert entries[0].extra == {
            "contact_threshold": 8.0,
            "l_divisor": 2,
            "use_cbeta": True,
            "use_categorical_jacobian": True,
            "categorical_jacobian_sample_size": 12,
        }

    def test_empty_extra_by_default(self) -> None:
        raw = {"heldout": {"path": "/data/heldout", "type": "sequence"}}
        entries = parse_eval_configs(raw, default_eval_every=1000)
        assert entries[0].extra == {}

    def test_extra_excludes_known_keys(self) -> None:
        raw = {
            "pdb": {
                "path": "/data/structures",
                "type": "structure",
                "eval_every": 5000,
                "metrics": ["precision_at_L"],
                "logreg_c": 0.15,
            },
        }
        entries = parse_eval_configs(raw, default_eval_every=1000)
        assert "path" not in entries[0].extra
        assert "type" not in entries[0].extra
        assert "eval_every" not in entries[0].extra
        assert "metrics" not in entries[0].extra
        assert entries[0].extra == {"logreg_c": 0.15}
