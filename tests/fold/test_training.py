"""FoldTask through the real Trainer on CPU: config building, gradients, overfit, resume, EMA."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import FoldRunConfig
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.modeling_fold import OplmForFolding
from oplm.fold.training import FoldTask, build_fold_config, load_initial_head, require_fold
from oplm.training.task import StepResult
from tests.fold.helpers import TINY_FOLD_MODEL, tiny_fold_run_cfg, tiny_lm
from tests.training.conftest import FullRecordingCallback, configure_accelerator_device

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def lm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("lm") / "tiny"
    tiny_lm().save_pretrained(path)
    return path


class _Recording(FoldTask):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[list[str]] = []

    def step(self, model, batch) -> StepResult:
        result = super().step(model, batch)
        self.ids.append(list(batch.ids))
        return result


def test_build_fold_config_infers_lm_shape_and_rejects_bad_keys() -> None:
    run = FoldRunConfig(lm="x", model=dict(TINY_FOLD_MODEL))
    cfg = build_fold_config(run, lm=tiny_lm())
    assert isinstance(cfg, FoldConfig)
    assert cfg.lm_hidden_size == 32 and cfg.lm_num_hidden_states == 3 and cfg.pair_width == 32
    assert cfg.lm_name_or_path == "x" and cfg.lm_max_chain_length == 1024
    with pytest.raises(ValueError, match="unknown FoldConfig key"):
        build_fold_config(FoldRunConfig(model={"pair_widht": 32}), lm=tiny_lm())
    with pytest.raises(ValueError, match="lm_name_or_path"):
        build_fold_config(FoldRunConfig(model={"lm_name_or_path": "y"}), lm=tiny_lm())
    explicit = build_fold_config(FoldRunConfig(model={"lm_hidden_size": 16}))
    assert explicit.lm_hidden_size == 16


def test_require_fold_names_the_missing_block(
    tmp_path: Path, fixture_dataset: Path, lm_dir: Path
) -> None:
    cfg = tiny_fold_run_cfg(tmp_path, fixture_dataset, lm_dir)
    assert require_fold(cfg) is cfg.fold
    cfg.fold = None
    with pytest.raises(ValueError, match="fold:"):
        require_fold(cfg)


def test_step_returns_accounting_and_gradients_stay_in_the_head(
    tmp_path: Path, fixture_dataset: Path, lm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_fold_run_cfg(tmp_path, fixture_dataset, lm_dir)
    task = FoldTask()
    model = task.build_model(cfg, None)
    assert model.training and model.lm is not None and not model.lm.training
    assert model.folding_trunk.gradient_checkpointing
    loader = task.build_dataloader(cfg)
    batch = next(iter(loader))
    result = task.step(model, batch)
    assert isinstance(result, StepResult) and result.samples == 2
    assert result.tokens == int(batch.features.token_mask.sum())
    assert {"diffusion_mse", "distogram", "confidence", "loops"} <= set(result.metrics)
    assert task.last_batch_ids == batch.ids and task.flops_per_token(cfg) is None
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in model.lm.parameters())


def test_dataloader_mixes_sources_with_the_run_seed(
    tmp_path: Path, fixture_dataset: Path, lm_dir: Path
) -> None:
    from oplm.data.sequence.dataset import InterleavedDataset
    from oplm.fold.data.dataset import StructureDataset

    cfg = tiny_fold_run_cfg(tmp_path, fixture_dataset, lm_dir, seed=11)
    cfg.data.train = {
        "a": {"path": str(fixture_dataset), "fraction": 0.5},
        "b": str(fixture_dataset),
    }
    loader = FoldTask().build_dataloader(cfg)
    dataset = loader.dataset
    assert isinstance(dataset, InterleavedDataset)
    sources = dataset._datasets
    assert all(isinstance(s, StructureDataset) and s._seed == 11 for s in sources)
    assert len(dataset) == sum(len(s) for s in sources)


@pytest.mark.slow
def test_trainer_overfits_one_chain(
    tmp_path: Path, fixture_dataset: Path, lm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_fold_run_cfg(
        tmp_path / "run", fixture_dataset, lm_dir, max_steps=40, batch_size=1, lr=2e-3
    )
    cfg.data.train = str(fixture_dataset)
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, task=FoldTask(), callbacks=[callback])
    trainer.train()
    losses = [m["train/loss"] for _, m in callback.train_logs]
    assert len(losses) == 40 and all(torch.isfinite(torch.tensor(losses)).tolist())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    assert "train/distogram" in callback.train_logs[0][1]


@pytest.mark.slow
def test_resume_restores_step_cursor_and_ema(
    tmp_path: Path, fixture_dataset: Path, lm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    full = _Recording()
    Trainer(
        tiny_fold_run_cfg(tmp_path / "full", fixture_dataset, lm_dir, max_steps=6, ema_decay=0.5),
        task=full,
    ).train()
    first = _Recording()
    Trainer(
        tiny_fold_run_cfg(
            tmp_path / "split", fixture_dataset, lm_dir, max_steps=3, save_every=3, ema_decay=0.5
        ),
        task=first,
    ).train()
    ckpt = tmp_path / "split" / "checkpoint-3"
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["ema"]["n_averaged"] == 3 and (ckpt / "hf_ema" / "config.json").is_file()
    assert '"model_type": "oplm_fold"' in (ckpt / "hf" / "config.json").read_text()
    second = _Recording()
    resumed = Trainer(
        tiny_fold_run_cfg(
            tmp_path / "split",
            fixture_dataset,
            lm_dir,
            max_steps=6,
            save_every=3,
            ema_decay=0.5,
            resume_from=str(ckpt),
        ),
        task=second,
    )
    assert resumed.global_step == 3 and int(resumed._ema.n_averaged) == 3
    resumed.train()
    assert first.ids + second.ids == full.ids
    assert int(resumed._ema.n_averaged) == 6


def test_init_from_loads_a_head_and_checks_its_architecture(tmp_path: Path) -> None:
    source = OplmForFolding(
        build_fold_config(FoldRunConfig(model=dict(TINY_FOLD_MODEL)), lm=tiny_lm())
    )
    source.save_pretrained(tmp_path / "hf")
    target = build_fold_config(FoldRunConfig(lm="other", model=dict(TINY_FOLD_MODEL)), lm=tiny_lm())
    loaded = load_initial_head(tmp_path / "hf", target)
    assert loaded.config.lm_name_or_path == "other"
    for k, v in source.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k
    wider = build_fold_config(
        FoldRunConfig(model={**TINY_FOLD_MODEL, "pair_width": 64}), lm=tiny_lm()
    )
    with pytest.raises(ValueError, match="pair_width"):
        load_initial_head(tmp_path / "hf", wider)
