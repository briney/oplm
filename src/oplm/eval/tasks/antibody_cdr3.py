"""Antibody CDR3 eval: per-position masked-LM metrics over CDR3 residues, by chain type."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, ClassVar

import pyarrow.parquet as pq
import torch

from oplm.data import get_tokenizer
from oplm.data.tokenizer import mask_token_id
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.model import OplmForMaskedLM

logger = logging.getLogger(__name__)

_COLUMNS = ("sequence", "locus", "cdr_mask_aa")
_GROUP_BY_LOCUS = {"IGH": 0, "IGK": 1, "IGL": 1}  # 0 heavy, 1 light
_PERPLEXITY_CAP = 1000.0


@register_eval_task("antibody_cdr3")
class AntibodyCdr3EvalTask(EvalTask["OplmForMaskedLM"]):
    """Score each CDR3 residue with only that residue masked (a CDR3-restricted PLL).

    Reads one parquet file with ``sequence``, ``locus`` (``IGH``/``IGK``/``IGL``) and
    ``cdr_mask_aa`` (one digit per residue, ``3`` marks CDR3), the held-out files of
    ``brineylab/oplm-antibody-sequences``. Every CDR3 position gets one forward pass with
    ``<mask>`` at that position alone, so the result is deterministic and covers every
    CDR3 residue. Metrics are ``{loss,accuracy,perplexity}_{heavy,light,all}``.

    Task key ``mask_batch_size`` (int, default 256, in ``[1, 4096]``): masked sequences per
    forward pass.
    """

    default_metrics: ClassVar[list[str]] = [
        f"{m}_{g}" for m in ("loss", "accuracy", "perplexity") for g in ("heavy", "light", "all")
    ]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.mask_batch_size = int(entry.extra.get("mask_batch_size", 256))
        if not 1 <= self.mask_batch_size <= 4096:
            raise ValueError("mask_batch_size must be in [1, 4096]")
        # (token ids, group, CDR3 token indices) per chain; loaded on first evaluate.
        self._chains: list[tuple[list[int], int, list[int]]] | None = None

    def _load(self) -> list[tuple[list[int], int, list[int]]]:
        table = pq.read_table(self.path)
        missing = set(_COLUMNS) - set(table.column_names)
        if missing:
            raise ValueError(f"{self.path} lacks columns {sorted(missing)}")
        tokenizer = get_tokenizer()
        max_len = self.cfg.model.max_position_embeddings
        chains = []
        for seq, locus, mask in zip(*(table.column(c).to_pylist() for c in _COLUMNS), strict=True):
            if locus not in _GROUP_BY_LOCUS:
                raise ValueError(f"{self.path}: unknown locus {locus!r}")
            if len(mask) != len(seq):
                raise ValueError(f"{self.path}: cdr_mask_aa length differs from sequence")
            if len(seq) + 2 > max_len:
                logger.warning("[%s] skipping a %d-residue chain (context)", self.name, len(seq))
                continue
            # Residue i sits at token i + 1 (BOS first).
            cdr3 = [i + 1 for i, d in enumerate(mask) if d == "3"]
            chains.append((tokenizer.encode(seq), _GROUP_BY_LOCUS[locus], cdr3))
        return chains

    def evaluate(self, model: OplmForMaskedLM, accelerator: Accelerator) -> dict[str, float]:
        """Run the per-position CDR3 pass on this rank's chains and reduce across ranks.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance.

        Returns:
            Dict of metric name to scalar value, filtered to requested metrics.
        """
        if self._chains is None:
            self._chains = self._load()
        device = accelerator.device
        tokenizer = get_tokenizer()
        mask_id = mask_token_id(tokenizer)
        pad_id = tokenizer.pad_token_id

        # Columns: summed loss, correct, count; rows: heavy, light.
        totals = torch.zeros(2, 3, device=device, dtype=torch.float64)
        # One (chain, token index) job per CDR3 residue; chains striped across ranks.
        jobs = [
            (ids, group, pos)
            for ids, group, cdr3 in self._chains[
                accelerator.process_index :: accelerator.num_processes
            ]
            for pos in cdr3
        ]
        with torch.no_grad():
            for start in range(0, len(jobs), self.mask_batch_size):
                chunk = jobs[start : start + self.mask_batch_size]
                width = max(len(ids) for ids, _, _ in chunk)
                input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                for row, (ids, _, pos) in enumerate(chunk):
                    input_ids[row, : len(ids)] = torch.tensor(ids)
                    attention_mask[row, : len(ids)] = 1
                    input_ids[row, pos] = mask_id
                rows = torch.arange(len(chunk))
                positions = torch.tensor([pos for _, _, pos in chunk])
                targets = torch.tensor([ids[pos] for ids, _, pos in chunk], device=device)
                groups = torch.tensor([g for _, g, _ in chunk], device=device)

                logits = model(
                    input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
                )["logits"][rows.to(device), positions.to(device)].float()  # (B, V)
                loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                correct = (logits.argmax(-1) == targets).double()
                stats = torch.stack([loss.double(), correct, torch.ones_like(correct)], dim=1)
                totals.index_add_(0, groups, stats)

        totals = accelerator.reduce(totals, reduction="sum")
        metrics: dict[str, float] = {}
        for name, (loss_sum, correct, n) in zip(
            ("heavy", "light", "all"), [*totals.tolist(), totals.sum(0).tolist()], strict=True
        ):
            loss = loss_sum / n if n else 0.0
            metrics[f"loss_{name}"] = loss
            metrics[f"accuracy_{name}"] = correct / n if n else 0.0
            metrics[f"perplexity_{name}"] = min(math.exp(loss), _PERPLEXITY_CAP)
        return {k: v for k, v in metrics.items() if k in self.metrics}
