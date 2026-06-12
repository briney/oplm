"""ProteinGym clinical-variant eval — Pathogenic-vs-Benign AUROC.

Zero-shot pathogenicity prediction on ProteinGym's clinical-substitution
benchmark. Each CSV is one protein; variants are labelled ``Pathogenic`` /
``Benign``. The model scores every variant by a per-position log-likelihood
ratio; AUROC is computed per protein and averaged across proteins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
import torch

from oplm.data import get_tokenizer, load_clinical_variant_assays
from oplm.data.tokenizer import mask_token_id
from oplm.data.variant.loader import parse_mutation
from oplm.eval.metrics.classification import roc_auc_score
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.data import VariantAssay
    from oplm.model import OplmForMaskedLM, OplmTokenizerFast

logger = logging.getLogger(__name__)
T = TypeVar("T")

_VALID_SCORING = ("masked_marginals", "wt_marginals")
_MULTI_MUTANT_SEP = ":"


@dataclass(frozen=True)
class ProteinGymClinicalTaskConfig:
    """Typed clinical-variant eval knobs, parsed from ``EvalDatasetEntry.extra``.

    Attributes:
        scoring: Variant-scoring method, ``"masked_marginals"`` (mask each
            mutated position; more accurate) or ``"wt_marginals"`` (one wild-type
            forward pass; cheaper).
        max_assays: Optional cap on the number of assays (CSV files) loaded.
        mask_batch_size: Masked sequences per forward pass (``masked_marginals``
            only).
    """

    scoring: str = "masked_marginals"
    max_assays: int | None = None
    mask_batch_size: int = 64

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> ProteinGymClinicalTaskConfig:
        """Build a config from an eval entry's ``extra`` block, validating values."""
        cfg = cls(
            scoring=str(extra.get("scoring", "masked_marginals")),
            max_assays=int(extra["max_assays"]) if "max_assays" in extra else None,
            mask_batch_size=int(extra.get("mask_batch_size", 64)),
        )
        if cfg.scoring not in _VALID_SCORING:
            raise ValueError(f"scoring must be one of {_VALID_SCORING}, got {cfg.scoring!r}")
        if cfg.max_assays is not None and cfg.max_assays < 0:
            raise ValueError("max_assays must be >= 0 when provided")
        if not 1 <= cfg.mask_batch_size <= 1024:
            raise ValueError("mask_batch_size must be in [1, 1024]")
        return cfg


@register_eval_task("proteingym_clinical")
class ProteinGymClinicalEvalTask(EvalTask):
    """Zero-shot clinical-variant pathogenicity prediction (Pathogenic vs Benign).

    Data format: a directory of CSV files from the ProteinGym clinical-substitution
    benchmark, one protein per file, with columns ``mutant`` (e.g. ``"R46W"``),
    ``protein_sequence`` (the constant wild-type), and ``DMS_bin_score`` (the
    categorical label ``Pathogenic`` / ``Benign``).

    Evaluation protocol, per assay:
        1. Score each variant by the summed per-position log-likelihood ratio
           ``LLR = log P(mut_aa | context) - log P(wt_aa | context)`` (over the
           variant's substitutions), using ``masked_marginals`` or ``wt_marginals``.
        2. A higher LLR means the substitution is more tolerated (Benign-like).
           Pathogenic is the positive class, so the **pathogenicity score is
           ``-LLR``**: a good model ranks Pathogenic variants higher and yields
           AUROC > 0.5.
        3. Compute AUROC over (label, pathogenicity-score) pairs. Assays without
           both classes (or fewer than two rows) are skipped.
    Report the mean AUROC across all scorable assays.

    Task-specific config (extra keys on the eval dataset entry):
        scoring (str): ``masked_marginals`` (default) or ``wt_marginals``.
        max_assays (int | None): Cap on assays loaded. Default None (all).
        mask_batch_size (int): Masked sequences per forward pass. Default 64.
    """

    default_metrics: ClassVar[list[str]] = ["auroc"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.tcfg = ProteinGymClinicalTaskConfig.from_extra(entry.extra)

        # Cached data (lazily loaded on first evaluate()).
        self._assays: list[VariantAssay] | None = None
        self._tokenizer: OplmTokenizerFast | None = None

    def evaluate(
        self,
        model: OplmForMaskedLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Score clinical variants and return mean per-assay AUROC.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance (for distributed gather).

        Returns:
            Dict of metric name to scalar value, filtered to requested metrics.
        """
        if self._assays is None:
            self._assays = load_clinical_variant_assays(self.path, max_assays=self.tcfg.max_assays)
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()

        if not self._assays:
            logger.warning("No clinical variant assays loaded from %s", self.path)
            return {m: 0.0 for m in self.metrics}

        # Shard whole assays across ranks (AUROC is computed per assay).
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        rank_assays = self._assays[rank::world_size]

        device = accelerator.device
        local_aurocs: list[float] = []
        for assay in rank_assays:
            auroc = self._score_assay(assay, model, device)
            if auroc is not None:
                local_aurocs.append(auroc)

        all_aurocs = self._gather_data(local_aurocs, accelerator)
        if not all_aurocs:
            logger.warning("No scorable clinical assays (need both classes present)")
            return {m: 0.0 for m in self.metrics}

        results = {"auroc": sum(all_aurocs) / len(all_aurocs)}
        return {k: v for k, v in results.items() if k in self.metrics}

    def _score_assay(
        self,
        assay: VariantAssay,
        model: OplmForMaskedLM,
        device: torch.device,
    ) -> float | None:
        """Compute AUROC for one assay, or None if it cannot be scored.

        Returns None when the wild-type exceeds the model's context length, an
        amino acid is unknown to the tokenizer, or AUROC is undefined (a single
        class present).
        """
        assert self._tokenizer is not None

        wildtype = assay.wildtype
        if len(wildtype) + 2 > self.cfg.model.max_position_embeddings:
            logger.debug(
                "Skipping %s: wild-type length %d exceeds max context %d",
                assay.name,
                len(wildtype),
                self.cfg.model.max_position_embeddings - 2,
            )
            return None

        # Parse each row into its constituent single substitutions (0-based pos).
        parsed_rows = [
            [parse_mutation(part) for part in token.split(_MULTI_MUTANT_SEP)]
            for token in assay.mutations
        ]
        positions = sorted({m.pos - 1 for row in parsed_rows for m in row})

        try:
            if self.tcfg.scoring == "wt_marginals":
                pos_logprobs = self._wt_marginal_logprobs(wildtype, positions, model, device)
            else:
                pos_logprobs = self._masked_marginal_logprobs(wildtype, positions, model, device)
            llr = self._variant_llrs(parsed_rows, pos_logprobs, assay.name)
            labels = np.asarray(assay.labels, dtype=np.float64)
            # Pathogenic (label 1) should score high → use -LLR (tolerated = low).
            return roc_auc_score(labels, -llr)
        except ValueError as err:
            logger.warning("[%s] skipping: %s", assay.name, err)
            return None

    def _variant_llrs(
        self,
        parsed_rows: list[list[Any]],
        pos_logprobs: dict[int, torch.Tensor],
        assay_name: str,
    ) -> np.ndarray:
        """Sum per-position log-likelihood ratios into one score per variant row."""
        assert self._tokenizer is not None
        tokenizer = self._tokenizer  # local narrowing so the closure sees a non-None tokenizer
        unk_id = tokenizer.unk_token_id
        aa_ids: dict[str, int] = {}

        def token_id(aa: str) -> int:
            if aa not in aa_ids:
                tid = tokenizer.convert_tokens_to_ids(aa)  # single token -> int
                if not isinstance(tid, int) or tid == unk_id:
                    raise ValueError(
                        f"[{assay_name}] amino acid {aa!r} maps to <unk>; cannot score"
                    )
                aa_ids[aa] = tid
            return aa_ids[aa]

        scores = np.empty(len(parsed_rows), dtype=np.float64)
        for k, row in enumerate(parsed_rows):
            total = 0.0
            for mutation in row:
                logprobs = pos_logprobs[mutation.pos - 1]
                total += float(logprobs[token_id(mutation.mut)] - logprobs[token_id(mutation.wt)])
            scores[k] = total
        return scores

    def _wt_marginal_logprobs(
        self,
        wildtype: str,
        positions: list[int],
        model: OplmForMaskedLM,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        """Log-softmax logits at each position from one wild-type forward pass.

        Returns a mapping ``{residue_index_0based: (V,) log-prob tensor}``.
        """
        assert self._tokenizer is not None
        input_ids = torch.tensor(self._tokenizer.encode(wildtype), dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        logprobs = torch.log_softmax(logits[0].float(), dim=-1).cpu()  # (T, V)
        # Residue i (0-based) lives at token index i+1 (BOS at index 0).
        return {i: logprobs[i + 1] for i in positions}

    def _masked_marginal_logprobs(
        self,
        wildtype: str,
        positions: list[int],
        model: OplmForMaskedLM,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        """Log-softmax logits at each position with that position masked.

        One masked forward per unique position, batched up to ``mask_batch_size``.
        Returns a mapping ``{residue_index_0based: (V,) log-prob tensor}``.
        """
        assert self._tokenizer is not None
        base = torch.tensor(self._tokenizer.encode(wildtype), dtype=torch.long)  # (T,)
        mask_id = mask_token_id(self._tokenizer)

        result: dict[int, torch.Tensor] = {}
        batch_size = self.tcfg.mask_batch_size
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            batch = base.unsqueeze(0).repeat(len(chunk), 1)  # (B, T)
            for row_idx, pos in enumerate(chunk):
                batch[row_idx, pos + 1] = mask_id  # BOS offset
            input_ids = batch.to(device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            for row_idx, pos in enumerate(chunk):
                result[pos] = torch.log_softmax(logits[row_idx, pos + 1].float(), dim=-1).cpu()
            del logits, input_ids, attention_mask, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return result

    def _gather_data(self, local_data: list[T], accelerator: Accelerator) -> list[T]:
        """Gather per-rank Python objects from all ranks into a flat list."""
        if accelerator.num_processes == 1:
            return local_data

        import torch.distributed as dist

        all_data_lists: list[list[T] | None] = [None] * accelerator.num_processes
        dist.all_gather_object(all_data_lists, local_data)

        gathered: list[T] = []
        for rank_data in all_data_lists:
            if rank_data is not None:
                gathered.extend(rank_data)
        return gathered
