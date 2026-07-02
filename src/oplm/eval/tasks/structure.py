"""Structure evaluation task — categorical-Jacobian precision@L contact prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
import torch

from oplm.data import StructureData, get_tokenizer, load_structures
from oplm.eval.metrics.categorical_jacobian import (
    StructurePairScoreData,
    build_structure_pair_score_data,
    categorical_jacobian_to_contact_map,
    compute_categorical_jacobian,
    compute_mean_pair_score_precision_at_l,
    get_canonical_amino_acid_token_ids,
)
from oplm.eval.metrics.contact import compute_contact_map
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.model import OplmForMaskedLM, OplmTokenizerFast

logger = logging.getLogger(__name__)
T = TypeVar("T")

_METRIC_DIVISORS: dict[str, int] = {
    "precision_at_L": 1,
    "precision_at_L_2": 2,
    "precision_at_L_5": 5,
}


@dataclass(frozen=True)
class StructureTaskConfig:
    """Typed structure-eval knobs, parsed from EvalDatasetEntry.extra."""

    contact_threshold: float = 8.0
    min_seq_sep: int = 24
    l_divisor: int = 1
    use_cbeta: bool = True
    categorical_jacobian_sample_size: int | None = None
    categorical_jacobian_sample_seed: int = 42
    categorical_jacobian_mutation_batch_size: int = 20
    max_structures: int | None = None

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> StructureTaskConfig:
        """Build a config from an eval entry's ``extra`` block, validating types."""

        def _opt_int(key: str) -> int | None:
            return int(extra[key]) if key in extra else None

        def _strict_bool(key: str, default: bool) -> bool:
            # Same rationale as parse_schedule_block: bool("false") is True, so a YAML
            # string would silently invert the flag. Require an actual bool.
            value = extra.get(key, default)
            if not isinstance(value, bool):
                raise ValueError(
                    f"structure-eval {key!r} must be a bool, got {value!r} ({type(value).__name__})"
                )
            return value

        cfg = cls(
            contact_threshold=float(extra.get("contact_threshold", 8.0)),
            min_seq_sep=int(extra.get("min_seq_sep", 24)),
            l_divisor=int(extra.get("l_divisor", 1)),
            use_cbeta=_strict_bool("use_cbeta", True),
            categorical_jacobian_sample_size=_opt_int("categorical_jacobian_sample_size"),
            categorical_jacobian_sample_seed=int(extra.get("categorical_jacobian_sample_seed", 42)),
            categorical_jacobian_mutation_batch_size=int(
                extra.get("categorical_jacobian_mutation_batch_size", 20)
            ),
            max_structures=_opt_int("max_structures"),
        )
        if cfg.categorical_jacobian_sample_size is not None and (
            cfg.categorical_jacobian_sample_size < 1
        ):
            raise ValueError("categorical_jacobian_sample_size must be >= 1 when provided")
        if not 1 <= cfg.categorical_jacobian_mutation_batch_size <= 20:
            raise ValueError("categorical_jacobian_mutation_batch_size must be in [1, 20]")
        return cfg


@register_eval_task("structure")
class StructureEvalTask(EvalTask):
    """Evaluate contact prediction quality via the categorical Jacobian.

    Data format: directory containing PDB and/or CIF files.

    Precision@L is computed from the model's categorical Jacobian: for each
    structure, every residue is mutated to all 20 canonical amino acids, the
    resulting logit deltas form an ``(L, A, L, A)`` coupling tensor, which is
    centered, symmetrized, APC-corrected, and reduced to an ``(L, L)`` contact
    map. Long-range pairs are ranked to produce P@L.

    Task-specific config (extra keys on the eval dataset entry):
        contact_threshold (float): Distance cutoff in Å. Default 8.0.
        min_seq_sep (int): Minimum sequence separation. Default 24
            (long-range contacts, matching the reference paper).
        l_divisor (int): L divisor (1=L, 2=L/2, 5=L/5). Default 1.
        use_cbeta (bool): Use Cβ distances. Default True.
        categorical_jacobian_sample_size (int | None): Optional deterministic
            subset of structures to score (the Jacobian is expensive). Default
            None (all eligible structures).
        categorical_jacobian_sample_seed (int): Subsampling seed. Default 42.
        categorical_jacobian_mutation_batch_size (int): Mutants per forward
            pass for Jacobian extraction. Default 20.
        max_structures (int | None): Max structures to load. Default None.
    """

    default_metrics: ClassVar[list[str]] = ["precision_at_L"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.tcfg = StructureTaskConfig.from_extra(entry.extra)

        # Cached data (lazily loaded)
        self._structures: list[StructureData] | None = None
        self._tokenizer: OplmTokenizerFast | None = None
        self._canonical_aa_token_ids: torch.Tensor | None = None

    def evaluate(
        self,
        model: OplmForMaskedLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Run categorical-Jacobian contact prediction evaluation.

        Processes structures one at a time for memory efficiency. The Jacobian
        and intermediate tensors are offloaded to CPU as soon as each forward
        pass completes.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance.

        Returns:
            Dict of metric name to scalar value, filtered to requested metrics.
        """
        # Lazy initialization
        if self._structures is None:
            self._structures = load_structures(self.path, self.tcfg.max_structures)
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()
        if self._canonical_aa_token_ids is None:
            self._canonical_aa_token_ids = get_canonical_amino_acid_token_ids(self._tokenizer)

        if not self._structures:
            logger.warning("No structures loaded from %s", self.path)
            return {m: 0.0 for m in self.metrics}

        requested_metrics = self._requested_metric_divisors(_METRIC_DIVISORS)
        if not requested_metrics:
            return {}

        # Shard structures across ranks
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        rank_structures = self._structures[rank::world_size]
        selected_names = self._select_structure_names()

        # Process structures one at a time
        device = accelerator.device
        pair_score_data_list: list[StructurePairScoreData] = []
        for struct in rank_structures:
            if struct.name not in selected_names:
                continue
            pair_score_data = self._process_single_structure(struct, model, device)
            if pair_score_data is not None:
                pair_score_data_list.append(pair_score_data)

        # Gather across ranks
        all_pair_score_data = self._gather_data(pair_score_data_list, accelerator)

        if not all_pair_score_data:
            logger.warning("No valid structures after processing")
            return {m: 0.0 for m in self.metrics}

        results: dict[str, float] = {}
        for metric_name, divisor in requested_metrics.items():
            results[metric_name] = compute_mean_pair_score_precision_at_l(
                all_pair_score_data,
                l_divisor=divisor,
                min_seq_sep=self.tcfg.min_seq_sep,
            )

        return {k: v for k, v in results.items() if k in self.metrics}

    def _process_single_structure(
        self,
        struct: StructureData,
        model: OplmForMaskedLM,
        device: torch.device,
    ) -> StructurePairScoreData | None:
        """Compute categorical-Jacobian pair scores for one structure.

        Intermediate tensors are offloaded to CPU as soon as each forward
        pass completes to minimize GPU memory usage.
        """
        assert self._tokenizer is not None
        assert self._canonical_aa_token_ids is not None

        seq_len = len(struct.sequence)
        if not self._is_structure_eligible(struct):
            logger.debug(
                "Skipping %s: sequence length %d exceeds max_length %d",
                struct.name,
                seq_len,
                self.cfg.model.max_position_embeddings - 2,
            )
            return None

        token_ids_cpu = torch.tensor(self._tokenizer.encode(struct.sequence), dtype=torch.long)
        input_ids = token_ids_cpu.unsqueeze(0).to(device)  # (1, T)
        attention_mask = torch.ones_like(input_ids)  # (1, T)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        true_contacts = compute_contact_map(
            struct.coords,
            threshold=self.tcfg.contact_threshold,
            use_cbeta=self.tcfg.use_cbeta,
        )

        canonical_token_ids_device = self._canonical_aa_token_ids.to(device)
        wildtype_logits = outputs["logits"]
        wildtype_canonical_logits = (
            wildtype_logits[0, 1 : seq_len + 1]
            .index_select(-1, canonical_token_ids_device)
            .detach()
            .cpu()
            .float()
        )

        def logits_fn(batch_input_ids: torch.Tensor) -> torch.Tensor:
            batch_input_ids_device = batch_input_ids.to(device)
            batch_attention_mask = torch.ones_like(batch_input_ids_device)
            with torch.no_grad():
                batch_outputs = model(
                    input_ids=batch_input_ids_device,
                    attention_mask=batch_attention_mask,
                )
            batch_logits: torch.Tensor = (
                batch_outputs["logits"][:, 1 : seq_len + 1]
                .index_select(-1, canonical_token_ids_device)
                .detach()
                .cpu()
                .float()
            )
            del batch_outputs, batch_input_ids_device, batch_attention_mask
            return batch_logits

        categorical_jacobian = compute_categorical_jacobian(
            wildtype_input_ids=token_ids_cpu,
            wildtype_logits=wildtype_canonical_logits,
            canonical_token_ids=self._canonical_aa_token_ids,
            logits_fn=logits_fn,
            mutation_batch_size=self.tcfg.categorical_jacobian_mutation_batch_size,
        )
        jacobian_contacts = categorical_jacobian_to_contact_map(
            categorical_jacobian,
            copy=False,
        )
        pair_score_data = build_structure_pair_score_data(
            jacobian_contacts,
            true_contacts,
            seq_len,
            self.tcfg.min_seq_sep,
        )

        del (
            wildtype_logits,
            wildtype_canonical_logits,
            categorical_jacobian,
            jacobian_contacts,
            outputs,
            input_ids,
            attention_mask,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return pair_score_data

    def _gather_data(
        self,
        local_data: list[T],
        accelerator: Accelerator,
    ) -> list[T]:
        """Gather per-rank Python objects from all ranks."""
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

    def _requested_metric_divisors(self, supported_metrics: dict[str, int]) -> dict[str, int]:
        """Return requested metric names mapped to their P@L divisors."""
        return {
            metric_name: divisor
            for metric_name, divisor in supported_metrics.items()
            if metric_name in self.metrics
        }

    def _is_structure_eligible(self, struct: StructureData) -> bool:
        """Cheap length eligibility check before running a model forward."""
        return bool(len(struct.sequence) + 2 <= self.cfg.model.max_position_embeddings)

    def _select_structure_names(self) -> set[str]:
        """Choose the deterministic structure subset for Jacobian evaluation."""
        assert self._structures is not None

        eligible_structures = [
            struct for struct in self._structures if self._is_structure_eligible(struct)
        ]
        if not eligible_structures:
            return set()

        sample_size = self.tcfg.categorical_jacobian_sample_size
        if sample_size is None or sample_size >= len(eligible_structures):
            return {struct.name for struct in eligible_structures}

        rng = np.random.RandomState(self.tcfg.categorical_jacobian_sample_seed)
        sampled_indices = sorted(
            rng.choice(len(eligible_structures), size=sample_size, replace=False)
        )
        return {eligible_structures[index].name for index in sampled_indices}
