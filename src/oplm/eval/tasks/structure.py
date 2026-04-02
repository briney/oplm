"""Structure evaluation task — precision@L contact prediction from attention."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, TypeVar

import numpy as np
import torch

from oplm.data.tokenizer import ProteinTokenizer
from oplm.eval.data.structure_loader import StructureData, load_structures
from oplm.eval.metrics.categorical_jacobian import (
    StructurePairScoreData,
    build_structure_pair_score_data,
    categorical_jacobian_to_contact_map,
    compute_categorical_jacobian,
    compute_mean_pair_score_precision_at_l,
    get_canonical_amino_acid_token_ids,
)
from oplm.eval.metrics.contact import (
    StructureContactData,
    _fallback_mean_attention_precision,
    build_structure_contact_data,
    compute_contact_map,
    compute_logreg_precision_at_l,
    extract_attention_contacts,
)
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.model.transformer import OplmForMLM

logger = logging.getLogger(__name__)
T = TypeVar("T")

_ATTENTION_METRIC_DIVISORS: dict[str, int] = {
    "precision_at_L": 1,
    "precision_at_L_2": 2,
    "precision_at_L_5": 5,
}
_CATEGORICAL_JACOBIAN_METRIC_DIVISORS: dict[str, int] = {
    "categorical_jacobian_precision_at_L": 1,
    "categorical_jacobian_precision_at_L_2": 2,
    "categorical_jacobian_precision_at_L_5": 5,
}


@register_eval_task("structure")
class StructureEvalTask(EvalTask):
    """Evaluate contact prediction quality from attention weights.

    Data format: directory containing PDB and/or CIF files.

    Default mode uses logistic regression P@L following the ESM-1b
    protocol (Rives et al., 2021): extract all-layer, all-head attention
    maps, symmetrize, APC-correct each individually, fit L1 logistic
    regression. Falls back to mean-attention P@L when insufficient
    structures are available.

    Task-specific config (extra keys on the eval dataset entry):
        contact_threshold (float): Distance cutoff in Å. Default 8.0.
        min_seq_sep (int): Minimum sequence separation. Default 6.
        l_divisor (int): L divisor (1=L, 2=L/2, 5=L/5). Default 1.
        use_cbeta (bool): Use Cβ distances. Default True.
        use_logistic_regression (bool): Use logreg P@L. Default True.
        logreg_n_train (int): Structures for logreg training. Default 20.
        logreg_n_iterations (int): Cross-validation iterations. Default 5.
        logreg_c (float): Inverse regularization strength. Default 0.15.
        use_categorical_jacobian (bool): Enable categorical-Jacobian P@L.
            Default False.
        categorical_jacobian_sample_size (int | None): Optional deterministic
            subset size for the Jacobian path. Default None.
        categorical_jacobian_sample_seed (int): Subsampling seed. Default 42.
        categorical_jacobian_mutation_batch_size (int): Mutants per forward
            pass for Jacobian extraction. Default 20.
        max_structures (int | None): Max structures to load. Default None.
    """

    default_metrics: ClassVar[list[str]] = ["precision_at_L"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        extra = entry.extra

        self.contact_threshold: float = float(extra.get("contact_threshold", 8.0))
        self.min_seq_sep: int = int(extra.get("min_seq_sep", 6))
        self.l_divisor: int = int(extra.get("l_divisor", 1))
        self.use_cbeta: bool = bool(extra.get("use_cbeta", True))
        self.use_logistic_regression: bool = bool(extra.get("use_logistic_regression", True))
        self.logreg_n_train: int = int(extra.get("logreg_n_train", 20))
        self.logreg_n_iterations: int = int(extra.get("logreg_n_iterations", 5))
        self.logreg_c: float = float(extra.get("logreg_c", 0.15))
        self.use_categorical_jacobian: bool = bool(extra.get("use_categorical_jacobian", False))
        self.categorical_jacobian_sample_size: int | None = (
            int(extra["categorical_jacobian_sample_size"])
            if "categorical_jacobian_sample_size" in extra
            else None
        )
        self.categorical_jacobian_sample_seed: int = int(
            extra.get("categorical_jacobian_sample_seed", 42)
        )
        self.categorical_jacobian_mutation_batch_size: int = int(
            extra.get("categorical_jacobian_mutation_batch_size", 20)
        )
        self.max_structures: int | None = (
            int(extra["max_structures"]) if "max_structures" in extra else None
        )
        if self.categorical_jacobian_sample_size is not None and (
            self.categorical_jacobian_sample_size < 1
        ):
            raise ValueError("categorical_jacobian_sample_size must be >= 1 when provided")
        if not 1 <= self.categorical_jacobian_mutation_batch_size <= 20:
            raise ValueError("categorical_jacobian_mutation_batch_size must be in [1, 20]")
        if entry.metrics is None and self.use_categorical_jacobian:
            self.metrics = ["precision_at_L", "categorical_jacobian_precision_at_L"]

        # Cached data (lazily loaded)
        self._structures: list[StructureData] | None = None
        self._tokenizer: ProteinTokenizer | None = None
        self._canonical_aa_token_ids: torch.Tensor | None = None

    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Run contact prediction evaluation.

        Processes structures one at a time for memory efficiency.
        Attention weights are offloaded to CPU immediately after the
        forward pass.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance.

        Returns:
            Dict of metric name to scalar value, filtered to requested metrics.
        """
        # Lazy initialization
        if self._structures is None:
            self._structures = load_structures(self.path, self.max_structures)
        if self._tokenizer is None:
            self._tokenizer = ProteinTokenizer()
        if self._canonical_aa_token_ids is None:
            self._canonical_aa_token_ids = get_canonical_amino_acid_token_ids(self._tokenizer)

        if not self._structures:
            logger.warning("No structures loaded from %s", self.path)
            return {m: 0.0 for m in self.metrics}

        attention_metrics = self._requested_metric_divisors(_ATTENTION_METRIC_DIVISORS)
        categorical_jacobian_metrics = self._requested_metric_divisors(
            _CATEGORICAL_JACOBIAN_METRIC_DIVISORS
        )
        if not attention_metrics and not categorical_jacobian_metrics:
            return {}

        # Shard structures across ranks
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        rank_structures = self._structures[rank::world_size]
        categorical_jacobian_names = self._select_categorical_jacobian_structure_names(
            needs_categorical_jacobian=bool(categorical_jacobian_metrics)
        )

        # Process structures one at a time
        device = accelerator.device
        contact_data_list: list[StructureContactData] = []
        categorical_jacobian_data_list: list[StructurePairScoreData] = []
        for struct in rank_structures:
            need_attention = bool(attention_metrics)
            need_categorical_jacobian = struct.name in categorical_jacobian_names
            if not need_attention and not need_categorical_jacobian:
                continue
            contact_data, categorical_jacobian_data = self._process_single_structure(
                struct,
                model,
                device,
                need_attention=need_attention,
                need_categorical_jacobian=need_categorical_jacobian,
            )
            if contact_data is not None:
                contact_data_list.append(contact_data)
            if categorical_jacobian_data is not None:
                categorical_jacobian_data_list.append(categorical_jacobian_data)

        # Gather across ranks
        all_contact_data = self._gather_data(contact_data_list, accelerator)
        all_categorical_jacobian_data = self._gather_data(
            categorical_jacobian_data_list, accelerator
        )

        if not all_contact_data and not all_categorical_jacobian_data:
            logger.warning("No valid structures after processing")
            return {m: 0.0 for m in self.metrics}

        results: dict[str, float] = {}
        if attention_metrics:
            if not all_contact_data:
                logger.warning("No valid structures for attention-based P@L")
                for metric_name in attention_metrics:
                    results[metric_name] = 0.0
            else:
                for metric_name, divisor in attention_metrics.items():
                    if self.use_logistic_regression:
                        p = compute_logreg_precision_at_l(
                            all_contact_data,
                            n_train=self.logreg_n_train,
                            n_iterations=self.logreg_n_iterations,
                            logreg_c=self.logreg_c,
                            l_divisor=divisor,
                            min_seq_sep=self.min_seq_sep,
                        )
                    else:
                        p = _fallback_mean_attention_precision(
                            all_contact_data,
                            l_divisor=divisor,
                            min_seq_sep=self.min_seq_sep,
                        )
                    results[metric_name] = p

        if categorical_jacobian_metrics:
            if not all_categorical_jacobian_data:
                logger.warning("No valid structures for categorical-Jacobian P@L")
                for metric_name in categorical_jacobian_metrics:
                    results[metric_name] = 0.0
            else:
                for metric_name, divisor in categorical_jacobian_metrics.items():
                    results[metric_name] = compute_mean_pair_score_precision_at_l(
                        all_categorical_jacobian_data,
                        l_divisor=divisor,
                        min_seq_sep=self.min_seq_sep,
                    )

        return {k: v for k, v in results.items() if k in self.metrics}

    def _process_single_structure(
        self,
        struct: StructureData,
        model: OplmForMLM,
        device: torch.device,
        need_attention: bool,
        need_categorical_jacobian: bool,
    ) -> tuple[StructureContactData | None, StructurePairScoreData | None]:
        """Process one structure for any requested contact-prediction paths.

        Attention weights are offloaded to CPU immediately after the
        forward pass to minimize GPU memory usage.
        """
        assert self._tokenizer is not None
        assert self._canonical_aa_token_ids is not None

        seq_len = len(struct.sequence)
        if not self._is_structure_eligible(struct):
            logger.debug(
                "Skipping %s: sequence length %d exceeds max_length %d",
                struct.name,
                seq_len,
                self.cfg.model.max_seq_len - 2,
            )
            return None, None

        token_ids_cpu = torch.tensor(self._tokenizer.encode(struct.sequence), dtype=torch.long)
        input_ids = token_ids_cpu.unsqueeze(0).to(device)  # (1, T)
        attention_mask = torch.ones_like(input_ids)  # (1, T)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                need_weights=need_attention,
            )

        true_contacts = compute_contact_map(
            struct.coords,
            threshold=self.contact_threshold,
            use_cbeta=self.use_cbeta,
        )

        attention_data: StructureContactData | None = None
        if need_attention:
            raw_attn = outputs["attention_weights"]
            if raw_attn is None:
                raise ValueError("Model did not return attention weights for structure evaluation")

            attn_weights_cpu: list[torch.Tensor] = []
            for weights in raw_attn:
                sliced = weights.squeeze(0)[:, 1 : seq_len + 1, 1 : seq_len + 1].cpu()
                attn_weights_cpu.append(sliced)
            attn_contacts = extract_attention_contacts(
                attn_weights_cpu,
                layer="all",
                head_aggregation=None,
            )
            attention_data = build_structure_contact_data(
                attn_contacts,
                true_contacts,
                seq_len,
                self.min_seq_sep,
            )
            del raw_attn, attn_weights_cpu, attn_contacts

        categorical_jacobian_data: StructurePairScoreData | None = None
        if need_categorical_jacobian:
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
                mutation_batch_size=self.categorical_jacobian_mutation_batch_size,
            )
            jacobian_contacts = categorical_jacobian_to_contact_map(
                categorical_jacobian,
                copy=False,
            )
            categorical_jacobian_data = build_structure_pair_score_data(
                jacobian_contacts,
                true_contacts,
                seq_len,
                self.min_seq_sep,
            )
            del wildtype_logits, wildtype_canonical_logits, categorical_jacobian, jacobian_contacts

        del outputs, input_ids, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return attention_data, categorical_jacobian_data

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
        return len(struct.sequence) + 2 <= self.cfg.model.max_seq_len

    def _select_categorical_jacobian_structure_names(
        self,
        needs_categorical_jacobian: bool,
    ) -> set[str]:
        """Choose the deterministic structure subset for Jacobian evaluation."""
        if not needs_categorical_jacobian:
            return set()
        assert self._structures is not None

        eligible_structures = [
            struct for struct in self._structures if self._is_structure_eligible(struct)
        ]
        if not eligible_structures:
            return set()

        sample_size = self.categorical_jacobian_sample_size
        if sample_size is None or sample_size >= len(eligible_structures):
            return {struct.name for struct in eligible_structures}

        rng = np.random.RandomState(self.categorical_jacobian_sample_seed)
        sampled_indices = sorted(
            rng.choice(len(eligible_structures), size=sample_size, replace=False)
        )
        return {eligible_structures[index].name for index in sampled_indices}
