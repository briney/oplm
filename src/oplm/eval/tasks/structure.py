"""Structure evaluation task — precision@L contact prediction from attention."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import torch

from oplm.data.tokenizer import ProteinTokenizer
from oplm.eval.data.structure_loader import StructureData, load_structures
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
        self.max_structures: int | None = (
            int(extra["max_structures"]) if "max_structures" in extra else None
        )

        # Cached data (lazily loaded)
        self._structures: list[StructureData] | None = None
        self._tokenizer: ProteinTokenizer | None = None

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

        if not self._structures:
            logger.warning("No structures loaded from %s", self.path)
            return {m: 0.0 for m in self.metrics}

        # Shard structures across ranks
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        rank_structures = self._structures[rank::world_size]

        # Process structures one at a time
        device = accelerator.device
        contact_data_list: list[StructureContactData] = []
        for struct in rank_structures:
            cd = self._process_single_structure(struct, model, device)
            if cd is not None:
                contact_data_list.append(cd)

        # Gather across ranks
        all_contact_data = self._gather_contact_data(contact_data_list, accelerator)

        if not all_contact_data:
            logger.warning("No valid structures after processing")
            return {m: 0.0 for m in self.metrics}

        # Compute metrics
        results: dict[str, float] = {}

        # Determine which l_divisors we need
        l_divisors: dict[str, int] = {}
        if "precision_at_L" in self.metrics:
            l_divisors["precision_at_L"] = 1
        if "precision_at_L_2" in self.metrics:
            l_divisors["precision_at_L_2"] = 2
        if "precision_at_L_5" in self.metrics:
            l_divisors["precision_at_L_5"] = 5

        # Default to precision_at_L if no specific metric matched
        if not l_divisors:
            l_divisors["precision_at_L"] = self.l_divisor

        for metric_name, divisor in l_divisors.items():
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

        return {k: v for k, v in results.items() if k in self.metrics}

    def _process_single_structure(
        self,
        struct: StructureData,
        model: OplmForMLM,
        device: torch.device,
    ) -> StructureContactData | None:
        """Process one structure: tokenize, forward pass, extract features.

        Attention weights are offloaded to CPU immediately after the
        forward pass to minimize GPU memory usage.
        """
        assert self._tokenizer is not None

        seq_len = len(struct.sequence)
        max_length = self.cfg.data.max_length
        if seq_len + 2 > max_length:  # +2 for CLS and EOS
            logger.debug(
                "Skipping %s: sequence length %d exceeds max_length %d",
                struct.name,
                seq_len,
                max_length - 2,
            )
            return None

        # Tokenize
        tokens = self._tokenizer.batch_encode([struct.sequence], max_length=max_length)
        input_ids = tokens["input_ids"].to(device)  # (1, T)
        attention_mask = tokens["attention_mask"].to(device)  # (1, T)

        # Forward pass with attention weights
        with torch.no_grad():
            # Expand 2D mask to 4D: (B, 1, 1, T) for the encoder
            attn_mask_4d = attention_mask.unsqueeze(1).unsqueeze(1).bool()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask_4d,
                need_weights=True,
            )

        # Immediately offload attention weights to CPU and strip special tokens.
        # Each layer's weights: (B, H, T, T) -> (H, L, L) after squeeze + slice.
        raw_attn = outputs["attention_weights"]
        attn_weights_cpu: list[torch.Tensor] = []
        for w in raw_attn:
            sliced = w.squeeze(0)[:, 1 : seq_len + 1, 1 : seq_len + 1].cpu()
            attn_weights_cpu.append(sliced)

        # Free GPU memory
        del raw_attn, outputs, input_ids, attention_mask, attn_mask_4d
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # All subsequent processing on CPU
        attn_contacts = extract_attention_contacts(
            attn_weights_cpu,
            layer="all",
            head_aggregation=None,
        )
        del attn_weights_cpu

        true_contacts = compute_contact_map(
            struct.coords,
            threshold=self.contact_threshold,
            use_cbeta=self.use_cbeta,
        )

        cd = build_structure_contact_data(attn_contacts, true_contacts, seq_len, self.min_seq_sep)
        del attn_contacts

        return cd

    def _gather_contact_data(
        self,
        local_data: list[StructureContactData],
        accelerator: Accelerator,
    ) -> list[StructureContactData]:
        """Gather StructureContactData from all ranks."""
        if accelerator.num_processes == 1:
            return local_data

        import torch.distributed as dist

        all_data_lists: list[list[StructureContactData] | None] = [None] * accelerator.num_processes
        dist.all_gather_object(all_data_lists, local_data)

        gathered: list[StructureContactData] = []
        for rank_data in all_data_lists:
            if rank_data is not None:
                gathered.extend(rank_data)
        return gathered
