"""Sequence dataloader builders.

``build_train_dataloader`` and ``build_sequence_eval_dataloader`` assemble the
datasets, MLM collator, and :class:`torch.utils.data.DataLoader` from an
:class:`oplm.config.OplmConfig`, applying train vs. eval shuffling/determinism
policy.
"""

from __future__ import annotations
