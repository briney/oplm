"""Tokenizer access layer.

A thin accessor over :class:`oplm.model.OplmTokenizerFast` (the single source of
truth for the vocabulary). Defines no vocabulary of its own; provides the
tokenizer accessor, derived id constants computed from the tokenizer instance,
and per-residue vector alignment to tokenized ``input_ids``.
"""

from __future__ import annotations
