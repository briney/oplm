"""Tokenization, padding, and masked-language-model collation.

``tokenize_and_pad`` is the shared pad/tokenize primitive (no masking).
:class:`MLMCollator` layers fixed-``k`` Gumbel-top-k masking with BERT 80/10/10
replacement on top, supporting optional per-residue weighted masking.
"""

from __future__ import annotations
