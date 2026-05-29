"""Variant-effect loading.

Loads zero-shot variant-effect assays (e.g. ProteinGym) into
:class:`VariantAssay`, parsing and validating mutations against the wild-type
sequence. Scoring itself lives in the eval harness; this module only loads and
validates.
"""

from __future__ import annotations
