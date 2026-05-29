"""Data-config parsing helpers.

Parsing for the train/eval dataset specifications declared in
:mod:`oplm.config` (``TrainDatasetEntry`` / ``EvalDatasetEntry`` are imported
from there, not redefined here). Normalizes train-dataset fractions and folds
unknown eval keys into ``extra``.
"""

from __future__ import annotations
