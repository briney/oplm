"""open protein language model."""

from __future__ import annotations

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .model import (
    LogitsConfig,
    LogitsOutput,
    OplmConfig,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
    OplmModel,
    OplmTokenizerFast,
)

__version__ = "0.0.1"

# (1) In-process registration so `import oplm` plus AutoModel*.from_pretrained
# works without trust_remote_code. HF's `register` raises on a duplicate
# model_type, so guard each call to keep re-imports idempotent (§13.3).
AutoConfig.register("oplm", OplmConfig, exist_ok=True)
AutoModel.register(OplmConfig, OplmModel, exist_ok=True)
AutoModelForMaskedLM.register(OplmConfig, OplmForMaskedLM, exist_ok=True)
AutoModelForSequenceClassification.register(
    OplmConfig, OplmForSequenceClassification, exist_ok=True
)
AutoModelForTokenClassification.register(OplmConfig, OplmForTokenClassification, exist_ok=True)
AutoTokenizer.register(OplmConfig, fast_tokenizer_class=OplmTokenizerFast, exist_ok=True)

# (2) Tell HF to copy the custom-code .py files when push_to_hub is called and
# to write the matching auto_map entries into config.json / tokenizer_config.json.
# Setting auto_map manually is NOT sufficient — register_for_auto_class is the
# documented hook for the file-copy step. These set a class attribute, so repeat
# calls are no-ops.
OplmConfig.register_for_auto_class("AutoConfig")
OplmModel.register_for_auto_class("AutoModel")
OplmForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
OplmForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
OplmForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
OplmTokenizerFast.register_for_auto_class("AutoTokenizer")

__all__ = [
    "LogitsConfig",
    "LogitsOutput",
    "OplmConfig",
    "OplmForMaskedLM",
    "OplmForSequenceClassification",
    "OplmForTokenClassification",
    "OplmModel",
    "OplmTokenizerFast",
    "__version__",
]
