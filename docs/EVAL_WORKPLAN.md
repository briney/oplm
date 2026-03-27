# Evaluation Harness — Technical Implementation Plan

## Overview

A modular, config-driven evaluation harness for OPLM that supports multiple eval dataset
types, per-dataset metric selection and scheduling, and seamless integration with the
existing Trainer and wandb logging infrastructure.

The harness is designed around three principles:
1. **Intuitive configuration** — eval datasets are specified via the same CLI/YAML
   override pattern used for training data, with sensible defaults per dataset type.
2. **Extensibility** — new eval types (ProteinGym, TAPE, etc.) are added by implementing
   a single class and registering it; no changes to the Evaluator or Trainer.
3. **Per-dataset scheduling** — expensive evals (e.g., precision@L) can run at a lower
   frequency than cheap ones (e.g., sequence perplexity), all managed transparently.

---

## Architecture

### Module layout

```
src/oplm/eval/
├── __init__.py              # Public API: Evaluator, EvalTask, register_eval_task
├── evaluator.py             # Evaluator orchestrator class
├── registry.py              # @register_eval_task decorator + task registry
├── tasks/
│   ├── __init__.py          # Imports all task modules to trigger registration
│   ├── base.py              # EvalTask abstract base class
│   ├── sequence.py          # SequenceEvalTask — MLM loss/accuracy/perplexity
│   ├── structure.py         # StructureEvalTask — precision@L contact prediction
│   ├── proteingym.py        # (stub) ProteinGymEvalTask — zero-shot fitness
│   ├── tape.py              # (stub) TapeEvalTask — TAPE benchmark
│   ├── proteinglue.py       # (stub) ProteinGlueEvalTask
│   └── everest.py           # (stub) EverestEvalTask
├── metrics/
│   ├── __init__.py
│   ├── mlm.py               # MLM loss, masked accuracy, perplexity
│   └── contact.py           # Contact map computation, APC, precision@L, logreg
└── data/
    ├── __init__.py
    ├── sequence_loader.py   # Load eval parquet → DataLoader (with fixed MLM masking)
    └── structure_loader.py  # Parse PDB/CIF directory → list of StructureData
```

### Key classes

```
┌──────────────────────────────────────────────────────────────────┐
│ Evaluator                                                        │
│                                                                  │
│  Holds all configured EvalTask instances.                        │
│  Called by the Trainer every optimizer step.                      │
│  Internally checks which tasks are due, runs them, returns       │
│  merged metrics dict.                                            │
│                                                                  │
│  Conforms to EvalFn signature so it plugs into Trainer as-is.    │
├──────────────────────────────────────────────────────────────────┤
│ EvalTask (abstract)                                              │
│                                                                  │
│  One per configured eval dataset. Knows how to load its data,    │
│  what metrics it computes, and how to run evaluation.            │
│  Subclasses: SequenceEvalTask, StructureEvalTask, ...            │
├──────────────────────────────────────────────────────────────────┤
│ Registry                                                         │
│                                                                  │
│  Maps type strings ("sequence", "structure", "proteingym", ...)  │
│  to EvalTask subclasses via @register_eval_task decorator.       │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow

```
Config (CLI/YAML)
    │
    ▼
parse_eval_configs()  ──►  list[EvalDatasetEntry]
    │
    ▼
Evaluator.__init__()
    │  For each entry:
    │    1. Look up task class from registry by entry.type
    │    2. Instantiate task with entry config (path, metrics, etc.)
    │    3. Store task with its eval_every schedule
    │
    ▼
Trainer.train() loop
    │  Every optimizer step:
    │    evaluator(model, accelerator, global_step)
    │      │
    │      ▼
    │    For each task where global_step % task.eval_every == 0:
    │      task.evaluate(model, accelerator) → dict[str, float]
    │      Prefix keys: eval/{task.name}/{metric}
    │
    ▼
wandb.log(merged_metrics, step=global_step)
```

---

## Phase 1: Config & Framework

### 1.1 Config changes (`src/oplm/config.py`)

Add an `EvalDatasetEntry` dataclass and extend `DataConfig`:

```python
@dataclass
class EvalDatasetEntry:
    """Parsed configuration for a single evaluation dataset."""

    name: str
    path: str
    type: str                         # "sequence", "structure", "proteingym", ...
    eval_every: int | None = None     # Per-dataset override; None → use train.eval_every
    metrics: list[str] | None = None  # Override default metrics; None → use type defaults
```

Extend `DataConfig`:

```python
@dataclass
class DataConfig:
    train: Any = None
    eval: Any = None   # ← new field, same flexible format as train

    # ...existing fields...
```

Add `parse_eval_configs(raw, default_eval_every)` following the pattern of
`parse_train_configs()`, but parsing `type`, `eval_every`, and `metrics` fields
from each entry instead of `fraction`.

**CLI usage:**

```bash
# Single sequence eval dataset
data.eval.heldout.path=/data/heldout data.eval.heldout.type=sequence

# Multiple datasets with different schedules
data.eval.heldout.path=/data/heldout \
data.eval.heldout.type=sequence \
data.eval.pdb_contacts.path=/data/structures \
data.eval.pdb_contacts.type=structure \
data.eval.pdb_contacts.eval_every=10000
```

**YAML equivalent:**

```yaml
data:
  eval:
    heldout:
      path: /data/heldout
      type: sequence
    pdb_contacts:
      path: /data/structures
      type: structure
      eval_every: 10000
    proteingym_subset:
      path: /data/proteingym
      type: proteingym
      eval_every: 20000
      metrics: [spearman]
```

### 1.2 Registry (`src/oplm/eval/registry.py`)

A simple decorator-based registry mapping type strings to `EvalTask` subclasses.

```python
EVAL_TASK_REGISTRY: dict[str, type[EvalTask]] = {}

def register_eval_task(type_name: str):
    """Decorator to register an EvalTask subclass for a dataset type."""
    def decorator(cls):
        EVAL_TASK_REGISTRY[type_name] = cls
        return cls
    return decorator

def get_eval_task_class(type_name: str) -> type[EvalTask]:
    """Look up a registered EvalTask class by type string."""
    if type_name not in EVAL_TASK_REGISTRY:
        available = ", ".join(sorted(EVAL_TASK_REGISTRY))
        raise ValueError(
            f"Unknown eval type {type_name!r}. Available: {available}"
        )
    return EVAL_TASK_REGISTRY[type_name]
```

### 1.3 EvalTask base class (`src/oplm/eval/tasks/base.py`)

```python
class EvalTask(ABC):
    """Abstract base class for evaluation tasks.

    Each subclass handles one eval dataset type: loading data, computing metrics,
    and returning results. Subclasses are registered via @register_eval_task.
    """

    # Class-level defaults. Subclasses override these.
    default_metrics: ClassVar[list[str]] = []

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        self.name = entry.name
        self.path = entry.path
        self.eval_every = entry.eval_every or cfg.train.eval_every
        self.metrics = entry.metrics or self.default_metrics
        self.cfg = cfg

    @abstractmethod
    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Run evaluation and return metrics.

        Keys should be bare metric names (e.g. "loss", "perplexity").
        The Evaluator prefixes them with eval/{self.name}/.

        Args:
            model: The unwrapped model in eval mode.
            accelerator: The Accelerator instance (for distributed ops).

        Returns:
            Mapping of metric name to scalar value.
        """
        ...
```

### 1.4 Evaluator (`src/oplm/eval/evaluator.py`)

The Evaluator is the single integration point between the training loop and all
eval tasks. It conforms to the existing `EvalFn` signature.

```python
class Evaluator:
    """Orchestrates evaluation across multiple datasets and schedules.

    Instantiated once at training start. Called every optimizer step by the
    Trainer. Internally determines which tasks are due at the current step
    and runs only those.
    """

    def __init__(self, cfg: OplmConfig) -> None:
        entries = parse_eval_configs(cfg.data.eval, cfg.train.eval_every)
        self.tasks: list[EvalTask] = []
        for entry in entries:
            cls = get_eval_task_class(entry.type)
            self.tasks.append(cls(entry, cfg))

    def __call__(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
        global_step: int,
    ) -> dict[str, float]:
        """Run all due evaluations for the current step.

        Args:
            model: Unwrapped model (Evaluator handles eval/train mode toggle).
            accelerator: Accelerator instance.
            global_step: Current optimizer step.

        Returns:
            Merged metrics dict with eval/{name}/{metric} keys.
            Empty dict if no evals are due.
        """
        due_tasks = [t for t in self.tasks if global_step % t.eval_every == 0]
        if not due_tasks:
            return {}

        model.eval()
        all_metrics: dict[str, float] = {}
        try:
            for task in due_tasks:
                raw = task.evaluate(model, accelerator)
                for key, value in raw.items():
                    all_metrics[f"eval/{task.name}/{key}"] = value
        finally:
            model.train()

        return all_metrics

    @property
    def has_tasks(self) -> bool:
        """Whether any eval tasks are configured."""
        return len(self.tasks) > 0
```

### 1.5 Trainer integration (`src/oplm/training/trainer.py`)

Changes to the Trainer:

1. **Replace `eval_fn` parameter** with optional `Evaluator`. If `data.eval` is
   configured, the Trainer builds an `Evaluator` automatically. A manually passed
   `eval_fn` is still supported for backward compatibility.

2. **Replace the fixed `eval_every` check** in the training loop. Instead of:
   ```python
   if self.global_step % cfg.eval_every == 0:
       eval_metrics = self.evaluate()
   ```
   Change to:
   ```python
   eval_metrics = self._run_eval()
   if eval_metrics:
       self.accelerator.log(eval_metrics, step=self.global_step)
   ```
   Where `_run_eval()` calls the Evaluator (which handles its own scheduling)
   and/or the legacy `eval_fn` (at the original `eval_every` cadence).

3. **The Evaluator handles `model.eval()`/`model.train()` internally**, so the
   Trainer just calls and logs.

Detailed changes:

```python
# In __init__:
self.eval_fn = eval_fn  # Legacy callback (unchanged)
self.evaluator: Evaluator | None = None
if cfg.data.eval is not None:
    from oplm.eval import Evaluator
    self.evaluator = Evaluator(cfg)

# New method:
def _run_eval(self) -> dict[str, float]:
    """Run all due evaluations for the current step."""
    metrics: dict[str, float] = {}

    # New evaluator (per-dataset scheduling)
    if self.evaluator is not None:
        unwrapped = self.accelerator.unwrap_model(self.model)
        metrics.update(self.evaluator(unwrapped, self.accelerator, self.global_step))

    # Legacy eval_fn (original eval_every cadence)
    if self.eval_fn is not None and self.global_step % self.cfg.train.eval_every == 0:
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        metrics.update(self.eval_fn(unwrapped, self.accelerator, self.global_step))
        self.model.train()

    return metrics

# In training loop, replace the eval block with:
eval_metrics = self._run_eval()
if eval_metrics:
    if "eval/loss" in eval_metrics:
        self._last_eval_loss = eval_metrics["eval/loss"]
    self.accelerator.log(eval_metrics, step=self.global_step)
```

Note: the `_run_eval()` call happens every optimizer step, but returns an empty
dict (and does essentially no work) when no evals are due. The cost is one
function call + len(tasks) modulo operations per step, which is negligible.

### 1.6 `__init__.py` public API (`src/oplm/eval/__init__.py`)

```python
from oplm.eval.evaluator import Evaluator
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask
```

The `tasks/__init__.py` imports all task modules to trigger registration:

```python
from oplm.eval.tasks import sequence, structure
# Future: proteingym, tape, proteinglue, everest
```

---

## Phase 2: Sequence Evaluation

The most common eval type. Computes MLM metrics on a held-out set of protein
sequences in the same parquet format as training data.

### 2.1 Default metrics

| Metric | Description |
|---|---|
| `loss` | Cross-entropy loss on masked positions |
| `accuracy` | Fraction of masked positions predicted correctly |
| `perplexity` | exp(loss), capped at 1000 |

### 2.2 Data loading (`src/oplm/eval/data/sequence_loader.py`)

Builds a DataLoader from eval parquet files using the existing data pipeline,
but with **fixed 15% MLM masking** independent of the training mask probability.

```python
def build_sequence_eval_dataloader(
    path: str,
    cfg: OplmConfig,
    accelerator: Accelerator,
) -> DataLoader:
    """Build an eval DataLoader from parquet files.

    Uses the same ShardedProteinDataset and ProteinTokenizer as training,
    but with a fixed 15% mask probability (independent of training config)
    and deterministic masking via a fixed random seed.

    The DataLoader is NOT prepared with accelerator.prepare() — the task
    handles distributed evaluation explicitly via accelerator.gather().
    """
```

Key differences from training data loading:
- **Fixed mask_prob=0.15** regardless of `data.mask_prob` in training config.
- **Deterministic masking** — the collator uses a fixed seed so that the same
  positions are masked on every eval run, making results comparable across
  training steps.
- **No shuffling** — eval data is iterated in a fixed order.
- **Distributed splitting** — use `accelerator.prepare()` on the DataLoader so
  each rank processes a disjoint shard, then gather metrics.

### 2.3 Metric computation (`src/oplm/eval/metrics/mlm.py`)

Simple, stateless metric functions:

```python
def compute_mlm_metrics(
    model: OplmForMLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Compute MLM metrics over an eval DataLoader.

    Runs the model on all batches, accumulating loss and accuracy across
    the full dataset. Handles distributed gathering so each rank contributes
    its shard and the final metrics reflect the complete dataset.

    Returns:
        Dict with keys: "loss", "accuracy", "perplexity".
    """
```

Implementation outline:
1. Iterate batches, run `model(input_ids, attention_mask, labels)`.
2. Accumulate:
   - `total_loss` (sum of per-token losses, not mean-reduced)
   - `total_correct` (count of correctly predicted masked tokens)
   - `total_masked` (count of masked positions)
3. Gather accumulators across ranks via `accelerator.reduce(tensor, "sum")`.
4. Compute final metrics on gathered totals:
   - `loss = total_loss / total_masked`
   - `accuracy = total_correct / total_masked`
   - `perplexity = min(exp(loss), 1000.0)`

### 2.4 SequenceEvalTask (`src/oplm/eval/tasks/sequence.py`)

```python
@register_eval_task("sequence")
class SequenceEvalTask(EvalTask):
    """Evaluate MLM performance on held-out protein sequences.

    Data format: single parquet file or directory of sharded parquet files
    with columns (sequence_id, sequence) — same as training data.
    """

    default_metrics: ClassVar[list[str]] = ["loss", "accuracy", "perplexity"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self._dataloader: DataLoader | None = None  # Lazy init

    def evaluate(self, model, accelerator) -> dict[str, float]:
        if self._dataloader is None:
            self._dataloader = build_sequence_eval_dataloader(
                self.path, self.cfg, accelerator
            )
        all_metrics = compute_mlm_metrics(model, self._dataloader, accelerator)
        # Filter to only requested metrics
        return {k: v for k, v in all_metrics.items() if k in self.metrics}
```

The DataLoader is lazily initialized on first eval call (not at Evaluator
construction time). This avoids loading eval data if training ends before the
first eval step, and ensures the accelerator is fully set up.

---

## Phase 3: Structure Evaluation (Precision@L)

Evaluates the model's learned contact information by predicting residue-residue
contacts from attention weights and comparing to ground-truth contact maps
derived from 3D structures.

### 3.1 Default metrics

| Metric | Description |
|---|---|
| `precision_at_L` | Precision of top-L long-range contact predictions |

Additional configurable variants (via metric overrides or task-level config):
- `precision_at_L_2` — top L/2 predictions
- `precision_at_L_5` — top L/5 predictions

### 3.2 Data loading (`src/oplm/eval/data/structure_loader.py`)

Structures are loaded from a directory of PDB/CIF files. Each file yields a
sequence and backbone coordinates.

```python
@dataclass
class StructureData:
    """Parsed protein structure for contact evaluation."""
    name: str               # PDB ID or filename stem
    sequence: str           # Amino acid sequence
    coords: torch.Tensor    # [L, 3, 3] — N, CA, C backbone atoms
    chain_id: str | None = None

def load_structures(
    directory: str | Path,
    max_structures: int | None = None,
) -> list[StructureData]:
    """Parse all PDB/CIF files in a directory.

    Uses BioPython PDBParser / MMCIFParser.
    Skips files that fail to parse (with warning).
    """
```

Key implementation details:
- Glob for `*.pdb`, `*.cif`, `*.ent`, `*.mmcif` files.
- Use BioPython `PDBParser` (for PDB) and `MMCIFParser` (for mmCIF).
- Extract backbone N, CA, C atom coordinates. Missing atoms → NaN.
- Standard + non-standard residue mapping (MSE→M, SEC→C, etc.).
- Sort files deterministically (alphabetical) for reproducible ordering.

### 3.3 Contact metrics (`src/oplm/eval/metrics/contact.py`)

Port and adapt from `libreplm/src/procoder/eval/metrics/contact.py`. The core
functions are retained with interface adjustments for oplm's model outputs.

#### Helper functions

```python
def compute_contact_map(
    coords: torch.Tensor,
    threshold: float = 8.0,
) -> torch.Tensor:
    """Binary contact map from CA coordinates. [L, 3, 3] → [L, L]."""

def apply_apc(matrix: torch.Tensor) -> torch.Tensor:
    """Average Product Correction on a [L, L] contact probability matrix."""

def extract_attention_contacts(
    attention_weights: list[torch.Tensor],
    layer: int | str = "last",
    head_aggregation: str = "mean",
    num_layers: int = 1,
) -> torch.Tensor:
    """Extract contact predictions from attention weight matrices.

    Args:
        attention_weights: List of [H, L, L] tensors, one per layer.
            (Note: unbatched — this operates on single structures.)
        layer: "last" (average final num_layers), "mean" (all), or int index.
        head_aggregation: "mean" or "max" across heads.
        num_layers: How many final layers to average when layer="last".

    Returns:
        Symmetrized, APC-corrected contact probability matrix [L, L].
    """

def extract_per_layer_head_attention(
    attention_weights: list[torch.Tensor],
) -> torch.Tensor:
    """All layer/head attentions, symmetrized and APC-corrected.

    Returns: [n_layers, n_heads, L, L] for logistic regression features.
    """
```

#### Precision@L computation

```python
def compute_precision_at_l(
    pred_contacts: torch.Tensor,
    true_contacts: torch.Tensor,
    seq_len: int,
    min_seq_sep: int = 6,
    l_divisor: int = 1,
) -> float:
    """Compute precision@(L/divisor) for a single structure.

    Args:
        pred_contacts: [L, L] predicted contact scores.
        true_contacts: [L, L] binary ground-truth contacts.
        seq_len: Effective sequence length (excluding padding).
        min_seq_sep: Minimum residue separation for long-range contacts.
        l_divisor: Denominator for L (1 → L, 2 → L/2, 5 → L/5).

    Returns:
        Precision as a float in [0, 1].
    """
```

#### Logistic regression P@L

```python
def compute_logreg_precision_at_l(
    structures: list[StructureContactData],
    n_train: int = 20,
    n_iterations: int = 5,
    logreg_lambda: float = 0.15,
    l_divisor: int = 1,
    min_seq_sep: int = 6,
) -> float:
    """Compute precision@L using logistic regression on attention features.

    For each iteration:
        1. Seed RNG with 42 + iteration (deterministic across eval runs).
        2. Randomly split structures into train (n_train) and test (remainder).
        3. Train L1-regularized logistic regression on train structures.
        4. Evaluate P@L on each test structure.
    Average across all iterations and test structures.

    Falls back to mean-attention P@L if fewer than n_train + 1 structures.
    """
```

`StructureContactData` is a lightweight container holding pre-extracted features
for one structure:

```python
@dataclass
class StructureContactData:
    """Pre-extracted data for one structure, used in logreg P@L."""
    features: torch.Tensor   # [n_pairs, n_layers * n_heads]
    labels: torch.Tensor     # [n_pairs] binary contact labels
    seq_len: int
```

### 3.4 StructureEvalTask (`src/oplm/eval/tasks/structure.py`)

```python
@register_eval_task("structure")
class StructureEvalTask(EvalTask):
    """Evaluate contact prediction quality from attention weights.

    Data format: directory containing PDB and/or CIF files.

    Default mode is logistic regression P@L with 20 training structures
    and 5 cross-validation iterations. Falls back to mean-attention P@L
    when insufficient structures are available.
    """

    default_metrics: ClassVar[list[str]] = ["precision_at_L"]
```

#### Task-level configuration

The `StructureEvalTask` accepts additional configuration beyond the base
`EvalDatasetEntry` fields. These are passed as extra keys in the YAML/CLI
config and parsed in `__init__`:

| Key | Type | Default | Description |
|---|---|---|---|
| `contact_threshold` | float | 8.0 | Distance cutoff (Å) for contacts |
| `min_seq_sep` | int | 6 | Minimum sequence separation |
| `l_divisor` | int | 1 | L divisor (1=L, 2=L/2, 5=L/5) |
| `use_logistic_regression` | bool | True | Use logreg or mean-attention |
| `logreg_n_train` | int | 20 | Structures for logreg training |
| `logreg_n_iterations` | int | 5 | Cross-validation iterations |
| `logreg_lambda` | float | 0.15 | L1 regularization strength |
| `attention_layer` | str\|int | "last" | Which layer(s) for attention |
| `num_layers` | int\|None | None | Layers to average (None → 10% of model layers) |
| `head_aggregation` | str | "mean" | Head aggregation method |

These are communicated through extra keys on the eval dataset config entry. The
`parse_eval_configs()` function preserves any extra keys beyond `path`, `type`,
`eval_every`, and `metrics` into an `extra` dict on `EvalDatasetEntry`:

```python
@dataclass
class EvalDatasetEntry:
    name: str
    path: str
    type: str
    eval_every: int | None = None
    metrics: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # Task-specific config
```

CLI example:

```bash
data.eval.pdb.path=/data/structures \
data.eval.pdb.type=structure \
data.eval.pdb.eval_every=10000 \
data.eval.pdb.l_divisor=2 \
data.eval.pdb.contact_threshold=8.0 \
data.eval.pdb.logreg_n_train=20
```

#### Evaluation flow

1. **Load structures** — parse all PDB/CIF files in `self.path` (cached after
   first call).
2. **Tokenize** — encode each sequence with `ProteinTokenizer`.
3. **Forward pass** — run model with `need_weights=True` to get attention
   matrices. Process structures one at a time (or in small batches) to manage
   memory, since attention tensors for long sequences are large.
4. **Extract features** — for each structure, extract per-layer/head attention
   features and compute the ground-truth contact map from coordinates.
5. **Compute P@L** — run logistic regression (or mean-attention) pipeline.
6. **Distributed handling** — each rank processes a subset of structures. After
   forward passes, gather all `StructureContactData` to rank 0 for the logreg
   fitting (logreg is cheap and must see all data). Broadcast the result.

```python
def evaluate(self, model, accelerator) -> dict[str, float]:
    structures = self._load_structures()  # Cached

    # Distribute structures across ranks
    rank_structures = self._shard_structures(structures, accelerator)

    # Forward pass on this rank's structures
    contact_data = []
    for struct in rank_structures:
        tokens = self._tokenize(struct)
        with torch.no_grad():
            outputs = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                need_weights=True,
            )
        contact_data.append(self._extract_contact_data(struct, outputs))

    # Gather all contact data to all ranks
    all_contact_data = self._gather_contact_data(contact_data, accelerator)

    # Compute P@L (all ranks compute independently — deterministic)
    if self.use_logistic_regression:
        p_at_l = compute_logreg_precision_at_l(
            all_contact_data,
            n_train=self.logreg_n_train,
            n_iterations=self.logreg_n_iterations,
            logreg_lambda=self.logreg_lambda,
            l_divisor=self.l_divisor,
            min_seq_sep=self.min_seq_sep,
        )
    else:
        p_at_l = mean([
            compute_precision_at_l(
                cd.pred_contacts, cd.true_contacts, cd.seq_len,
                min_seq_sep=self.min_seq_sep, l_divisor=self.l_divisor,
            )
            for cd in all_contact_data
        ])

    return {"precision_at_L": p_at_l}
```

### 3.5 Dependencies

Structure evaluation requires BioPython and scikit-learn. These should be added
as optional dependencies:

```toml
[project.optional-dependencies]
eval = [
    "biopython>=1.83",
    "scikit-learn>=1.4",
]
```

---

## Phase 4: Labeled Dataset Stubs

Each labeled benchmark (ProteinGym, TAPE, ProteinGlue, EVEREST) gets a stub
implementation that:
1. Registers the type with the registry.
2. Defines `default_metrics` for the benchmark.
3. Has a `evaluate()` that raises `NotImplementedError` with a clear message.
4. Documents the expected data format and evaluation protocol in the docstring.

### 4.1 ProteinGym (`src/oplm/eval/tasks/proteingym.py`)

```python
@register_eval_task("proteingym")
class ProteinGymEvalTask(EvalTask):
    """Zero-shot variant effect prediction using ProteinGym.

    Evaluates the model's ability to predict the functional effects of
    protein mutations using pseudo-log-likelihood scoring (masked marginals).

    Data format: directory of CSV files from the ProteinGym substitution
    benchmark. Each CSV has columns: mutant (e.g. "A42T"), DMS_score (float),
    and the wild-type sequence in the metadata.

    Evaluation protocol:
        For each assay:
        1. Encode wild-type sequence.
        2. For each mutant, compute the log-likelihood ratio:
           score = log P(mutant_aa | context) - log P(wt_aa | context)
           using masked marginal scoring (mask each mutant position, score).
        3. Compute Spearman correlation between model scores and DMS_score.
        4. Report mean Spearman across all assays.

    Task-specific config:
        max_assays: int | None     — limit number of assays (for cost control)
        scoring: str               — "masked_marginals" (default) or "wild_type_marginals"
    """

    default_metrics: ClassVar[list[str]] = ["spearman", "ndcg"]

    def evaluate(self, model, accelerator) -> dict[str, float]:
        raise NotImplementedError(
            "ProteinGym evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
```

### 4.2 TAPE (`src/oplm/eval/tasks/tape.py`)

```python
@register_eval_task("tape")
class TapeEvalTask(EvalTask):
    """TAPE benchmark evaluation.

    Tasks: secondary structure prediction (Q3/Q8), contact prediction,
    remote homology detection, fluorescence prediction, stability prediction.

    Data format: TAPE benchmark LMDB files or converted parquet/CSV.
    See https://github.com/songlab-cal/tape for format details.

    Evaluation protocol:
        Each sub-task trains a lightweight head (linear or small MLP) on
        frozen embeddings, then evaluates on the test split.
    """

    default_metrics: ClassVar[list[str]] = [
        "ss3_accuracy", "ss8_accuracy", "contact_precision",
        "homology_accuracy", "fluorescence_spearman", "stability_spearman",
    ]

    def evaluate(self, model, accelerator) -> dict[str, float]:
        raise NotImplementedError(
            "TAPE evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
```

### 4.3 ProteinGlue (`src/oplm/eval/tasks/proteinglue.py`)

```python
@register_eval_task("proteinglue")
class ProteinGlueEvalTask(EvalTask):
    """ProteinGlue benchmark evaluation.

    Multi-task protein understanding benchmark covering fold classification,
    enzyme reaction classification, gene ontology prediction, and more.

    Data format: ProteinGlue benchmark files.
    See https://github.com/ibivu/protein-glue for format details.
    """

    default_metrics: ClassVar[list[str]] = [
        "fold_accuracy", "enzyme_accuracy", "go_fmax",
    ]

    def evaluate(self, model, accelerator) -> dict[str, float]:
        raise NotImplementedError(
            "ProteinGlue evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
```

### 4.4 EVEREST (`src/oplm/eval/tasks/everest.py`)

```python
@register_eval_task("everest")
class EverestEvalTask(EvalTask):
    """EVEREST benchmark for priority virus variant effect prediction.

    Evaluates zero-shot variant effect prediction on clinically relevant
    viral protein datasets.

    Data format: EVEREST benchmark CSV files.
    See https://github.com/debbiemarkslab/priority-viruses for details.
    """

    default_metrics: ClassVar[list[str]] = ["spearman", "auroc"]

    def evaluate(self, model, accelerator) -> dict[str, float]:
        raise NotImplementedError(
            "EVEREST evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
```

---

## Phase 5: Tests

### 5.1 Unit tests

| Test file | Covers |
|---|---|
| `tests/eval/test_config.py` | `parse_eval_configs()` — all input forms, edge cases, validation |
| `tests/eval/test_registry.py` | Registration, lookup, duplicate detection, unknown type errors |
| `tests/eval/test_evaluator.py` | Scheduling logic, metric prefixing, empty-dict fast path |
| `tests/eval/test_mlm_metrics.py` | MLM loss/accuracy/perplexity computation correctness |
| `tests/eval/test_contact.py` | Contact map, APC, precision@L, logreg pipeline |
| `tests/eval/test_sequence_task.py` | Full SequenceEvalTask with real parquet data |
| `tests/eval/test_structure_task.py` | Full StructureEvalTask with real PDB files |

### 5.2 End-to-end test

Extend `tests/test_e2e.py` to include eval:

```python
def test_training_with_eval():
    """Train a small model with sequence eval and verify metrics are produced."""
    cfg = load_config([
        "--preset", "small",
        "train.max_steps=20",
        "train.eval_every=10",
        f"data.train={TRAIN_DATA_PATH}",
        f"data.eval.test.path={EVAL_DATA_PATH}",
        "data.eval.test.type=sequence",
    ])
    trainer = Trainer(cfg)
    trainer.train()
    # Verify eval metrics were logged
```

### 5.3 Test data

- **Sequence eval**: use the existing `tests/fixtures/training/test_sequences.parquet`
  (or a small held-out split of it).
- **Structure eval**: add a small set (3-5) of PDB files to
  `tests/fixtures/eval/structures/`. These should be small proteins (<100 residues)
  to keep test runtime low.

---

## Implementation Order

### Phase 1 — Framework (~files: 6, ~lines: 350)

1. `src/oplm/eval/__init__.py`
2. `src/oplm/eval/registry.py`
3. `src/oplm/eval/tasks/__init__.py`
4. `src/oplm/eval/tasks/base.py`
5. `src/oplm/config.py` — add `EvalDatasetEntry`, extend `DataConfig`, add `parse_eval_configs()`
6. `src/oplm/eval/evaluator.py`

Tests: `tests/eval/test_config.py`, `tests/eval/test_registry.py`, `tests/eval/test_evaluator.py`

### Phase 2 — Sequence eval (~files: 4, ~lines: 250)

1. `src/oplm/eval/data/__init__.py`
2. `src/oplm/eval/data/sequence_loader.py`
3. `src/oplm/eval/metrics/mlm.py`
4. `src/oplm/eval/tasks/sequence.py`

Tests: `tests/eval/test_mlm_metrics.py`, `tests/eval/test_sequence_task.py`

### Phase 3 — Trainer integration (~lines: 50 changed)

1. `src/oplm/training/trainer.py` — add Evaluator support, replace eval scheduling
2. Update `tests/test_e2e.py` with eval integration test

### Phase 4 — Structure eval (~files: 4, ~lines: 500)

1. `src/oplm/eval/data/structure_loader.py`
2. `src/oplm/eval/metrics/__init__.py`
3. `src/oplm/eval/metrics/contact.py`
4. `src/oplm/eval/tasks/structure.py`

Tests: `tests/eval/test_contact.py`, `tests/eval/test_structure_task.py`

Update `pyproject.toml` with `eval` optional dependencies.

### Phase 5 — Labeled dataset stubs (~files: 4, ~lines: 150)

1. `src/oplm/eval/tasks/proteingym.py`
2. `src/oplm/eval/tasks/tape.py`
3. `src/oplm/eval/tasks/proteinglue.py`
4. `src/oplm/eval/tasks/everest.py`

### Phase 6 — Tests & polish (~files: 7)

1. Full test suite for all phases
2. Test fixtures (PDB files for structure eval)
3. Update `tests/test_e2e.py` with comprehensive eval integration

---

## Design Decisions & Rationale

### Why the Evaluator is called every step (not gated by should_run_eval)

The Evaluator's `__call__` is invoked every optimizer step. When no tasks are
due, it returns `{}` after `len(tasks)` modulo checks — effectively free. This
keeps the Trainer completely ignorant of eval scheduling: it just calls the
Evaluator and logs whatever comes back. The Evaluator handles `model.eval()` /
`model.train()` toggling internally, only when it actually runs something.

### Why metric override replaces rather than extends defaults

When a user specifies `metrics: [loss, accuracy]`, they get exactly those two
metrics — not those plus whatever the type's defaults are. This makes the
behavior explicit and predictable. If you want the defaults, omit the `metrics`
key entirely.

### Why eval masking is fixed at 15%

Different training runs may use different mask probabilities. Using a fixed 15%
eval masking rate ensures that eval metrics are directly comparable across runs,
regardless of training masking configuration.

### Why the DataLoader is lazily initialized

Eval data is not loaded at Trainer construction time. Instead, each `EvalTask`
loads its data on the first `evaluate()` call. This avoids unnecessary I/O
if training ends before the first eval step, and ensures the Accelerator is
fully initialized (important for distributed DataLoader preparation).

### Why structure eval gathers contact data rather than metrics

For logistic regression P@L, the regression must see structures from all ranks
to form proper train/test splits. Gathering raw `StructureContactData` (features
+ labels) to all ranks, then computing P@L independently on each rank (with
deterministic seeding), avoids the complexity of running logreg on only one rank
and broadcasting. The data volume is modest (tens of structures with compressed
feature vectors).

### Why eval tasks handle distributed logic internally

Different eval types have fundamentally different distributed patterns:
- **Sequence eval**: shard the DataLoader, accumulate tensors, reduce across ranks.
- **Structure eval**: shard structures, gather features, compute logreg on all ranks.
- **ProteinGym** (future): shard assays across ranks, gather Spearman correlations.

Pushing distributed logic into each task (rather than the Evaluator) keeps the
Evaluator simple and gives each task full control over its communication pattern.

### Why stubs raise NotImplementedError rather than returning empty dicts

A clear error is better than silent no-ops. If someone configures
`type=proteingym` and expects metrics, they should get an explicit error telling
them it's not yet implemented — not a silent empty wandb panel.
