# Structure Prediction Head — Design

**Date:** 2026-10-07
**Status:** Design agreed in discussion on 2026-10-07; completed written spec
ready for review. Implementation has not started.
**Scope:** A structure prediction head for OPLM, modeled on ESMFold2, with
state-of-the-art kernels, single-GPU training at 2048-token crops, a general
all-atom representation populated with protein-only data in v1, and a near-term
roadmap to ligands, nucleic acids, and context-parallel inference.

## 1. Intent and agreed requirements

Build a folding head that attaches to a frozen OPLM language model and predicts
all-atom 3D structure from sequence. The reference design is Biohub's ESMFold2
(May 2026): a frozen ESMC-6B feeding a pair-only recurrent trunk made of
triangle multiplication and feed-forward blocks, an all-atom EDM diffusion
module, and a confidence head. We keep that architecture and modernize the
execution: fused kernels for every hot path, a memory plan that reaches
2048-token training crops on one B200, and code written so the pair tensor can
later be sharded across GPUs without a redesign.

Agreed decisions:

- **Representation is general, data is protein-only.** Tokens, flat ragged atom
  arrays, CCD-derived reference geometry, molecule-type fields, and bond slots
  are all built on day one. v1 preprocessing keeps only protein polymer chains.
  Ligands and nucleic acids are a near-term roadmap item, not a maybe; the
  design must make them a data-pipeline change with zero model changes.
- **Multi-chain, single-sequence.** Multimers from the start. No MSA encoder in
  v1; the slot is additive later.
- **Length target (option 2).** Training crops to 2048 tokens on a single GPU
  via nested activation checkpointing and fused triangle multiplication;
  single-GPU inference to roughly 4k–6k tokens; the trunk written to the
  block-local discipline in §5.6 so Fold-CP-style context parallelism can be
  added later without rewriting features. Context parallelism itself is
  deferred.
- **Success criteria.** Milestone 1 (pipeline proof): a 24-layer trunk trained
  from an existing OPLM checkpoint, judged on monomer TM-score against
  held-out CAMEO and CASP targets. v1: match ESMFold2-Fast on FoldBench
  protein-protein and antibody-antigen DockQ success rate, single-sequence,
  using the phase-2 (2048-context) OPLM checkpoint, as measured in our own
  harness against a same-harness baseline run.
- **Approach.** Port the ESMFold2 model modules (Apache 2.0) into OPLM
  conventions; write the training side (losses, data, trainer adapter) fresh in
  OPLM style; validate the port against the released ESMFold2 weights and the
  losses against Boltz-2's implementations.
- **Base LM.** The phase-1 scaling runs train at 512 positions, but phase 2 of
  pretraining uses a 2048 maximum length, so the final LM is comparable to
  ESMC in context. The head attaches to any OPLM checkpoint; the shim reads
  layer count and width from the LM config.
- **Open throughout.** Publish all training, preprocessing, evaluation, and
  inference code on GitHub; publish the training data, final weights, and
  useful intermediate checkpoints on Hugging Face. Preserve phase and stage
  boundaries so others can vary the data mix, learning-rate decay, and
  post-training recipe. Prefer permissive licenses and redistributable inputs
  throughout, with artifact-level provenance and license records (§11).

## 2. What is kept from ESMFold2 and what changes

Kept, with defaults matching ESMFold2 so its released weights load for parity:

- Pair-only trunk: no single track, no triangle attention. Each block is an
  outgoing triangle multiplication, an incoming one, and a SwiGLU transition.
- LM features via a learned softmax mix over all layers, projected and expanded
  to pair space by an outer product and difference MLP, refined by four pair
  blocks every loop, with heavy dropout and input-residue masking as
  regularizers.
- Contractive recurrence: per-channel learned decay in (0, 1) times the pair
  state plus a learned injection of the normalized inputs, then the shared
  trunk. Loop count sampled from a clamped Poisson at train time, gradient
  through the last two loops, test-time scaling via more loops.
- EDM atom-level diffusion with a sliding-window atom encoder/decoder, 3D RoPE,
  a token-level DiT with adaptive LayerNorm and pair-biased gated attention, a
  truncated inference schedule, and a separate confidence head with its own pair
  trunk.
- Three-stage crop curriculum and the 30/70 PDB/distillation mix.

Changed:

- OPLM replaces ESMC. Chains run through the LM as separate batch rows.
- cuEquivariance's fused triangle multiplication for inference and training,
  with a fused-forward/reference-backward path where only inference kernels
  exist for our shape. Biohub's inference-only Triton fusions are not ported.
- FlexAttention for pair-biased DiT attention and sliding-window atom
  attention, replacing dense-bias SDPA and the FlashAttention dependency.
- Nested (loop-level plus block-level) activation checkpointing to reach
  2048-token crops.
- Pair width is a measured choice (128 vs 256), not an inherited default.
- All pair features are block-local by construction (§5.6).
- Additional training sources: AFDB predicted dimers, and antibody-antigen
  sequence-pair distillation with a pluggable ranker (§7.5). An
  interface-weighted coordinate loss as a per-stage knob (§6.2).
- Stages are separate Trainer runs chained with `init_from`; EMA weights are
  exported alongside each checkpoint.

## 3. Package layout and boundaries

All fold-specific code lives in `src/oplm/fold/`, mirroring how `oplm/model/`
is organized: one public modeling file for HF remote-code compatibility,
building blocks in small focused modules.

```
src/oplm/fold/
├── configuration_fold.py   # FoldConfig (PretrainedConfig)
├── modeling_fold.py        # public classes only: OplmFoldPreTrainedModel, OplmForFolding
├── lm_shim.py              # per-layer LN + proj, softmax layer mix, outer product/difference -> pair
├── pair.py                 # block-local pair features: relpos, token bonds, pair mask, pair init
├── trimul.py               # staged trimul (pre / contraction / post) + cuEquivariance dispatch
├── trunk.py                # pair update block, folding trunk, contractive recurrence
├── atoms.py                # atom encoder / decoder: sliding-window attention + 3D RoPE
├── diffusion.py            # DiT with adaLN + pair-bias attention, EDM schedule, sampler
├── confidence.py           # confidence head + distogram head
├── losses.py               # diffusion MSE w/ weighted rigid align, smooth LDDT, distogram, confidence, bond
├── data/
│   ├── ccd.py              # CCD load, reference conformers, element / charge tables
│   ├── mmcif.py            # mmCIF -> MolecularComplex (flat atoms, tokens, chains, entities)
│   ├── featurize.py        # MolecularComplex -> model inputs
│   ├── crop.py             # contiguous / spatial / interface crops
│   ├── dataset.py          # preprocessing to shards + StructureDataset (Trainer resume contract)
│   └── collate.py          # ragged atom batching, pad-to-multiple
├── eval/
│   ├── metrics.py          # TM-score, lDDT, DockQ
│   └── tasks.py            # fold_monomer / fold_complex tasks, registered in oplm.eval.registry
├── training.py             # FoldTask: model factory, dataloader, step fn, flops; stage knobs
├── predict.py              # fold(seqs) -> structures, mmCIF writer
├── distill.py              # predict + ranking hook -> distilled training source (milestone 4)
└── cli.py                  # oplm fold train | predict | preprocess | eval | distill | bench-kernels
tests/fold/                 # mirrors the above
docs/FOLD.md                # architecture contract, like MODEL_ARCHITECTURE.md
THIRD_PARTY_NOTICES.md      # Apache 2.0 attribution for ported Biohub modules
```

Rules:

- **Dependency direction.** `oplm.fold` imports from `oplm.model`,
  `oplm.training`, `oplm.data`, and `oplm.eval`. Core packages import nothing
  from `oplm.fold` except the CLI registration and eval task discovery. The
  existing backbone-only structure loader and the categorical-Jacobian eval are
  untouched.
- **The LM is a frozen input**, held by the fold model as an unregistered
  attribute (§6.6).
- **Core integration stays narrow:** a `TrainTask` protocol in the Trainer
  (§6.4), EMA/checkpoint lifecycle support (§6.5), config and CLI registration,
  and generalizing the eval model type. The default task reproduces current
  MLM behaviour exactly; there is no second training loop.
- **Config.** One `fold:` block in the top-level run config carrying the LM
  checkpoint path and the `FoldConfig` fields. `train` and `slurm` are reused
  as is. Structure datasets appear under `data.train` with new entry types.
- **New dependencies, all optional extras.** `fold`: `cuequivariance-torch`
  plus `cuequivariance-ops-torch-cu13` (reference path used when absent),
  `gemmi` (mmCIF and CCD parsing). `eval`: `DockQ`. External binary for v1
  clustering: MMseqs2. FlexAttention and SDPA use the existing torch dependency;
  the supported CUDA/package combination is pinned after the kernel benchmark.
- **Attribution.** Ported modules carry a header naming the Biohub source file
  and stating modifications; `THIRD_PARTY_NOTICES.md` carries the Apache 2.0
  text. Apache 2.0 code is compatible with OPLM's MIT license.

## 4. Representation contract

### 4.1 Tokenization

AF3 and ESMFold2 rules, exactly. One token per standard amino acid or
nucleotide; one token per atom for ligands and for residues with no standard
parent. The token-type vocabulary is AF3's 32-class set from day one.
Non-standard residues that the CCD maps to a parent (`MSE` to `MET`, etc.)
become the parent token with the parent's atom set. The LM always sees the
canonical 20-letter alphabet plus `X`.

### 4.2 `MolecularComplex`

Flat arrays, never a per-residue layout.

| Level | Fields |
|---|---|
| atoms | coordinates (fp32), resolved mask, element (atomic number), atom name, formal charge, atom-to-token index, B-factor or pLDDT |
| tokens | token type, molecule type, chain index (asym id), entity id, symmetry copy id, residue index, center atom index, representative atom index, atom range (start, count) |
| bonds | atom index pairs for covalent bonds between tokens; empty in v1, always present |
| metadata | id, source, resolution, release date, method, dropped-entity flags, per-chain mean pLDDT |

Molecule type takes four values: protein, DNA, RNA, ligand. It is carried on
every token and keys the per-atom loss weights and the LM scatter mask. In v1
only protein appears. Center atom is CA for protein, C1' for nucleotides, the
atom itself for atom tokens. Representative atom (for distograms and PDE) is
CB (CA for glycine) for protein, C4 for nucleotides, the atom itself for atom
tokens.

### 4.3 Reference geometry from the CCD

The wwPDB Chemical Component Dictionary is parsed once with gemmi into a small
cache keyed by CCD code: atom names, elements, charges, ideal coordinates,
leaving-atom flags, parent code. The twenty standard amino acids come from the
same table as everything else; ligands later are more cache entries. Per-atom
model features are AF3's 389-dimensional set, which ESMFold2 also uses: ideal
position (3), mask (1), element one-hot (128), charge (1), four-character
atom-name encoding (256). Reference conformers are randomly rotated and
centered per token at featurization time, with a per-token space id so the
model knows which atoms share a frame.

### 4.4 v1 preprocessing filters

Applied at parse time and recorded in metadata:

- Keep protein polymer chains from the asymmetric unit. Drop waters, ions,
  ligands, and nucleic acid chains, flagging that they were dropped so entries
  can be reprocessed later.
- Token sequence comes from the entity polymer sequence, so unresolved residues
  exist as tokens with masked coordinates and the LM sees the full chain.
- Highest-occupancy altloc, first model, hydrogens dropped.
- Resolution 9 Å or better; release date before the training cutoff
  (2021-09-30 by default); drop chains with fewer than 4 resolved residues.
- Bioassembly expansion is deferred.

### 4.5 LM input contract

Each protein chain runs through the LM as its own batch row with BOS and EOS,
padded to the longest chain. This matches ESMFold2's block-diagonal chain
masking without touching OPLM's attention mask. Hidden states from every layer
(embedding output plus one per block) are projected by the shim and scattered
onto protein tokens by chain and residue index. Non-protein tokens receive
zeros. Residues collapsed from modified residues map to the parent letter.

### 4.6 Block-local pair features

Every pair feature is a function of a row index tensor and a column index
tensor returning the corresponding block; the full tensor is the all-by-all
call. No hidden `arange(L)` anywhere. The set:

- relative position encoding: AF3's same-chain, same-entity, same-residue
  flags, clipped residue-index difference (r_max 32), token-index difference,
  clipped chain difference (s_max 2);
- token bonds (zero in v1);
- pair mask (outer product of token masks);
- LM pair initialization (outer product and difference of per-token vectors);
- single-to-pair outer sum;
- distance-bin embedding of predicted coordinates for the confidence head.

A single test asserts, for random row and column subsets, that the block call
equals the sliced full call for every function in this set (§9).

### 4.7 Batching and storage

Tokens pad to the crop size; atoms pad to `atoms_per_token_budget` (default
24, enough for standard nucleotides) times the crop size. This is a total
batching budget, not a fixed atom layout. Reject or recrop an over-budget
example before collation; never truncate its atom arrays. Keep token padding,
atom padding, reference-geometry validity, and experimentally resolved masks
distinct. Unresolved atoms are valid prediction slots but have no coordinate
supervision.

Crop sizes are fixed within a stage and multiples of 128. Atom padding also
respects the attention kernel's block multiple. This lets the existing static
bucket compile recipe apply; diffusion sample count and microbatching are
explicit stage settings rather than accidental dynamic dimensions.

Parsed complexes are stored as PyArrow shards, one row per entry with ragged
list columns, plus a small metadata Parquet index for filtering and sampling.
Cropping and featurization happen at load time so epochs see fresh crops.
Shard manifests record the schema, CCD version, input checksums, preprocessing
configuration, source licenses, and split membership. These same artifacts
are the publishable training datasets (§11).

## 5. Model, kernels, and memory

### 5.1 Forward data flow and component defaults

The frozen LM supplies per-chain hidden states. The trainable shim builds LM
pair features, while the inputs embedder maps reference atom features to token
inputs and initial pair features. A shared recurrent trunk refines the pair
state. The distogram head reads that state; two coda blocks prepare diffusion
conditioning. The atom denoiser predicts coordinates, and the confidence head
scores a selected sample.

The following are the discussion's starting settings. The checkpoint-parity
configuration must reproduce the pinned upstream checkpoint exactly; training
experiments may override these values explicitly.

| Component | Design | Starting settings |
|---|---|---|
| LM shim | Per-layer LayerNorm and projection, softmax layer mixture, outer product/difference MLP and LayerNorm | Pair width 256; pair dropout 0.25; input residue masking 10% |
| LM pair encoder | Pair update blocks, rerun with fresh dropout each recurrence loop | 4 blocks |
| Inputs embedder | Sliding-window atom attention and 3D RoPE over reference conformers, aggregated to tokens; zero MSA/deletion slots retained for parity | Atom width 128; 3 blocks; window 128 |
| Pair initialization | Outer sum of token inputs, relative-position and token-bond embeddings | AF3 relpos with `r_max=32`, `s_max=2` |
| Recurrence | Learned per-channel decay and input injection, then the shared trunk | Poisson mean 3, clamped to 1–6 loops; gradients through last 2; inference default 10 |
| Trunk | Outgoing trimul, incoming trimul, SwiGLU transition; row-shared residual dropout | Width 128 or 256; transition expansion 4; 24 blocks for pipeline proof, 48 for v1 |
| Coda | Pair update blocks for diffusion conditioning | 2 blocks |
| Diffusion | EDM; Fourier noise embedding, adaLN token DiT, pair-biased gated attention, atom encoder/decoder | `sigma_data=16`; log-normal parameters −1.2 and 1.5; 68 inference steps; token width 768; 12 blocks; 16 heads |
| Confidence | Predicted-distance pair embedding, row-attention pooling, separate pair trunk; atom pLDDT/resolved and token-pair PAE/PDE heads | 4 blocks; 50 pLDDT bins; 64 PAE/PDE bins |
| Distogram | Linear projection of the symmetrized pair state | 128 bins |

The recurrence starts from a truncated-normal pair state. Each loop combines
the previous state, multiplied by a learned channelwise decay in `(0, 1)`,
with a learned injection of normalized initial and refined LM pair features.
The decay makes the carry term contractive; it does not prove that the entire
nonlinear trunk is a contraction. In training, detach the early loops and
retain gradients only through the last `min(2, n_loops)` loops.

Reference conformer rotations, LM masking, loop count, dropout, and diffusion
noise use recorded RNG state. Checkpoint recomputation must replay the same
random choices. Inference disables masking and dropout and exposes loop,
step, sample, and seed settings.

### 5.2 Triangle multiplication dispatch

Keep three explicit stages in the reference implementation: local projections,
normalization and gates; triangular contraction; local normalization,
projection and output gate. Chunk contraction output rows when necessary and
accumulate in fp32 without materializing both full operands in fp32. Incoming
and outgoing operations share the same parameter conventions as the port.

Dispatch depends on the actual device, dtype, shape, layout, and gradient
requirement:

1. Use cuEquivariance autograd when the installed backend supports that
   training configuration.
2. If only a fused inference kernel is supported, use fused forward with
   recomputation through the differentiable reference for backward. Gradients
   must reach both the input and every trainable projection, norm, and gate.
3. Use the compiled reference when the library or a compatible kernel is
   absent. CPU tests always have this path.

This is an explicit mixed execution path, not an assumption that the library
automatically combines a fused forward with our backward. Implement it at one
recomputation boundary and test its interaction with nested checkpointing;
avoid accidentally retaining reference activations or recomputing the same
block twice beyond the chosen checkpoint scheme. Early no-gradient recurrence
loops can use supported inference kernels directly.

Do not infer triangle-multiplication support from triangle-attention limits.
Record support and timings for widths 128 and 256 on the target B200 and pin
the tested package versions. The design does not depend on future 256-wide
Blackwell backward kernels. NVIDIA documents the operation and evolving
coverage in its [API reference](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html)
and [release notes](https://docs.nvidia.com/cuda/cuequivariance/changelog.html).

`oplm fold bench-kernels` measures forward, backward, checkpointed execution,
and peak allocated/reserved memory at lengths 384, 768, 1024, 1536, and 2048,
for both widths and both contraction directions. Warm up compilation, synchronize
timing, and report hardware, versions, actual dispatch path, and numerical
error. Width selection uses these measurements plus pipeline-proof accuracy.
Halving pair width halves pair-state storage; it does not double the supported
length because storage grows quadratically with length.

### 5.3 Attention and compilation

Use FlexAttention for pair-biased token attention in the diffusion transformer
and for sliding-window atom attention. Read pair bias inside the attention
kernel; use block masks for padding and actual atom-window sparsity. Preserve
the upstream window indexing and reference-space semantics, not merely a
similar-looking local mask. Verify gradients into the pair-bias projection as
well as Q, K, and V.

Retain a dense PyTorch attention formulation as the small-input oracle and
unsupported-device path. Fully masked padded queries must produce finite,
masked outputs. `torch.compile` handles pointwise work; no Biohub
inference-only Triton fusions or new FlashAttention dependency are needed.
Pair-bias caching across diffusion steps is deferred until profiling justifies
the additional live memory.

### 5.4 Precision and initialization

Run the frozen LM in bf16 under `no_grad`, with gradients enabled for the shim.
Using `no_grad` avoids passing inference-only tensors into trainable layers
that need to save their inputs for backward. Keep the LM in eval mode even
when the fold head enters training mode.

The pair stream is bf16, contractions accumulate in fp32, and normalization
uses fp32 internals. Coordinates, rigid alignment, diffusion computation, and
loss/logit calculations remain fp32; disable outer autocast locally where
needed. Fast attention paths must support these intended dtypes rather than
silently changing the diffusion precision policy.

Reuse OPLM's norm utilities and weight-initialization tags where compatible.
Preserve the reference port's residual-output initialization and gate biases,
including its zero-initialized residual projections and −2 gate biases where
specified. A generic HF initialization pass must not overwrite these choices.

### 5.5 Memory budget and length acceptance

For batch one, a bf16 pair tensor uses `2 * L * L * c_pair` bytes:

| Tokens | Width 128 | Width 256 |
|---|---|---|
| 1024 | 0.25 GiB | 0.5 GiB |
| 2048 | 1 GiB | 2 GiB |
| 4096 | 4 GiB | 8 GiB |
| 6144 | 9 GiB | 18 GiB |

Checkpoint both retained recurrence loops and individual blocks, including
the LM pair encoder, coda, diffusion, and confidence blocks. The intent is to
recompute one loop's block boundaries at a time rather than retain both
loops' entire stacks. At width 256 and length 2048, 48 block-boundary pair
tensors alone represent 96 GiB; the discussion's roughly 140 GiB activation
estimate is a planning estimate, not a measured total.

The measured peak must also include the frozen LM and its all-layer hidden
states, trainable parameters and gradients, AdamW state, EMA, masks, logits,
checkpoint inputs, temporary contraction buffers, compiler workspaces, and
allocator headroom. A nominal GPU capacity is not interchangeable with usable
GiB. Diffusion noise samples can be microbatched over shared trunk conditioning
to avoid multiplying every activation by the sample count.

Acceptance is a complete 2048-token forward/backward/optimizer step on one
B200, with the selected width and recorded stage settings, followed by enough
steady-state steps to expose allocator or compilation growth. A 1536-token
training run alone does not satisfy the 2048 target. If needed, measure width
128, fewer gradient-bearing loops, or offloaded loop inputs; record any such
tradeoff in the run config and result.

The 4k–6k single-GPU inference range is likewise a target to benchmark. At
6144 tokens, ten width-256 pair equivalents already consume 180 GiB before
other allocations. Report actual maximum lengths for each width and sampling
configuration. Total complex length and individual LM chain length are separate
limits: independent chains may collectively exceed the LM context, while a
single over-context chain requires an explicit extension policy and quality
validation. v1 must fail clearly rather than silently truncate that chain.

### 5.6 Discipline for later Fold-CP inference

All pair features follow §4.6, and trimul keeps the staged boundary from §5.2.
Pointwise operations must not assume that local pair axes span the entire
complex. Pair symmetrization is an explicit operation so it can become a
mirror-rank block exchange later.

The later distributed path adds row/column operand communication for triangle
contraction, pair-transpose exchange, and distributed pair-biased diffusion
attention through bias gathering or a ring algorithm. It cannot simply call
the unsharded fused kernel on each block. Start with square GPU grids as in
Fold-CP; rectangular grids and context-parallel training are separate work.
Single-GPU block-feature tests protect these boundaries now without introducing
distributed machinery into v1.

### 5.7 Public model and prediction surface

`FoldConfig`, `OplmFoldPreTrainedModel`, and `OplmForFolding` are the public HF
classes. Training consumes a featurized batch and returns a differentiable
total loss, detached per-term metrics, and requested structure/confidence
outputs. Prediction consumes sequences, chain ids/copy counts, and sampling
settings; returns flat atom coordinates with their mapping and confidence
outputs; and can write mmCIF with pLDDT in the B-factor field.

Derived pTM and ipTM come from PAE. Rank monomer samples by pTM and complex
samples by ipTM, recording all scores and the chosen sample. The ESMFold2
weight-name remap is a parity-test utility, not a second public model API.

## 6. Losses, training stages, and Trainer integration

### 6.1 One training step

1. Crop and featurize a structure; create independent chain LM rows and mask
   10% of valid protein residues. Run the frozen LM, then the trainable shim.
2. Sample recurrence count, detach early loops, and checkpoint the last two.
   Refresh LM-pair dropout on each loop.
3. Compute distogram logits from the symmetrized final trunk pair.
4. Run the coda. For each configured diffusion noise sample, augment and center
   the reference structure, draw log-normal noise, and evaluate the denoiser.
   Share trunk conditioning and normalize over samples so microbatching does
   not change the objective.
5. Produce one confidence-training structure with a short no-gradient sampling
   rollout, initially 20 steps. Stop gradients through coordinates and trunk
   inputs to the confidence head; confidence loss trains only that head.
6. Return the weighted loss, per-term metrics, valid token count, and structure
   count. Diffusion replicas do not multiply training token/sample accounting.

### 6.2 Loss definitions and interface weighting

Use AF3-style definitions with the agreed starting weights, checked against
pinned Boltz-2 implementations for the corresponding mathematical terms.

| Term | Definition | Starting weight / stages |
|---|---|---|
| Diffusion MSE | Coordinate error after weighted rigid alignment of reference onto prediction; EDM noise weighting | 4.0 |
| Smooth lDDT | Alignment-free local-distance objective | 1.0 in stage 1; off subsequently |
| Bond | Squared error in inter-token bond lengths | Available from stage 2; zero in protein-only v1 with its empty explicit bond list |
| Distogram | Cross-entropy of representative-atom distance bins | 0.03 |
| Confidence | Atom pLDDT/resolved, token-frame PAE, representative-atom PDE cross-entropies | 1e-4 multiplying the reference-style composite |

Coordinate atom weights are 1 for protein, 5 for nucleotides, and 10 for
ligands. Apply resolved and padding masks before alignment and reduction.
Reference targets and alignments are detached. Invalid frames, empty valid
sets, and degenerate alignment cases need explicit finite behavior. Record
the precise bin edges, reduction conventions, and confidence subterm weights
in the config/parity fixtures instead of relying on prose to reconstruct them.

Multimer supervision must handle interchangeable chains and symmetry-equivalent
atom naming. Establish a consistent target assignment before coordinate and
confidence-target calculations, using the reference training convention.
Otherwise correct homodimer predictions can receive arbitrary penalties.
The empty v1 bond feature denotes omitted explicit bond supervision, not an
absence of peptide bonds in the molecule.

Add `interface_mse_weight` as a per-stage setting. An interface atom is a valid
reference atom within 10 Å of a valid atom on another chain. Compute the mask
from the full reference before cropping and retain it for selected atoms;
use chunked distances so featurization does not allocate a dense all-atom
distance matrix. For distillates, the accepted predicted structure is the
reference.

Align with the ordinary molecule-type weights over all valid atoms, then add
an interface-only mean of the same aligned squared errors:

```text
coordinate_loss = EDM_weight(sigma) * (
    global_weighted_MSE + interface_mse_weight * interface_weighted_MSE
)
```

The interface term has its own masked normalization and is zero when there
are no interface atoms. Interface weighting does not change the alignment.
This follows the motivation and global-alignment convention in the
[TorchFold paper](https://torchx-cpl.github.io/assets/papers/torchfold.pdf).
Its reported fine-tuning changes combine data, loss, and schedule changes, so
they do not isolate the benefit of this term.

Default the weight to zero in stage 1. Compare zero against a small positive
weight during general multimer training; reserve larger values for Ab–Ag
fine-tuning. Log global and interface errors separately. Per-source noise
parameters and an optional stage-local schedule are configuration choices;
do not import another model's noise-parameter values without matching its
parameterization.

### 6.3 Chained training recipe

Use separate Trainer runs joined through `train.init_from` and Slurm job
dependencies. A stage starts with fresh optimizer/scheduler state; a resume
restores that stage's complete state. The following is the initial experiment
recipe, not a claim that these settings have already reproduced ESMFold2.

| Stage | Crop | Initial step budget | Trainable components and purpose |
|---|---|---|---|
| 1 | 384 | 50k | Entire fold head; smooth lDDT on; initial global batch 128 |
| 2 | 768 | 10k | Entire fold head; introduce larger contexts and multimer loss ablation |
| 3 | 1536 | 5k | Entire fold head; lower LR, longer complexes |
| 4 | 1536 | 5k | Load stage-3 EMA; freeze the trunk/conditioning pathway, train diffusion and confidence |
| 5 | Fixed per run, chosen from measured feasible crops | Set from accepted data volume | Ab–Ag fine-tuning on experimental structures plus accepted distillates; unfreeze the folding pathway |

The agreed curriculum uses 1536 in stages 3–4, while the implementation target
is 2048. Demonstrate 2048 training separately under §5.5, then use that crop in
a recorded stage variant if the throughput/accuracy tradeoff supports it.

Start with 30% experimental PDB and 70% monomer distillation. For v1, split
the predicted-data allocation between monomers and AFDB dimers as an explicit
run-config choice; preserve source-specific metrics and sampling controls.
Stage 5 has its own experimental/distilled Ab–Ag mix.

Use AdamW, warmup, and existing scheduler options with μP disabled for the
head. Use DDP and the existing gradient accumulation controls for global
batch size. The frozen LM is replicated on each device; HSDP and LM fp8 are
deferred. Stage-4 freezing must include the input embedder, shim, LM pair
encoder, recurrence/trunk, coda, and distogram pathway so diffusion and
confidence train against fixed conditioning. The LM stays frozen in all stages.

### 6.4 Trainer task seam and dataset resume

Introduce a small `TrainTask` protocol that builds the model, builds the
dataloader, executes a step returning `StepResult`, and supplies a FLOP estimate
or `None`. The default MLM implementation preserves existing behavior. Fold
code provides `FoldTask`; the Trainer retains acceleration, optimization,
gradient accumulation, evaluation cadence, fault tolerance, and checkpointing.

`StepResult` carries the differentiable loss, valid token count, sample count,
and detached scalar metrics. Extend logging to include those metrics. If FLOPs
are unknown, omit FLOP/MFU estimates rather than reuse the LM formula: folding
cost depends on crop length, recurrence loops, and diffusion samples.

The dataset supports `__len__`/`total_length`, `set_epoch`, `set_resume_skip`,
and `clear_resume_skip`, with the same rank/worker striping as the sequence
pipeline. Reuse `InterleavedDataset` for source fractions. Its current exact
resume path also calls a source's `_arm_sample_skip`; `StructureDataset` must
support that contract or the shared seam must be generalized narrowly. Merely
providing the public batch-skip methods is insufficient for mixed-source resume.

Derive crop/augmentation randomness from the recorded seed, epoch, source,
and sample occurrence so skipping consumed examples restores the next example
and its features. Test standalone and mixed-source resume with multiple worker
counts under the existing supported resume-topology rules.

Top-level config gains `fold:` and structure dataset entry types. Generalize
the eval task model annotation to the common module interface and register
fold tasks lazily so importing normal MLM tooling does not require fold extras.

### 6.5 EMA and checkpoint lifecycle

Add EMA with decay 0.999 using PyTorch's averaged-model utility. Update once
after a successful optimizer step, never on accumulation microsteps or a
skipped update. It covers the fold head only. Export ordinary and EMA HF
directories together at each checkpoint; later stages explicitly select the
intended directory.

The current callbacks are observation-oriented and have no optimizer-step or
state-restore event. Add the minimum lifecycle support needed for EMA, with
consistent behavior on every DDP rank. Save EMA tensors and update count as
checkpoint tensor state or a sidecar covered by the atomic checkpoint commit
and remote mirror. The existing `extra_state` is JSON metadata and cannot
store the tensor state itself; it can identify the EMA artifact and settings.
Resume must restore EMA as well as model, optimizer, scheduler, RNG, and cursor.

Public stage-boundary artifacts include both ordinary and EMA weights and the
state needed to continue training (§11). Checkpoint retention must preserve
those boundaries rather than deleting them as ordinary rolling checkpoints.

### 6.6 Frozen LM ownership and export

Hold the frozen `OplmModel` outside the fold model's registered module tree.
Normal assignment of an `nn.Module` attribute registers it, so the ownership
mechanism must explicitly avoid registration. Verify exclusion from
`named_parameters`, `state_dict`, DDP synchronization, EMA, and HF exports.

Load/move the LM on the process's actual device and keep it in eval mode.
Because an unregistered module does not follow `.to()` or `.train()` on the
head automatically, handle device changes and mode explicitly. `FoldConfig`
records an immutable LM revision and compatible shape metadata, with a local
path or HF identifier and a documented override on load. A published checkpoint
must resolve its LM without relying on a private filesystem path.

## 7. Data acquisition, preprocessing, and distillation

### 7.1 Sources and reproducible acquisition

No new folding datasets were present at the time of the discussion; acquisition
is part of the work. Each download has a versioned manifest with URLs, dates,
checksums, licenses, and expected contents. Partial downloads are resumable and
completed files are verified before preprocessing.

| Source | Use and acquisition |
|---|---|
| wwPDB mmCIF archive and CCD | Experimental structures and reference geometry; versioned archive snapshot and components dictionary |
| OpenFold3 monomer distillation | Filtered subset of publicly distributed predicted structures; inspect metadata before selecting size and confidence thresholds |
| AFDB monomer representatives | Alternative distillation source if the OpenFold3 subset is unsuitable |
| AFDB predicted dimers | Additional v1 multimer source, with confidence filters and cluster-aware homo/heterodimer sampling |
| CAMEO and CASP | Monomer evaluation; acquire post-cutoff CAMEO and CASP15, verify the existing CASP14 configuration/data |
| FoldBench | Fixed protein-protein and antibody-antigen evaluation targets and interface definitions |
| SAbDab | Held-out Ab–Ag validation; separately curated, disjoint training subset for stage 5 |
| Antibody–antigen sequence pairs | Known-binder pairs for milestone-4 structural distillation, subject to provenance and redistribution requirements |

The [OpenFold3 AWS registry](https://registry.opendata.aws/openfold3/) provides
public access to long and short MGnify monomer distillation sets and lists
CC BY 4.0 terms. These are not simply an AFDB snapshot: preserve their actual
source and teacher metadata. Download the selected structures rather than
unneeded MSA archives.

### 7.2 Preprocessing and splits

`oplm fold preprocess` parses files in parallel, applies §4.4, writes shards
and the metadata index, and reports accepted/rejected counts by reason. Index
fields include chain sequences and lengths, source, release date, resolution,
per-chain mean pLDDT, clusters, and dropped-entity flags. Low-confidence atoms
in predicted structures are masked out of coordinate supervision; residues
remain in the sequence. Pin confidence thresholds in the source manifest.

The default experimental training cutoff is 2021-09-30. Use post-cutoff
structures for validation, with SAbDab resolution at most 2.5 Å. A date cutoff
does not remove sequence homologs or teacher-training overlap: build and
publish explicit target exclusions across every training source, including
predicted structures. Keep validation and final benchmark manifests distinct.

Cluster protein chains at 40% identity using MMseqs2 for v1 sampling weights;
record coverage and clustering settings. Pipeline proof may sample uniformly.
Apply filtering before fitting sampling weights. Do not silently drop targets
that the model fails to predict or parse; report coverage and exclusions.

### 7.3 Runtime crops and featurization

Sample an entry, then a contiguous, spatial, or spatial-interface crop using
source-specific probabilities recorded in config. Interface crops retain
tokens from at least two interacting chains. Enforce the atom budget, preserve
original residue/chain identities, and remap all atom and token indices.

Build reference geometry from the CCD cache with fresh per-token rotations,
pair-feature indices, resolved-coordinate targets, and loss masks. Center on
valid atoms. Prediction uses the same token/reference featurizer without
ground-truth coordinates; target-derived interface masks never become model
input features. Write predictions through gemmi with chain ids and atom names
preserved.

Asymmetric-unit-only parsing is the agreed v1 boundary. Record when this
excludes an experimental biological interface; do not claim coverage of
assemblies that were never supplied. Keep benchmark assembly definitions intact
when loading the provided evaluation complexes.

### 7.4 AFDB dimers

Include homo- and heterodimer predictions from the
[AFDB quaternary-structure release](https://www.biorxiv.org/content/10.64898/2026.03.27.714458v2)
as a separate source. Preserve source confidence and cluster metadata, filter
for reliable interfaces, and inspect the downloaded release's actual counts.
The discussion identified homodimer dominance and strong cluster redundancy;
use cluster-aware sampling and an explicit heterodimer weight rather than
uniform sampling of all files.

Keep teacher confidence separate from experimentally measured accuracy.
Evaluate this source and the interface loss as separate ablations where
practical, with protein-protein DockQ as the main outcome.

### 7.5 Antibody–antigen structural distillation

Milestone 4 adds `oplm fold distill`: curated sequence pairs, teacher prediction,
ranking/acceptance, and conversion to the same structure shards. Teacher choice
is configurable and may be an external model or a later OPLM checkpoint;
self-distillation is not required. External teachers can supply predictions
through the shared structure format without embedding another model stack in
the OPLM training environment.

Provide a scoring callable that receives candidate structures, sequences, and
available confidence outputs and returns ranking scores and an acceptance
decision. This is where the user's Ab–Ag ranking method plugs in; acceptance
must not be hard-wired to ipTM. Calibrate thresholds on a separate validation
set and freeze them before final benchmark evaluation.

Each accepted label records sequence-pair provenance and binding evidence,
teacher checkpoint/version, prediction settings and seed, ranker version,
score, threshold, round, and applicable licenses. Retain rejection statistics
and distinguish experimentally supported binders from designed candidates.
Optional later rounds can re-predict rejected or ambiguous pairs with an
improved teacher; every round is a separately versioned dataset.

Exclude evaluation antigens at sequence level and filter antibody CDR-H3
identity against FoldBench-AB and held-out SAbDab. Publish identity/coverage
thresholds, deduplication rules, and exclusion manifests before training; they
must not be tuned on final test outcomes. Fine-tune in chained stage 5 and
report both Ab–Ag gains and general-folding retention.

ASD was proposed as an input source, but its
[published terms](https://naturalantibody.com/asd/) are CC BY-NC 4.0. Under the
openness requirement, it is not an unconditional dependency of the permissive
release: use suitably permitted source collections, obtain compatible rights,
or keep an explicitly restricted experiment separate. A teacher's permissive
code license does not by itself establish rights for every input dataset or
derived artifact. The release must identify the actual terms of each source.

## 8. Evaluation and success criteria

Register `fold_monomer` and `fold_complex` in the existing eval registry.
Run a small fixed validation subset at training cadence and full datasets
offline with `oplm fold eval`. Preserve the existing backbone-only and
categorical-Jacobian tasks.

| Task | Targets | Reported metrics |
|---|---|---|
| `fold_monomer` | CAMEO, CASP14, CASP15 | TM-score, CA and all-atom lDDT, mean pLDDT, pTM |
| `fold_complex` | FoldBench protein-protein and Ab–Ag; disjoint SAbDab validation | DockQ, success fraction at DockQ ≥ 0.23, ipTM |

Implement TM-score and lDDT with explicit residue correspondence, masks, and
normalization conventions. TM-score still requires iterative superposition;
a single RMSD-optimal fit is not equivalent. Use the DockQ package for interface
definitions and chain-mapping permutations and match the benchmark's aggregation
rules. Record per-target/per-interface values as well as aggregate metrics.

Run released ESMFold2 and ESMFold2-Fast through the same target manifests,
metric versions, and exclusion rules before comparison. Keep the baseline in
its own pinned environment. Record single-sequence mode, loops, diffusion
steps, samples, ranking rule, seeds, and wall time for every method. Compare
ranked predictions; an oracle best-of-samples result is a separately labeled
diagnostic.

Pipeline proof succeeds when the 24-layer model improves held-out monomer
TM-score across checkpoints and completes the training/resume workflow. v1
targets matching ESMFold2-Fast on both FoldBench subsets with the phase-2 LM.
Report paired target-level differences and bootstrap confidence intervals;
fix the operational equivalence margin in the evaluation protocol before the
final run. Overlapping standalone intervals alone do not prove equivalence.

Benchmark output includes failures and coverage, per-target scores, sampling
metadata, and reproducible aggregate tables. Publish these with checkpoints
so claims can be recomputed without rerunning all predictions.

## 9. Testing and validation

Use pytest under `tests/fold/`, real structure fixtures where possible, and
`@pytest.mark.slow` for GPU and end-to-end work. The three main oracles are
the released ESMFold2 model, pinned Boltz-2 loss functions, and the staged
PyTorch kernel reference.

| Layer | Required checks | Execution |
|---|---|---|
| Data | CCD/parent mapping, unresolved sequence positions, altloc/model selection, dropped entities, index consistency, crops, atom budgets | Fast CPU, small real fixtures |
| Pair features | Arbitrary row/column blocks equal slices of the full call for every §4.6 function | Fast CPU |
| Model | Shapes/masks, recurrence gradient scope, checkpoint equivalence, finite sampler output, frozen-LM exclusion | Tiny CPU models; GPU where needed |
| Losses | Boltz-2 term parity, rigid-transform recovery, symmetry assignment, masks/degeneracies, confidence bins, interface-only weighting | Fast CPU |
| Kernels | Incoming/outgoing trimul forward and gradients, mixed fused/reference path, FlexAttention outputs and bias gradients | Slow GPU |
| ESMFold2 port | Deterministic intermediate tensors at width 256 | Slow GPU, fixture-gated |
| Trainer | Real steps, checkpoint/resume, EMA restoration, mixed-source cursor continuity, unchanged default MLM behavior | Existing slow E2E pattern |
| Metrics | Identity cases and known structure-pair scores; correct DockQ mapping and aggregation | Fast fixtures / optional eval extra |

Data fixtures include a small monomer, heterodimer, modified-residue entry with
an unresolved loop, and multi-model NMR entry, plus the CCD subset they need.
Test interface crops, missing frames, empty masks, and overflow handling. Keep
fixture provenance and redistribution notices alongside the files.

Vendor only the needed Boltz-2 loss functions as frozen, attributed test
oracles, with their source revision. Compare matched definitions and reductions.
Perfect coordinates should give zero coordinate/bond errors and correct target
bins; finite cross-entropy logits and smooth surrogate losses need not be
exactly zero, so test their actual mathematical expectations.

Generate ESMFold2 golden fixtures once in a separate pinned environment:
the repo currently requires `transformers<5.4`, while the upstream HF port
uses a later release. Save hidden states, weights/remap metadata, feature
inputs, RNG choices, and deterministic intermediate outputs for 3–5 examples.
Compare shim output, pair states after 1/2/3 loops, distogram logits, a denoiser
evaluation at fixed noisy coordinates and noise level, and confidence on fixed
coordinates. Use saved recurrence initial states and conformer rotations;
do not compare independently sampled structures as a port oracle. Skip with
an explicit reason when fixture artifacts are unavailable, and require the
fixture-backed run before starting training.

Kernel parity covers widths 128/256, lengths 128–1024, masks, both directions,
and gradients for inputs and parameters. Set tolerances from fp32-reference
comparisons and publish them; do not assert bf16 bitwise identity. Exercise
the fused-forward/reference-backward path together with checkpointing. The
larger §5.2 benchmark is a CLI experiment, not a routine test-suite burden.

End-to-end fold tests follow [the repository E2E plan](../../TESTING_E2E.md).
Confirm finite gradients, that a tiny overfit run can reduce its objective,
and that resume restores step, data cursor, and EMA. Existing MLM tests remain
the regression gate. Run `ruff check src/`, formatting checks, `ty check src/`,
and the relevant pytest suites on implementation changes.

DDP remains part of v1 training and needs a small distributed smoke/resume
check. Fold-CP collectives and context-parallel training tests are deferred
with those features. Prediction accuracy is measured by §8, not a flaky unit
test threshold.

## 10. Milestones and decision gates

The numbered milestones below expand the discussion's first user-facing
milestone, pipeline proof, into prerequisite work. Downloads can overlap
kernel and port work; no new training run begins before deterministic parity.

| # | Deliverables | Acceptance |
|---|---|---|
| 0 — Foundations and kernels | Fold config/package, task seam, EMA lifecycle, staged/fused trimul, attention paths, B200 benchmark | MLM regressions clean; forward/backward parity on B200; published width-128/256 cost and memory results |
| 1 — Inference port | Shim, pair features, recurrence/trunk, atoms, diffusion, confidence, distogram, predict/mmCIF, golden fixture tooling | Deterministic ESMFold2 parity and block-feature parity pass |
| 2 — Pipeline proof | PDB/CCD/distillation subset, crops/loaders/losses, monomer eval, 24-layer training at crop 384 | Slurm run with exercised resume; improving held-out TM-score; width selected from accuracy and benchmark cost |
| 3 — v1 | Phase-2 LM, 48-layer trunk, stages 1–4, clustered sampling, AFDB dimers, interface-loss ablation, FoldBench/SAbDab and same-harness baselines | Both FoldBench success comparisons meet the preregistered criterion; measured 2048-token training and inference length results; complete release artifacts |
| 4 — Ab–Ag distillation | Permitted sequence-pair curation, ranker hook, leakage exclusions, versioned distillates, stage 5 | Improved held-out Ab–Ag DockQ with general metrics reported; reproducible data/ranker provenance and releasable artifacts |
| 5 — Ligands and nucleic acids | CCD ligand/NA ingestion, leaving atoms and covalent bonds, bond supervision, continuation training | Ligand/NA benchmark evaluation, retained protein performance, unchanged model representation/checkpoint dimensions |
| 6 — Fold-CP inference | Pair sharding, distributed contractions/transposes and pair-biased attention | Single-/multi-GPU parity on fitting examples, plus successful inference on an example exceeding one GPU's measured capacity |

Milestone 3 depends on the phase-2 LM, but its infrastructure can be developed
using milestone-2 checkpoints. Milestone 5 requires continued training with
the trunk unfrozen: general representation alone does not teach ligand or
nucleic-acid behavior. Milestone 6 changes execution and should not require
retraining; context-parallel training is a separate project.

Estimate allocation time from measured complete training steps, validation,
and checkpoint overhead after milestones 0–2. The discussion's several-day
B200 estimate is only an order-of-magnitude planning assumption, particularly
with longer crops and a reference backward at width 256.

Decisions intentionally resolved by experiments or data inspection:

- Pair width: select after milestone-2 accuracy/cost comparison; retain the
  width-256 parity configuration regardless.
- Monomer subset size, confidence filters, and dimer fractions: freeze after
  inspecting source metadata, before the corresponding run.
- Interface-loss weight and optional noise schedule: controlled stage ablations.
- Exact LR/global-batch settings and use of 2048 in the long-crop curriculum:
  set from the pipeline proof and measured memory/throughput.
- Stage-5 sources, teacher, and ranker: require a reproducible permitted path
  under §11; restricted inputs are not a dependency of the core release.
- Benchmark equivalence margin: record before final evaluation, alongside
  the fixed targets and sampling budget.

These are explicit decision gates, not missing subsystem designs. Deferred
features are MSA encoding, LM fp8, head μP/HSDP, biological-assembly expansion,
sampler distillation, and context-parallel training. Ligand/NA support and
Fold-CP inference remain committed near-term milestones.

## 11. Open release and reproducibility contract

Openness is a design requirement and a milestone acceptance criterion. A
usable release includes the path from source data to trained checkpoint, not
just the final inference package.

| Artifact | Destination | Required contents |
|---|---|---|
| Code | GitHub | Model, losses, training, preprocessing/acquisition, distillation/ranking integration, evaluation, kernel benchmarks, tests and fixture-generation tools |
| Training datasets | Hugging Face datasets | Actual redistributable training shards, metadata and splits, source/version/checksum manifests, exclusion and clustering rules, CCD provenance, dataset cards and applicable licenses |
| Weights | Hugging Face models | Final ordinary/EMA weights and useful intermediate checkpoints; HF config/code references, immutable LM dependency, metrics, training recipe and model card |
| Continuation state | Versioned checkpoint artifacts linked from Hugging Face | Optimizer/scheduler, EMA, RNG, data cursor, resolved config, code/environment revision, dataset revisions, and documented resume procedure |
| Evaluation/benchmarks | GitHub plus linked Hugging Face artifacts | Per-target scores and permitted predictions, fixed manifests, baseline settings, kernel timings/memory, reproduction commands |

Preserve phase-1 LM checkpoints before phase-2 data rebalancing and LR decay,
and each useful folding stage boundary before changing crops, objectives, or
freezing. Include selected pipeline-proof/width-ablation checkpoints when
they enable meaningful comparisons. This extends the project's release
policy to the LM dependency; it does not require retraining the LM as part
of the head implementation.

Inference-only safetensors and fully resumable training states serve different
uses. Publish both at designated continuation points, with topology/version
constraints documented for distributed state. Users should be able to begin
an alternative continuation from a published boundary without reconstructing
an unavailable optimizer or dataset cursor. Local checkpoint garbage collection
must protect designated release checkpoints until publication is verified.

Keep new OPLM code under the repository's MIT license and preserve upstream
headers/notices for copied components. Select permissive weight and data
licenses wherever the rights allow, and document actual licenses separately
for code, weights, datasets, teacher outputs, and vendored fixtures. A top-level
MIT label must not erase a source's attribution or noncommercial conditions.

The default release recipe must be reproducible from publicly obtainable,
redistributable inputs. ASD or a private ranker cannot become an undocumented
requirement: use a permitted alternative, make the required method/artifacts
releasable, or label that experiment separately from the open baseline. A
private ranker can be supported through the hook, but results depending on it
are not fully reproducible until its required scoring method and dependencies
are available. Teacher-generated labels carry both teacher and input provenance.

For each release, publish the actual curated training data rather than only
download scripts wherever redistribution is permitted. If a proposed source
cannot meet that requirement, record the exception and keep it outside the
default permissive recipe; an acquisition manifest is useful but is not an
equivalent substitute for the requested public training dataset.

Release verification loads a checkpoint by its public identifier in a clean
environment, resolves its frozen LM, runs prediction, and exercises the
documented continuation path on the published data. Record artifact checksums
and exact commands. This document specifies those future release deliverables;
publication and training are subsequent implementation work.
