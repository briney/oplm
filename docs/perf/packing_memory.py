"""Packed rows that fit one GPU without activation checkpointing vs rows real batches need.

With packing, switching activation checkpointing off made the 170M step 24% faster (303 to 232 ms
on 8 B200s), and typical packed batches fit. But a batch's 128 sequences need as many 512-token
rows as their lengths demand, and a padded 128 x 512 batch does not fit without checkpointing. A
batch that does not fit ends the run, and since the data order is deterministic, the requeue hits
the same batch again. This measures both sides:

- rows needed: the first-fit-decreasing row count (as in MLMCollator) of many simulated batches,
  lengths drawn from several shards of each data.train source, mixed by its fraction;
- rows that fit: peak memory of a compiled training forward + backward (bf16 autocast, no
  activation checkpointing) for packed batches of increasing row count, until one runs out of
  memory. Optimizer state and DDP buckets come on top: a margin is subtracted.

    python docs/perf/packing_memory.py --checkpoint <run>/checkpoint-<step>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from oplm.config import load_config
from oplm.data.config import parse_train_configs
from oplm.data.tokenizer import canonical_amino_acid_ids, get_tokenizer
from oplm.model import OplmForMaskedLM
from oplm.training.stream_replay import compile_like_trainer

MARGIN_GIB = 6.0  # optimizer state (~3 fp32 copies of 170M params) + DDP buckets + headroom


def token_lengths(cfg, shards_per_source: int) -> list[tuple[float, np.ndarray]]:
    """Per data.train source: its fraction, and tokenized lengths (with <cls>/<eos>, capped)."""
    rng = np.random.default_rng(0)
    cap = cfg.model.max_position_embeddings
    out = []
    for entry in parse_train_configs(cfg.data.train):
        path = Path(entry.path)
        shards = [path] if path.suffix == ".parquet" else sorted(path.glob("*.parquet"))
        picks = rng.choice(len(shards), min(shards_per_source, len(shards)), replace=False)
        lengths = np.concatenate(
            [
                pc.utf8_length(
                    pq.read_table(shards[i], columns=["sequence"])["sequence"]
                ).to_numpy()
                for i in picks
            ]
        )
        out.append((entry.fraction, np.minimum(lengths + 2, cap)))
    return out


def ffd_rows(lengths: list[int], capacity: int) -> int:
    """Rows first-fit decreasing packs ``lengths`` into (MLMCollator's packing)."""
    free: list[int] = []
    for n in sorted(lengths, reverse=True):
        for i, room in enumerate(free):
            if n <= room:
                free[i] -= n
                break
        else:
            free.append(capacity - n)
    return len(free)


def sample_batch(sources, rng, size: int) -> list[int]:
    fractions = np.array([f for f, _ in sources])
    picks = rng.choice(len(sources), size, p=fractions / fractions.sum())
    return [int(sources[s][1][rng.integers(len(sources[s][1]))]) for s in picks]


def packed_batch(
    sources, rng, rows: int, width: int, vocab: torch.Tensor
) -> dict[str, torch.Tensor]:
    """``rows`` full rows of real-length segments (random residues, 15% labeled)."""
    segment_ids = torch.zeros(rows, width, dtype=torch.long)
    for r in range(rows):
        start, seg = 0, 1
        while start < width:
            n = min(sample_batch(sources, rng, 1)[0], width - start)
            segment_ids[r, start : start + n] = seg
            start, seg = start + n, seg + 1
    input_ids = vocab[torch.randint(len(vocab), (rows, width))]
    labels = torch.where(torch.rand(rows, width) < 0.15, input_ids, -100)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(rows, width, dtype=torch.long),
        "labels": labels,
        "segment_ids": segment_ids,
    }
    return {k: v.cuda() for k, v in batch.items()}


def peak_gib(model, batch) -> float:
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shards-per-source", type=int, default=6)
    parser.add_argument("--batches", type=int, default=20_000)
    args = parser.parse_args()
    print(torch.__version__, torch.cuda.get_device_name())
    cfg = load_config(["--config", str(args.checkpoint / "config.yaml")])
    width, size = cfg.model.max_position_embeddings, cfg.train.batch_size
    rng = np.random.default_rng(1)

    sources = token_lengths(cfg, args.shards_per_source)
    for (fraction, lengths), entry in zip(
        sources, parse_train_configs(cfg.data.train), strict=True
    ):
        print(
            f"{entry.name}: fraction {fraction:.2f}, {len(lengths)} sequences, mean "
            f"{lengths.mean():.0f} tokens, {np.mean(lengths == width):.1%} at the {width} cap"
        )
    needed = np.array(
        [ffd_rows(sample_batch(sources, rng, size), width) for _ in range(args.batches)]
    )
    mean, sd = needed.mean(), needed.std()
    print(
        f"\nrows needed per batch of {size} ({args.batches} simulated batches): mean {mean:.1f}, "
        f"sd {sd:.2f}, p99 {np.percentile(needed, 99):.0f}, p99.99 "
        f"{np.percentile(needed, 99.99):.0f}, max {needed.max()}; mean + 6 sd = {mean + 6 * sd:.1f}"
    )

    model = OplmForMaskedLM.from_pretrained(args.checkpoint / "hf").cuda().train()
    model.gradient_checkpointing_disable()  # explicit: the point is to run without it
    compiled = compile_like_trainer(model, cfg, ddp=False)
    vocab = canonical_amino_acid_ids(get_tokenizer())
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"\nGPU {total:.1f} GiB; model weights: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    print(f"{'rows':>6s} {'peak GiB':>10s}")
    fits: list[tuple[int, float]] = []
    for rows in range(32, size + 1, 16):
        try:
            peak = peak_gib(compiled, packed_batch(sources, rng, rows, width, vocab))
        except torch.OutOfMemoryError:
            print(f"{rows:6d} {'OOM':>10s}")
            break
        fits.append((rows, peak))
        print(f"{rows:6d} {peak:10.1f}")
    if len(fits) >= 2:
        (r0, p0), (r1, p1) = fits[0], fits[-1]
        per_row = (p1 - p0) / (r1 - r0)
        limit = r1 + (total - MARGIN_GIB - p1) / per_row
        tail = (limit - mean) / sd
        print(
            f"\n{per_row:.2f} GiB per row; about {limit:.0f} rows fit with a {MARGIN_GIB:.0f} GiB "
            f"margin. Real batches need {mean:.0f} +/- {sd:.1f}: the limit is {tail:.1f} sd above."
        )


if __name__ == "__main__":
    main()
