"""Packed-attention cost by row width and kernel: dense-mask SDPA vs FlexAttention.

With a dense block-diagonal mask, packed attention computes every score of a row, so its cost
grows with the row width. FlexAttention with a block mask skips the 128x128 blocks between
sequences. This times a compiled training forward + backward (bf16 autocast) on packed batches
with the same number of token slots but different widths (default 84 x 512 and 21 x 2048),
with each kernel, and reports synced step time and peak memory. Sequence lengths are drawn from
the run's data.train sources by their fractions, as docs/perf/packing_memory.py does (imported
from the same directory). The model is a fresh preset (weights do not affect timing).

    python docs/perf/attention_width_bench.py --config <run.yaml> --preset 170M
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packing_memory import packed_batch, token_lengths  # noqa: E402 — sibling script

from oplm.config import load_config  # noqa: E402
from oplm.data.tokenizer import canonical_amino_acid_ids, get_tokenizer  # noqa: E402
from oplm.model import OplmForMaskedLM  # noqa: E402
from oplm.training.stream_replay import compile_like_trainer  # noqa: E402


def parse_shapes(text: str) -> list[tuple[int, int]]:
    """``"84x512,21x2048"`` -> ``[(84, 512), (21, 2048)]`` as (rows, width)."""
    return [tuple(int(v) for v in item.split("x")) for item in text.split(",")]  # type: ignore[misc]


def train_step(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> None:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()


def time_steps(model: torch.nn.Module, batch, warmup: int, steps: int) -> tuple[float, float]:
    """Mean synced ms per step and peak GiB over ``steps`` steps after ``warmup``."""
    for _ in range(warmup):
        train_step(model, batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps * 1000, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="run YAML (for data.train)")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--shapes", default="84x512,21x2048")
    parser.add_argument("--backends", default="sdpa,flex")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--shards-per-source", type=int, default=4)
    parser.add_argument("--checkpointing", action="store_true", help="activation checkpointing")
    args = parser.parse_args()
    print(torch.__version__, torch.cuda.get_device_name())

    shapes = parse_shapes(args.shapes)
    cfg = load_config(["--config", str(args.config), "--preset", args.preset])
    cfg.model.max_position_embeddings = max(width for _, width in shapes)
    sources = token_lengths(cfg, args.shards_per_source)
    vocab = canonical_amino_acid_ids(get_tokenizer())
    rng = np.random.default_rng(0)
    batches = {shape: packed_batch(sources, rng, *shape, vocab) for shape in shapes}

    model = OplmForMaskedLM(cfg.model).cuda().train()
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.preset}: {params:.0f}M params, checkpointing={args.checkpointing}")
    print(f"{'kernel':>6s} {'rows x width':>13s} {'real tokens':>12s} {'ms':>8s} {'GiB':>7s}")

    results: dict[tuple[str, tuple[int, int]], float] = {}
    for backend in args.backends.split(","):
        model.config.packed_attention = backend
        for shape, batch in batches.items():
            torch._dynamo.reset()
            compiled = compile_like_trainer(model, cfg, ddp=False)
            label = f"{shape[0]} x {shape[1]}"
            real = int(batch["attention_mask"].sum())
            try:
                ms, gib = time_steps(compiled, batch, args.warmup, args.steps)
            except torch.OutOfMemoryError:
                print(f"{backend:>6s} {label:>13s} {real:12d} {'OOM':>8s}")
                torch.cuda.empty_cache()
                continue
            results[backend, shape] = ms
            print(f"{backend:>6s} {label:>13s} {real:12d} {ms:8.1f} {gib:7.1f}")

    narrow, wide = min(shapes, key=lambda s: s[1]), max(shapes, key=lambda s: s[1])
    for backend in args.backends.split(","):
        if (backend, narrow) in results and (backend, wide) in results:
            ratio = results[backend, wide] / results[backend, narrow]
            print(f"{backend}: {wide[0]} x {wide[1]} costs {ratio:.2f}x {narrow[0]} x {narrow[1]}")


if __name__ == "__main__":
    main()
