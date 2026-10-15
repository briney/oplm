"""Check the packed training path on one GPU: packed rows vs one padded row per sequence.

The CPU tests (tests/model/test_packing.py) show that a packed row computes what each
sequence's own padded row computes, with PyTorch's reference attention in fp32. On a GPU the
packed path runs cuDNN's fused attention with a block-diagonal mask instead, under bf16
autocast and torch.compile. This runs a real checkpoint on real training sequences at
training shapes (train mode, activation checkpointing as configured) and compares the
logits at every real token, each sequence's masked-token loss, and the parameter gradients:

  cudnn      packed vs padded, attention forced to cuDNN in both (the eager training path)
  efficient  packed vs padded, packed attention forced to the memory-efficient kernel
  compiled   compiled packed vs eager packed (the trainer's torch.compile settings)
  floor      padded bf16 vs padded fp32: how far bf16 alone moves the same numbers

A masking bug shows up far above the floor (a leak between sequences moves logits by about
their own size); bf16 reduction-order noise stays near or below it.

    python docs/perf/packing_check.py --checkpoint <run>/checkpoint-<step> [--batches 4]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from oplm.config import load_config
from oplm.data.config import parse_train_configs
from oplm.data.sequence.collate import MLMCollator
from oplm.data.tokenizer import get_tokenizer
from oplm.model import OplmForMaskedLM
from oplm.training.stream_replay import compile_like_trainer


def sample_sequences(cfg, n: int) -> list[str]:
    """``n`` sequences mixed by the data.train fractions, drawn from each source's first shard."""
    rng = np.random.default_rng(0)
    entries = parse_train_configs(cfg.data.train)
    out: list[str] = []
    for i, entry in enumerate(entries):
        k = n - len(out) if i == len(entries) - 1 else round(n * entry.fraction)
        path = Path(entry.path)
        shard = path if path.suffix == ".parquet" else sorted(path.glob("*.parquet"))[0]
        seqs = pq.read_table(shard, columns=["sequence"]).column("sequence").to_pylist()
        out += [seqs[j] for j in rng.choice(len(seqs), k, replace=False)]
    rng.shuffle(out)
    return out


def run(model, params, batch, *, fp32: bool = False):
    """Logits, loss and flat fp32 gradients of one training forward + backward."""
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            segment_ids=batch.get("segment_ids"),
        )
    out.loss.backward()
    grads = torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params]
    )
    return {"batch": batch, "logits": out.logits.detach().float(), "grads": grads}


def sequences(result) -> dict[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Per sequence (keyed by its token ids): logits at its real tokens, and its labels."""
    batch, logits = result["batch"], result["logits"]
    seg = batch.get("segment_ids", batch["attention_mask"])
    found: dict[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for r in range(seg.shape[0]):
        for s in seg[r].unique().tolist():
            if s:
                cols = seg[r] == s
                ids = tuple(batch["input_ids"][r, cols].tolist())
                found[ids].append((logits[r, cols], batch["labels"][r, cols]))
    return found


def compare(ref, got) -> dict[str, float]:
    """Logit, per-sequence loss and gradient differences of ``got`` relative to ``ref``."""
    a, b = sequences(ref), sequences(got)
    assert sorted((k, len(v)) for k, v in a.items()) == sorted((k, len(v)) for k, v in b.items())
    la, lb, loss_diff = [], [], []
    for ids, items in a.items():
        for (lg_a, labels), (lg_b, _) in zip(items, b[ids], strict=True):
            la.append(lg_a)
            lb.append(lg_b)
            if (labels != -100).any():
                ce = [F.cross_entropy(lg, labels, ignore_index=-100) for lg in (lg_a, lg_b)]
                loss_diff.append(abs(ce[0] - ce[1]).item())
    la, lb = torch.cat(la), torch.cat(lb)
    ga, gb = ref["grads"], got["grads"]
    return {
        "logits_rel": ((la - lb).norm() / la.norm()).item(),
        "logits_max": (la - lb).abs().max().item(),
        "seq_loss_max": max(loss_diff),
        "grad_rel": ((ga - gb).norm() / ga.norm()).item(),
        "grad_cos": F.cosine_similarity(ga, gb, dim=0).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    print(torch.__version__, torch.cuda.get_device_name())
    cfg = load_config(["--config", str(args.checkpoint / "config.yaml")])
    model = OplmForMaskedLM.from_pretrained(args.checkpoint / "hf").cuda().train()
    if model.config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    params = list(model.parameters())
    compiled = compile_like_trainer(model, cfg, ddp=False)
    size = cfg.train.batch_size
    kwargs = dict(
        max_length=cfg.model.max_position_embeddings,
        mask_prob=cfg.data.mask_prob,
        mask_token_prob=cfg.data.mask_token_prob,
        random_token_prob=cfg.data.random_token_prob,
        deterministic=True,  # same seed and batch index -> the same masks in both layouts
        pad_to_multiple_of=cfg.data.pad_to_multiple_of,
    )
    padded = MLMCollator(get_tokenizer(), **kwargs)
    packed = MLMCollator(get_tokenizer(), pack_sequences=True, **kwargs)
    seqs = sample_sequences(cfg, size * args.batches)

    worst: dict[str, dict[str, float]] = defaultdict(dict)
    for i in range(args.batches):
        rows = [{"sequence": s} for s in seqs[i * size : (i + 1) * size]]
        pad = {k: v.cuda() for k, v in padded(rows).items() if isinstance(v, torch.Tensor)}
        pack = {k: v.cuda() for k, v in packed(rows).items() if isinstance(v, torch.Tensor)}
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            pad_bf16 = run(model, params, pad)
            pack_cudnn = run(model, params, pack)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            pack_efficient = run(model, params, pack)
        pad_fp32 = run(model, params, pad, fp32=True)
        pack_compiled = run(compiled, params, pack)
        print(
            f"batch {i}: {size} sequences, padded {tuple(pad['input_ids'].shape)}, "
            f"packed {tuple(pack['input_ids'].shape)}, "
            f"{int(pad['attention_mask'].sum())} real tokens"
        )
        for name, ref, got in (
            ("cudnn", pad_bf16, pack_cudnn),
            ("efficient", pad_bf16, pack_efficient),
            ("compiled", pack_cudnn, pack_compiled),
            ("floor", pad_fp32, pad_bf16),
        ):
            for key, value in compare(ref, got).items():
                keep = min if key == "grad_cos" else max
                worst[name][key] = keep(worst[name].get(key, value), value)

    keys = list(worst["floor"])
    print(f"\nworst over {args.batches} batches")
    print(f"{'':10s}" + "".join(f"{k:>14s}" for k in keys))
    for name, row in worst.items():
        print(f"{name:10s}" + "".join(f"{row[k]:14.3e}" for k in keys))
    floor = worst["floor"]
    for name in ("cudnn", "efficient", "compiled"):
        ratio = max(worst[name][k] / floor[k] for k in ("logits_rel", "grad_rel", "seq_loss_max"))
        verdict = "PASS" if ratio <= 2 else "CHECK"
        print(f"{name}: {verdict} (worst difference = {ratio:.2f}x the bf16 floor)")


if __name__ == "__main__":
    main()
