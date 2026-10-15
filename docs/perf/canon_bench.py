"""Time candidate implementations of Canon's depthwise convolution at training shapes (1 GPU).

Canon is a kernel-size-k depthwise convolution along the sequence axis of a (B, T, D) tensor,
with pad positions zeroed first. A 170M training step (batch 128 per GPU, sequences padded to
512) runs it 144 times. This script times each formulation's forward and forward+backward
under bf16 autocast, compiled the way the trainer compiles (dynamic=True) and statically, and
reports the effective bandwidth (input read + output written) against what one pass should cost.

    python docs/perf/canon_bench.py [--batch 128] [--length 512] [--kernel 7]
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from oplm.model.conv import CanonConv


def shift_add(x: torch.Tensor, mask: torch.Tensor, w: torch.Tensor, upcast: bool) -> torch.Tensor:
    k = w.shape[1]
    x = x * mask.unsqueeze(-1).to(x.dtype)
    left = k // 2
    p = F.pad(x, (0, 0, left, k - 1 - left))
    if upcast:
        p, w = p.float(), w.float()
    else:
        w = w.to(p.dtype)
    t = x.shape[1]
    out = p[:, :t] * w[:, 0]
    for i in range(1, k):
        out = out + p[:, i : i + t] * w[:, i]
    return out.to(x.dtype)


def unfold_sum(x: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    k = w.shape[1]
    x = x * mask.unsqueeze(-1).to(x.dtype)
    left = k // 2
    p = F.pad(x, (0, 0, left, k - 1 - left))
    windows = p.unfold(1, k, 1)  # (B, T, D, k), a view
    return (windows.float() * w.float()).sum(-1).to(x.dtype)


def conv1d_bdt(x: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    k = w.shape[1]
    x = (x * mask.unsqueeze(-1).to(x.dtype)).transpose(1, 2)
    left = k // 2
    x = F.pad(x, (left, k - 1 - left))
    return F.conv1d(x, w.unsqueeze(1).to(x.dtype), groups=w.shape[0]).transpose(1, 2)


def conv2d_channels_last(x: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # (B, T, D) is already channels-last for a (B, D, 1, T) image: no copy needed.
    k = w.shape[1]
    x = x * mask.unsqueeze(-1).to(x.dtype)
    img = x.permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, T), channels_last strides
    left = k // 2
    img = F.pad(img, (left, k - 1 - left))
    out = F.conv2d(img, w.view(w.shape[0], 1, 1, k).to(x.dtype), groups=w.shape[0])
    return out.squeeze(2).permute(0, 2, 1)


def bench(fn, x, mask, w, backward: bool, iters: int = 20) -> float:
    upstream = torch.randn_like(x)
    for _ in range(3):
        if backward:
            x.grad = w.grad = None
            fn(x, mask, w).backward(upstream)
        else:
            with torch.no_grad():
                fn(x, mask, w)
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if backward:
            x.grad = w.grad = None
            fn(x, mask, w).backward(upstream)
        else:
            with torch.no_grad():
                fn(x, mask, w)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--kernel", type=int, default=7)
    args = parser.parse_args()
    dev = torch.device("cuda")
    print(torch.__version__, torch.cuda.get_device_name(dev))
    for channels in (768, 2048):
        torch.manual_seed(0)
        x = torch.randn(args.batch, args.length, channels, device=dev, dtype=torch.bfloat16)
        x.requires_grad_(True)
        mask = torch.ones(args.batch, args.length, device=dev, dtype=torch.long)
        mask[:, args.length // 3 :] = torch.rand(args.batch, 1, device=dev) > 0.8  # padded rows
        conv = CanonConv(channels, args.kernel).to(dev)
        w = conv.conv.weight.detach().squeeze(1).clone().requires_grad_(True)
        gbytes = 2 * x.numel() * x.element_size() / 1e9  # one read + one write
        candidates = {
            "CanonConv (oplm, eager)": lambda a, m, b, conv=conv: conv(a, m),
            "CanonConv (oplm, compile dynamic)": torch.compile(
                lambda a, m, b, conv=conv: conv(a, m), dynamic=True
            ),
            "shift_add fp32 (compile dynamic)": torch.compile(
                lambda a, m, b: shift_add(a, m, b, True), dynamic=True
            ),
            "shift_add fp32 (compile static)": torch.compile(
                lambda a, m, b: shift_add(a, m, b, True), dynamic=False
            ),
            "shift_add bf16 (compile dynamic)": torch.compile(
                lambda a, m, b: shift_add(a, m, b, False), dynamic=True
            ),
            "unfold_sum (compile dynamic)": torch.compile(unfold_sum, dynamic=True),
            "conv1d (B,D,T) eager [old path]": conv1d_bdt,
            "conv2d channels-last eager (cuDNN)": conv2d_channels_last,
            "conv2d channels-last (compile dynamic)": torch.compile(
                conv2d_channels_last, dynamic=True
            ),
        }
        print(
            f"\nD={channels}  B={args.batch} T={args.length} k={args.kernel}  "
            f"one memory-bound pass moves {gbytes:.2f} GB"
        )
        print(f"{'implementation':44s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} {'fwd GB/s':>9s}")
        for name, fn in candidates.items():
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fwd = bench(fn, x, mask, w, backward=False)
                    both = bench(fn, x, mask, w, backward=True)
                print(f"{name:44s} {fwd:8.3f} {both:11.3f} {gbytes / (fwd / 1e3):9.0f}")
            except Exception as exc:  # noqa: BLE001
                print(f"{name:44s} failed: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
