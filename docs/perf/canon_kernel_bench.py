"""GPU benchmark: Triton Canon kernel vs torch.compile'd reference, forward+backward.

Packed rows of 65,536 tokens (B=32, T=2048) with lognormal protein-like segment lengths
(median ~300, clipped to [20, 2048]). Run on one CUDA GPU:

    python canon_kernel_bench.py --out canon-kernel-bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

MINIMAL_TRAFFIC_NOTE = (
    "ms = back-to-back average of fwd+bwd; GB/s = (fwd: read x, write out; bwd: read dy, read x, "
    "write dx) / ms. The model ignores the dW-partials write and the mask/segment reads. "
    "out = residual output dtype, promote(x, bf16)."
)


def packed_segments(b: int, t: int, seed: int) -> torch.Tensor:
    """`(B, T)` int32 segment ids (1..n per row), rows filled completely; last segment truncated."""
    g = torch.Generator().manual_seed(seed)
    seg = torch.zeros(b, t, dtype=torch.int32)
    for r in range(b):
        pos, sid = 0, 1
        while pos < t:
            n = int(torch.empty(1).log_normal_(5.7, 0.7, generator=g).clamp(20, t).item())
            seg[r, pos : pos + n] = sid
            pos, sid = pos + n, sid + 1
    return seg


def time_ms(fn, iters: int, warmup: int) -> tuple[float, float]:
    """(back-to-back ms/call, per-call synced median ms). The first is primary: `iters` calls
    between one event pair, no sync inside, so CPU launch latency overlaps GPU work as in training.
    The second includes launch latency and is secondary."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    batched = s.elapsed_time(e) / iters
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return batched, statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument("--widths", default="768,2048")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    a = p.parse_args()

    from oplm.model.canon_kernel import canon_conv_kernel
    from oplm.model.conv import canon_conv_reference

    dev = torch.device("cuda")
    seg = packed_segments(a.batch, a.seq_len, seed=0).to(dev)
    mask = seg > 0
    print(
        f"{torch.cuda.get_device_name()} | torch {torch.__version__} | "
        f"mean segment {a.seq_len * a.batch / int(seg.max(1).values.sum()):.0f} tokens"
    )

    rows = []
    for d in (int(w) for w in a.widths.split(",")):
        w = torch.randn(d, a.k, device=dev, requires_grad=True)
        for x_dtype in (torch.bfloat16, torch.float32):
            out_dtype = torch.bfloat16
            x = torch.randn(a.batch, a.seq_len, d, device=dev, dtype=x_dtype, requires_grad=True)
            res_dtype = torch.promote_types(x_dtype, out_dtype)
            dy = torch.randn(a.batch, a.seq_len, d, device=dev, dtype=res_dtype)
            n = a.batch * a.seq_len * d
            xb, ob = x.element_size(), dy.element_size()
            traffic = n * (xb + ob) + n * (ob + xb + xb)  # fwd + bwd

            def step(fn, x=x, w=w, dy=dy, out_dtype=out_dtype):
                def run():
                    x.grad = w.grad = None
                    fn(x, w, mask, seg, out_dtype, True).backward(dy)

                return run

            row = {
                "D": d,
                "k": a.k,
                "x_dtype": str(x_dtype).removeprefix("torch."),
                "out_dtype": str(res_dtype).removeprefix("torch."),
                "traffic_bytes": traffic,
            }
            torch._dynamo.reset()  # fresh static-shape compile per configuration
            ref = torch.compile(canon_conv_reference, dynamic=False)
            for name, fn in (("compiled_ref", ref), ("kernel", canon_conv_kernel)):
                ms, call_ms = time_ms(step(fn), a.iters, a.warmup)
                row[f"{name}_ms"] = ms
                row[f"{name}_call_median_ms"] = call_ms
                row[f"{name}_gbps"] = traffic / (ms * 1e-3) / 1e9
            row["speedup"] = row["compiled_ref_ms"] / row["kernel_ms"]
            rows.append(row)

    print(
        f"\n{'D':>5} {'x':>9} {'out':>9} {'ref ms':>8} {'kern ms':>8} {'speedup':>8} "
        f"{'ref GB/s':>9} {'kern GB/s':>10}"
    )
    for r in rows:
        print(
            f"{r['D']:>5} {r['x_dtype']:>9} {r['out_dtype']:>9} "
            f"{r['compiled_ref_ms']:>8.3f} {r['kernel_ms']:>8.3f} "
            f"{r['speedup']:>7.2f}x {r['compiled_ref_gbps']:>9.0f} {r['kernel_gbps']:>10.0f}"
        )
    print(MINIMAL_TRAFFIC_NOTE)
    with open(a.out, "w") as f:
        json.dump(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "batch": a.batch,
                "seq_len": a.seq_len,
                "results": rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
