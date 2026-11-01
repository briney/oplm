"""GPU benchmark: Triton Canon kernel vs torch.compile'd reference, forward and backward.

Packed rows of 65,536 tokens (B=32, T=2048) with lognormal protein-like segment lengths
(median ~300, clipped to [20, 2048]). Run on one CUDA GPU:

    python canon_kernel_bench.py --out canon-kernel-bench.json            # kernel vs reference
    python canon_kernel_bench.py --out canon-kernel-bench.json --sweep    # + launch configs

`--sweep` also times `canon_forward` and `canon_backward` separately for every `FwdConfig` /
`BwdConfig` in a grid. Each config is first checked against the reference at the timed shape; a
config that fails to compile or disagrees is reported (`error`), not timed. The best config per
(D, dtypes, phase) is printed as `FWD_CONFIGS` / `BWD_CONFIGS` entries for `canon_kernel.py`.
`--tiny` runs everything on a toy shape (on CPU under `TRITON_INTERPRET=1` too), a smoke test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

MINIMAL_TRAFFIC_NOTE = (
    "ms = back-to-back average; bwd = (fwd+bwd step) - fwd. GB/s = (fwd: read x, write out; "
    "bwd: read dy, read x, write dx) / ms. The model ignores the dW-partials write and the "
    "mask/segment reads. out = residual output dtype, promote(x, bf16)."
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


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn: Callable[[], object], iters: int, warmup: int, median: bool = True):
    """(back-to-back ms/call, per-call synced median ms). The first is primary: `iters` calls
    between two syncs, so CPU launch latency overlaps GPU work as in training. The second
    includes launch latency and is secondary (NaN without `median`)."""
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    batched = (time.perf_counter() - start) * 1e3 / iters
    if not median:
        return batched, float("nan")
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - start) * 1e3)
    return batched, statistics.median(times)


def fwd_grid():
    """Forward configs; skips tiles over 64 fp32 accumulator elements per thread (spills)."""
    from oplm.model.canon_kernel import FwdConfig

    return [
        FwdConfig(bt, bd, w)
        for bt, bd, w in itertools.product((16, 32, 64, 128), (32, 64, 128), (2, 4, 8))
        if bt * bd / (32 * w) <= 64
    ]


def bwd_grid(k: int):
    """Backward configs: tiles per program for a 256- or 512-token span, dw reduced per tile or
    deferred. Skips 3D (taps x tile) tensors over 128 fp32 elements per thread (spills)."""
    from oplm.model.canon_kernel import BwdConfig

    k_pow2 = 1 << (k - 1).bit_length()
    grid = itertools.product((8, 16, 32, 64), (32, 64, 128), (2, 4, 8), (True, False), (256, 512))
    return [
        BwdConfig(bt, bd, span // bt, defer, w)
        for bt, bd, w, defer, span in grid
        if k_pow2 * bt * bd / (32 * w) * (2 if defer else 1) <= 128
    ]


def max_rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error relative to the reference's max magnitude."""
    ref = ref.float()
    return ((got.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


# Correctness gate per config: out / dx within 1% of max |ref| (bf16 rounding flips), dW within
# 0.1% (fp32 sums in another order). Miscompiles (missed tiles, garbage) are far beyond either.
SWEEP_TOL = (1e-2, 1e-3)


def sweep(a, base: dict, x, w, dy, mask, seg, out_dtype) -> list[dict]:
    """Every grid config, forward and backward separately, at this (D, dtypes)."""
    from oplm.model.canon_kernel import BwdConfig, FwdConfig, canon_backward, canon_forward
    from oplm.model.conv import canon_conv_reference

    xr, wr = x.detach().requires_grad_(), w.detach().requires_grad_()
    ref_out = canon_conv_reference(xr, wr, mask, seg, out_dtype, True)
    ref_out.backward(dy)
    refs = {"fwd": (ref_out.detach(),), "bwd": (xr.grad, wr.grad)}
    del ref_out, xr, wr
    n = x.numel()
    traffic = {"fwd": n * (x.element_size() + dy.element_size())}
    traffic["bwd"] = n * (dy.element_size() + 2 * x.element_size())
    args = (x, w, mask, seg, out_dtype, True)
    calls = {
        "fwd": (FwdConfig(), fwd_grid(), lambda c: (canon_forward(*args, config=c),)),
        "bwd": (BwdConfig(), bwd_grid(a.k), lambda c: canon_backward(dy, *args, config=c)),
    }
    rows = []
    for phase, (default, configs, call) in calls.items():
        for cfg in configs:
            row = {**base, "phase": phase, "config": cfg._asdict(), "default": cfg == default}
            run = lambda call=call, cfg=cfg: call(cfg)  # noqa: E731
            try:
                errs = [max_rel_err(g, r) for g, r in zip(run(), refs[phase], strict=True)]
            except Exception as e:  # noqa: BLE001 - compile failures (e.g. out of resources)
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            else:
                row["max_rel_err"] = errs
                if any(e > tol for e, tol in zip(errs, SWEEP_TOL, strict=False)):
                    row["error"] = f"mismatch vs reference: max rel err {errs}"
                else:
                    row["ms"] = time_ms(run, a.iters, a.warmup, median=False)[0]
                    row["gbps"] = traffic[phase] / (row["ms"] * 1e-3) / 1e9
            rows.append(row)
            status = row.get("error") or f"{row['ms']:.3f} ms"
            print(f"  {phase} D={base['D']} {base['x_dtype']} {cfg}: {status}", flush=True)
    return rows


def best_configs(rows: list[dict]) -> list[dict]:
    """Fastest correct config per (D, x dtype, conv dtype, phase), with the default's time."""
    best = {}
    for r in rows:
        key = (r["D"], r["x_dtype"], r["conv_dtype"], r["phase"])
        if r["default"]:
            best.setdefault(key, {})["default_ms"] = r.get("ms")
        if "ms" in r and r["ms"] < best.setdefault(key, {}).get("ms", float("inf")):
            best[key].update(ms=r["ms"], gbps=r["gbps"], config=r["config"])
    return [
        {"D": k[0], "x_dtype": k[1], "conv_dtype": k[2], "phase": k[3], **v}
        for k, v in best.items()
    ]


def print_sweep(rows: list[dict], best: list[dict]) -> None:
    bad = [r for r in rows if "error" in r]
    print(f"\nsweep: {len(rows)} configs, {len(bad)} failed or mismatched")
    for r in bad:
        print(f"  {r['phase']} D={r['D']} {r['x_dtype']} {r['config']}: {r['error']}")
    print(f"\n{'D':>5} {'x':>9} {'phase':>5} {'default ms':>10} {'best ms':>8} {'GB/s':>6}  config")
    for b in best:
        if "config" not in b:
            continue
        default = b.get("default_ms")
        print(
            f"{b['D']:>5} {b['x_dtype']:>9} {b['phase']:>5} "
            f"{default if default is not None else float('nan'):>10.3f} {b['ms']:>8.3f} "
            f"{b['gbps']:>6.0f}  {b['config']}"
        )
    for phase, table, cls in (
        ("fwd", "FWD_CONFIGS", "FwdConfig"),
        ("bwd", "BWD_CONFIGS", "BwdConfig"),
    ):
        print(f"\n{table} = {{")
        for b in best:
            if b["phase"] == phase and "config" in b:
                fields = ", ".join(f"{k}={v}" for k, v in b["config"].items())
                key = f"({b['D']}, torch.{b['x_dtype']}, torch.{b['conv_dtype']})"
                print(f"    {key}: {cls}({fields}),")
        print("}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument("--widths", default="768,2048")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--sweep", action="store_true", help="time every launch config, fwd and bwd")
    p.add_argument("--tiny", action="store_true", help="toy shape and 1 iteration (smoke test)")
    a = p.parse_args()
    if a.tiny:
        a.batch, a.seq_len, a.widths, a.iters, a.warmup = 2, 96, "64", 1, 0

    from oplm.model.canon_kernel import canon_conv_kernel
    from oplm.model.conv import canon_conv_reference

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU: interpreter only
    device_name = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    seg = packed_segments(a.batch, a.seq_len, seed=0).to(dev)
    mask = (seg > 0).long()  # int64, as training's collator builds it
    print(
        f"{device_name} | torch {torch.__version__} | "
        f"mean segment {a.seq_len * a.batch / int(seg.max(1).values.sum()):.0f} tokens"
    )

    rows, sweep_rows = [], []
    for d in (int(w) for w in a.widths.split(",")):
        w = torch.randn(d, a.k, device=dev, requires_grad=True)
        for x_dtype in (torch.bfloat16, torch.float32):
            out_dtype = torch.bfloat16
            x = torch.randn(a.batch, a.seq_len, d, device=dev, dtype=x_dtype, requires_grad=True)
            res_dtype = torch.promote_types(x_dtype, out_dtype)
            dy = torch.randn(a.batch, a.seq_len, d, device=dev, dtype=res_dtype)
            n = a.batch * a.seq_len * d
            xb, ob = x.element_size(), dy.element_size()
            traffic = {"fwd": n * (xb + ob), "bwd": n * (ob + xb + xb)}
            traffic["step"] = traffic["fwd"] + traffic["bwd"]

            def fwd(fn, x=x, w=w, out_dtype=out_dtype):
                return lambda: fn(x, w, mask, seg, out_dtype, True)

            def step(fn, x=x, w=w, dy=dy, out_dtype=out_dtype):
                def run():
                    x.grad = w.grad = None
                    fn(x, w, mask, seg, out_dtype, True).backward(dy)

                return run

            row = {
                "D": d,
                "k": a.k,
                "x_dtype": str(x_dtype).removeprefix("torch."),
                "conv_dtype": str(out_dtype).removeprefix("torch."),
                "out_dtype": str(res_dtype).removeprefix("torch."),
                "traffic_bytes": traffic["step"],
            }
            base = dict(row)
            torch._dynamo.reset()  # fresh static-shape compile per configuration
            ref = torch.compile(canon_conv_reference, dynamic=False)
            for name, fn in (("compiled_ref", ref), ("kernel", canon_conv_kernel)):
                ms, call_ms = time_ms(step(fn), a.iters, a.warmup)
                row[f"{name}_ms"] = ms
                row[f"{name}_call_median_ms"] = call_ms
                row[f"{name}_gbps"] = traffic["step"] / (ms * 1e-3) / 1e9
                row[f"{name}_fwd_ms"] = time_ms(fwd(fn), a.iters, a.warmup, median=False)[0]
                row[f"{name}_bwd_ms"] = ms - row[f"{name}_fwd_ms"]
                for phase in ("fwd", "bwd"):
                    row[f"{name}_{phase}_gbps"] = (
                        traffic[phase] / (row[f"{name}_{phase}_ms"] * 1e-3) / 1e9
                    )
            row["speedup"] = row["compiled_ref_ms"] / row["kernel_ms"]
            rows.append(row)
            if a.sweep:
                x_in = x.detach()
                sweep_rows += sweep(a, base, x_in, w.detach(), dy, mask, seg, out_dtype)

    print(
        f"\n{'D':>5} {'x':>9} {'out':>9} {'ref fwd':>8} {'ref bwd':>8} {'kern fwd':>8} "
        f"{'kern bwd':>8} {'speedup':>8} {'ref GB/s':>9} {'kern GB/s':>10}"
    )
    for r in rows:
        print(
            f"{r['D']:>5} {r['x_dtype']:>9} {r['out_dtype']:>9} "
            f"{r['compiled_ref_fwd_ms']:>8.3f} {r['compiled_ref_bwd_ms']:>8.3f} "
            f"{r['kernel_fwd_ms']:>8.3f} {r['kernel_bwd_ms']:>8.3f} "
            f"{r['speedup']:>7.2f}x {r['compiled_ref_gbps']:>9.0f} {r['kernel_gbps']:>10.0f}"
        )
    print(MINIMAL_TRAFFIC_NOTE)
    result = {
        "device": device_name,
        "torch": torch.__version__,
        "batch": a.batch,
        "seq_len": a.seq_len,
        "results": rows,
    }
    if a.sweep:
        best = best_configs(sweep_rows)
        print_sweep(sweep_rows, best)
        result.update(sweep=sweep_rows, best=best)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
