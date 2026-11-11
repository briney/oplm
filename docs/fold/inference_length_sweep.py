"""OPLMFold inference memory sweep: peak allocated GiB, per phase, against token length.

The head is randomly initialized (memory does not depend on the weights) and the frozen-LM
states are random, so the LM itself is excluded. Otherwise the run follows
``oplm.fold.predict.fold``: fp32 parameters under the model's own bf16 autocast, eval mode,
``no_grad``, tokens padded to a multiple of 128. Loops and steps are cut to 2: under ``no_grad``
only one loop or step is live at a time, so the peak does not depend on them (each width first
checks this at 1024 tokens against the config's defaults). Every case starts from a reset
dynamo cache and runs twice, the first compiling FlexAttention; the second is recorded. A
configuration stops at its first OOM, which records the phase it was in.

    python docs/fold/inference_length_sweep.py --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from typing import Any

import torch

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.modeling_fold import OplmForFolding

GIB = 2**30
AMINO = "ACDEFGHIKLMNPQRSTVWY"
# (submodule, method) pairs timed and measured as phases; missing ones are skipped.
PHASES = [
    ("input_embedder", "forward"),
    ("language_model", "forward"),
    ("parcae", "run"),
    ("parcae", "readout"),
    ("structure_head", "prepare"),
    ("structure_head", "sample"),
    ("confidence_head", "forward"),
]


class PhaseTracker:
    """Wraps each phase to record its starting, peak and ending allocation and its time."""

    def __init__(self, model: OplmForFolding) -> None:
        self.phases: dict[str, dict[str, float]] = {}
        self.current: str | None = None
        self.peak = 0
        for mod_name, meth in PHASES:
            mod = getattr(model, mod_name, None)
            if mod is not None and hasattr(mod, meth):
                setattr(mod, meth, self._wrap(f"{mod_name}.{meth}", getattr(mod, meth)))

    def reset(self) -> None:
        self.phases, self.current, self.peak = {}, None, 0
        torch.cuda.reset_peak_memory_stats()

    def _wrap(self, name: str, fn: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            # the peak since the previous phase covers unwrapped glue (distogram, casts)
            self.peak = max(self.peak, torch.cuda.max_memory_allocated())
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self.current, t0 = name, time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            self.peak = max(self.peak, peak)
            self.phases[name] = {
                "start_gib": start / GIB,
                "peak_gib": peak / GIB,
                "end_gib": torch.cuda.memory_allocated() / GIB,
                "seconds": time.perf_counter() - t0,
            }
            self.current = None
            return out

        return wrapped


def run_case(
    model: OplmForFolding, tracker: PhaseTracker, length: int, samples: int, loops: int, steps: int
) -> dict[str, Any]:
    """One (length, samples) case: warmup run, then the measured run."""
    cfg = model.config
    pad = -(-length // 128) * 128
    rec: dict[str, Any] = {
        "width": cfg.pair_width,
        "length": length,
        "padded": pad,
        "samples": samples,
        "loops": loops,
        "steps": steps,
        "pair_unit_gib": pad * pad * cfg.pair_width * 2 / GIB,
    }
    # Each case compiles new FlexAttention shapes; past dynamo's recompile limit flex silently
    # runs eagerly with full score matrices, so start every case like a fresh process.
    torch._dynamo.reset()
    try:
        seq = "".join(AMINO[i] for i in torch.randint(0, len(AMINO), (length,)).tolist())
        feats = featurize([ChainSpec(seq, "A")], pad_tokens_to=pad)
        shape = (1, pad, cfg.lm_num_hidden_states, cfg.lm_hidden_size)
        mask = feats.token_mask.to("cuda", torch.bfloat16)[..., None, None]
        lm = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * mask
        for _ in range(2):
            tracker.reset()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(
                    feats,
                    lm_hidden_states=lm,
                    num_loops=loops,
                    num_samples=samples,
                    num_steps=steps,
                    generator=torch.Generator(device="cuda").manual_seed(0),
                )
            torch.cuda.synchronize()
        tracker.peak = max(tracker.peak, torch.cuda.max_memory_allocated())
        rec.update(
            status="ok",
            seconds=time.perf_counter() - t0,
            peak_allocated_gib=tracker.peak / GIB,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
        )
    except torch.cuda.OutOfMemoryError as exc:
        rec.update(status="oom", phase=tracker.current or "between phases", error=str(exc)[:200])
    rec["phases"] = dict(tracker.phases)
    tracker.current = None
    torch.cuda.empty_cache()
    return rec


def environment() -> dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = None
    try:
        from importlib.metadata import version

        cueq = version("cuequivariance-torch")
    except Exception:  # noqa: BLE001  # optional dependency
        cueq = None
    return {
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "total_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
        "torch": torch.__version__,
        "cuequivariance_torch": cueq,
        "git": rev,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--widths", default="128,256")
    p.add_argument("--samples", default="1,5")
    p.add_argument("--lengths", default="512,1024,1536,2048,3072,4096,5120,6144,7168,8192")
    p.add_argument("--trunk-blocks", type=int, default=24)
    args = p.parse_args()
    report: dict[str, Any] = {"env": environment(), "params_gib": {}, "cases": []}

    def emit(rec: dict[str, Any]) -> None:
        report["cases"].append(rec)
        keys = ("width", "samples", "length", "loops", "status", "phase", "peak_allocated_gib")
        print(json.dumps({k: rec[k] for k in keys if k in rec}), flush=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)

    for width in map(int, args.widths.split(",")):
        cfg = FoldConfig(
            pair_width=width,
            trunk_blocks=args.trunk_blocks,
            lm_hidden_size=1152,  # ESM-C 600M: 36 blocks -> 37 hidden states
            lm_num_hidden_states=37,
        )
        model = OplmForFolding(cfg).to("cuda").eval()
        torch.cuda.synchronize()
        report["params_gib"][width] = torch.cuda.memory_allocated() / GIB
        tracker = PhaseTracker(model)
        for loops, steps in ((cfg.inference_num_loops, cfg.inference_num_steps), (2, 2)):
            emit({**run_case(model, tracker, 1024, 1, loops, steps), "tag": "loops-check"})
        for samples in map(int, args.samples.split(",")):
            for length in map(int, args.lengths.split(",")):
                rec = run_case(model, tracker, length, samples, 2, 2)
                emit(rec)
                if rec["status"] != "ok":
                    break
        del model, tracker
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
