"""Golden-fixture tooling for the ESMFold2 parity oracle (spec §9).

Two halves. Pure-torch helpers that run anywhere: the released-config -> :class:`FoldConfig`
map, head-only weight extraction from the HF shards, fixture loading, LM-row splitting. And
the generator, which imports the upstream ``esm`` package lazily and runs only inside the
pinned fixture venv (docs/fold/b200-fixtures.sbatch): it featurizes each case with upstream's
own input builder, runs the upstream model on CPU in fp32 with dropout, masking and kernels
off and the loop count and sampler pinned, records every stage through forward hooks, and
writes one safetensors file per case plus a manifest. Fixture tensors keep upstream's dtype.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, safe_open, save_file

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.ccd import PROTEIN_RES_TYPES, ReferenceConformers
from oplm.fold.data.featurize import ChainSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = [
    "FIXTURE_CASES",
    "UPSTREAM_REPO",
    "UPSTREAM_REVISION",
    "FixtureCase",
    "extract_head_weights",
    "fold_config_from_upstream",
    "generate_fixtures",
    "load_fixture",
    "upstream_lm_rows",
]

UPSTREAM_REPO = "biohub/ESMFold2-Fast"
UPSTREAM_REVISION = "45fe8656f5b3ef493c17fcf9abe9a2968902e712"


@dataclass(frozen=True)
class FixtureCase:
    """One parity example: short enough for a CPU run of the 6B LM and the 24-block trunk."""

    name: str
    chains: tuple[ChainSpec, ...]
    num_loops: int = 2  # upstream runs num_loops + 1 = 3 iterations (spec §9: states after 1/2/3)
    num_steps: int = 3  # sampler steps before the cap; the denoiser oracle is the first call


FIXTURE_CASES: tuple[FixtureCase, ...] = (
    FixtureCase("trp_cage", (ChainSpec("NLYIQWLKDGGPSSGRPPPS", "A"),)),
    FixtureCase("villin", (ChainSpec("LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", "A"),)),
    FixtureCase(
        "gb1", (ChainSpec("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "A"),)
    ),
    FixtureCase("heterodimer", (ChainSpec("GSHMKTAYIAKQRQ", "A"), ChainSpec("WKLLSDEDFKAV", "B"))),
    FixtureCase("homodimer_x", (ChainSpec("MKVLAXGG", "A", copies=2),)),
)


def fold_config_from_upstream(config: dict[str, Any], **overrides: Any) -> FoldConfig:
    """Map the released ESMFold2 HF ``config.json`` onto :class:`FoldConfig` fields.

    ``inference_num_loops`` is ``num_loops + 1`` (iterations executed); ``trunk_dropout`` is 0
    (upstream's block dropout is ``DropoutResidual(0.0)``; its ``folding_trunk_dropout`` key is
    never read); ``inference_num_samples`` is 1 (the ``fold()`` default, not the forward
    default of 32). ``lm_name_or_path`` records the bundled ESMC as ``<repo>#esmc`` so a load
    never silently attaches an OPLM to a head trained against ESMC.

    Raises:
        ValueError: The derived ``single_inputs_width``/``atom_feature_dim`` disagree with the
            released ``single_inputs_size``/``atom_feature_dim``.
    """
    ae, sh, ch = config["atom_encoder"], config["structure_head"], config["confidence_head"]
    dm = sh["diffusion_module"]
    pair = int(config["pairwise_hidden_size"])
    fields: dict[str, Any] = dict(
        lm_name_or_path=f"{UPSTREAM_REPO}#esmc",
        lm_hidden_size=config["lm_d_model"],
        lm_num_hidden_states=config["lm_num_layers"] + 1,
        lm_input_mask_fraction=config["lm_mask_pct"],
        num_res_types=config["num_res_types"],
        max_atomic_number=config["max_atomic_number"],
        atom_name_chars=config["max_chars"],
        atom_name_vocab=config["char_vocab_size"],
        max_atoms_per_token=config["max_atoms_per_token"],
        pair_width=pair,
        token_width=ae["token_hidden_size"],
        atom_width=ae["hidden_size"],
        atom_encoder_blocks=ae["num_hidden_layers"],
        atom_encoder_heads=ae["num_attention_heads"],
        atom_window=config["sliding_window"],
        spatial_rope_base=ae["spatial_rope_base_frequency"],
        spatial_rope_pairs_per_axis=ae["num_spatial_rope_pairs_per_axis"],
        uid_rope_pairs=ae["num_uid_rope_pairs"],
        uid_rope_base=ae["uid_rope_base_frequency"],
        relpos_r_max=config["num_relative_residx_bins"],
        relpos_s_max=config["num_relative_chain_bins"],
        trunk_blocks=config["folding_trunk_num_hidden_layers"],
        lm_encoder_blocks=config["lm_encoder"]["num_hidden_layers"],
        coda_blocks=config["parcae_num_coda_layers"],
        transition_expansion=config["pair_transition_intermediate_size"] // pair,
        pair_dropout=config["lm_encoder"]["lm_dropout"],
        trunk_dropout=0.0,
        recurrence_poisson_mean=config["parcae"]["poisson_mean"],
        recurrence_min_loops=config["parcae"]["min_steps"],
        recurrence_max_loops=config["parcae"]["max_steps"],
        inference_num_loops=config["num_loops"] + 1,
        sigma_data=dm["sigma_data"],
        fourier_dim=dm["fourier_dim"],
        diffusion_blocks=dm["token_num_blocks"],
        diffusion_heads=dm["token_num_heads"],
        diffusion_transition_multiplier=dm["transition_multiplier"],
        diffusion_atom_blocks=dm["atom_encoder"]["num_hidden_layers"],
        diffusion_atom_heads=dm["atom_encoder"]["num_attention_heads"],
        inference_num_steps=sh["inference_num_steps"],
        inference_sigma_max=sh["inference_sigma_max_ratio"],
        inference_sigma_min=sh["inference_sigma_min_ratio"],
        inference_rho=sh["inference_exponent"],
        inference_sigma_cap=sh["inference_sigma_cap"],
        gamma_0=sh["gamma_0"],
        gamma_min=sh["gamma_min"],
        noise_scale=sh["noise_scale"],
        step_scale=sh["step_scale"],
        inference_num_samples=1,
        distogram_bins=sh["num_distogram_bins"],
        confidence_blocks=ch["num_hidden_layers"],
        plddt_bins=ch["num_plddt_bins"],
        pae_bins=ch["num_pae_bins"],
        pde_bins=ch["num_pde_bins"],
        confidence_dist_bins=ch["distogram_bins"],
        confidence_min_dist=ch["min_dist"],
        confidence_max_dist=ch["max_dist"],
    )
    fields.update(overrides)
    cfg = FoldConfig(**fields)
    if cfg.single_inputs_width != config["single_inputs_size"]:
        raise ValueError(
            f"single_inputs_width {cfg.single_inputs_width} != "
            f"released {config['single_inputs_size']}"
        )
    if cfg.atom_feature_dim != config["atom_feature_dim"]:
        raise ValueError(
            f"atom_feature_dim {cfg.atom_feature_dim} != released {config['atom_feature_dim']}"
        )
    return cfg


def extract_head_weights(snapshot_dir: Path, out: Path) -> Path:
    """Write every non-``esmc.*`` tensor of a sharded HF snapshot to one safetensors file."""
    index = json.loads((Path(snapshot_dir) / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        if not key.startswith("esmc."):
            by_shard.setdefault(shard, []).append(key)
    tensors: dict[str, Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(Path(snapshot_dir) / shard), "pt") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    save_file(tensors, str(out), metadata={"source": f"{UPSTREAM_REPO}@{UPSTREAM_REVISION}"})
    return Path(out)


def load_fixture(fixtures_dir: Path, case_name: str) -> dict[str, Tensor]:
    """Load one case's recorded tensors."""
    return load_file(str(Path(fixtures_dir) / f"{case_name}.safetensors"))


def upstream_lm_rows(input_ids: Tensor, *, bos: int, eos: int, pad: int) -> list[list[int]]:
    """Split upstream's packed ``[BOS c1 EOS BOS c2 EOS PAD...]`` LM row into per-chain rows."""
    rows: list[list[int]] = []
    for tok in input_ids[0].tolist():
        if tok == pad:
            break
        if tok == bos:
            rows.append([tok])
        else:
            rows[-1].append(tok)
    return rows


# --- generator (runs only in the esm venv) ---------------------------------------------------


class _Recorder:
    """Forward hooks that stash tensors under ``<name>.in.<k>`` / ``<name>.out.<k>`` per call."""

    def __init__(self) -> None:
        self.tensors: dict[str, Tensor] = {}
        self.calls: dict[str, int] = {}
        self.handles: list[Any] = []

    def remove_all(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _put(self, key: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value.detach().clone().cpu()

    def attach(
        self, module: torch.nn.Module, name: str, *, inputs: bool = False, kwargs: bool = False
    ) -> None:
        def hook(_m: Any, args: Any, kw: Any = None, output: Any = None) -> None:
            if output is None:  # with_kwargs=False signature: (module, args, output)
                output, kw = kw, {}
            i = self.calls.get(name, 0)
            self.calls[name] = i + 1
            if inputs:
                for j, a in enumerate(args):
                    self._put(f"{name}.in.{i}.arg{j}", a)
                for k, v in (kw or {}).items():
                    self._put(f"{name}.in.{i}.{k}", v)
            if isinstance(output, dict):
                for k, v in output.items():
                    self._put(f"{name}.out.{i}.{k}", v)
            elif isinstance(output, tuple):
                for j, v in enumerate(output):
                    self._put(f"{name}.out.{i}.{j}", v)
            else:
                self._put(f"{name}.out.{i}", output)

        self.handles.append(module.register_forward_hook(hook, with_kwargs=kwargs))


def _dump_reference_conformers(out: Path, base: ReferenceConformers) -> None:
    """Our residue/atom tables with positions from upstream's ``get_idealized_atom_pos``."""
    from esm.models.esmfold2.conformers import get_idealized_atom_pos  # ty: ignore[unresolved-import]  # esm venv only

    residues = {}
    for code in base.codes:
        t = base[code]
        positions = []
        for name in t.atoms:
            pos = get_idealized_atom_pos(PROTEIN_RES_TYPES[code], name)
            positions.append([0.0, 0.0, 0.0] if pos is None else [float(v) for v in pos])
        residues[code] = {
            "atoms": list(t.atoms),
            "elements": list(t.elements),
            "charges": list(t.charges),
            "positions": positions,
        }
    out.write_text(
        json.dumps(
            {
                "source": f"esm {version('esm')} get_idealized_atom_pos over ccd.pkl of "
                f"{UPSTREAM_REPO}@{UPSTREAM_REVISION} (Computed conformer, raw)",
                "charge_table": base.source,
                "residues": residues,
            },
            indent=1,
        )
        + "\n"
    )


def generate_fixtures(
    out_dir: Path,
    *,
    repo: str = UPSTREAM_REPO,
    revision: str = UPSTREAM_REVISION,
    cases: Sequence[FixtureCase] = FIXTURE_CASES,
    seed: int = 0,
) -> Path:
    """Run upstream ESMFold2 on CPU in fp32 for each case and record the parity oracle.

    Requires the ``esm`` package (``esm==3.4.1.post1``) and network/cache access to ``repo``.
    Determinism: ``set_kernel_backend(None)``, ``set_chunk_size(None)``, LM dropout off
    (``_lm_dropout_context(model, None)``), ``lm_mask_pct=0``, ``torch.manual_seed(seed)``
    before every forward; the initial pair state is captured by wrapping ``_init_pair_state``.
    """
    from esm.models.esmfold2 import (  # ty: ignore[unresolved-import]  # esm venv only
        ESMFold2InputBuilder,
        EsmFold2Model,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.models.esmfold2.processor import _lm_dropout_context  # ty: ignore[unresolved-import]  # esm venv only
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.pkl"]
        )
    )
    config = json.loads((snapshot / "config.json").read_text())
    config.pop("esmc_config", None)
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    extract_head_weights(snapshot, out_dir / "head.safetensors")

    torch.set_default_dtype(torch.float32)
    # If this signature rejects `revision`, pass the `snapshot` path instead (same files).
    model = EsmFold2Model.from_pretrained(repo, revision=revision, device="cpu").eval().float()
    model.set_kernel_backend(None)
    model.set_chunk_size(None)
    builder = ESMFold2InputBuilder()
    _dump_reference_conformers(out_dir / "reference_conformers.json", ReferenceConformers.load())

    manifest: dict[str, Any] = {
        "repo": repo,
        "revision": revision,
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "esm": version("esm"),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cases": {},
    }
    for case in cases:
        rec = _Recorder()
        rec.attach(model.inputs_embedder, "inputs_embedder")
        for name in (
            "z_init_1",
            "z_init_2",
            "rel_pos",
            "token_bonds",
            "language_model",
            "lm_encoder",
            "folding_trunk",
            "parcae_input_norm",
            "parcae_readout",
            "parcae_coda",
            "distogram_head",
        ):
            rec.attach(getattr(model, name), name, inputs=(name == "language_model"))
        rec.attach(model.structure_head.diffusion_module, "diffusion", inputs=True, kwargs=True)
        rec.attach(model.confidence_head.folding_trunk, "confidence_trunk", inputs=True)
        original_init = model._init_pair_state

        def init_pair_state(ref: Tensor) -> Tensor:
            z0 = original_init(ref)  # noqa: B023  # only called inside this iteration's forward
            rec.tensors["z0"] = z0.detach().clone().cpu()  # noqa: B023  # ditto
            return z0

        model._init_pair_state = init_pair_state  # instance override
        spi = StructurePredictionInput(
            sequences=[
                ProteinInput(id=f"{spec.chain_id}{'' if k == 0 else k + 1}", sequence=spec.sequence)
                for spec in case.chains
                for k in range(spec.copies)
            ]
        )
        features, _chain_infos = builder.prepare_input(spi, seed=seed, device="cpu")
        torch.manual_seed(seed)
        with torch.no_grad(), _lm_dropout_context(model, None):
            output = model(
                **features,
                num_loops=case.num_loops,
                num_diffusion_samples=1,
                num_sampling_steps=case.num_steps,
                lm_mask_pct=0.0,
            )
        model._init_pair_state = original_init  # restore
        tensors = {
            f"features.{k}": v.cpu() for k, v in features.items() if isinstance(v, torch.Tensor)
        }
        tensors.update(
            {
                f"output.{k}": v.detach().cpu()
                for k, v in output.items()
                if isinstance(v, torch.Tensor)
            }
        )
        tensors.update(rec.tensors)
        # Clone: no shared storage with the recorded language_model input.
        tensors["lm_hidden_states"] = rec.tensors["language_model.in.0.arg0"].clone()
        # Hooks ran under upstream's @inference_mode; clone outside it so safetensors gets plain
        # tensors.
        save_file(
            {k: v.detach().clone().contiguous() for k, v in tensors.items()},
            str(out_dir / f"{case.name}.safetensors"),
        )
        manifest["cases"][case.name] = {
            "chains": [[c.sequence, c.chain_id, c.copies] for c in case.chains],
            "num_loops": case.num_loops,
            "num_steps": case.num_steps,
            "keys": sorted(tensors),
            "denoiser_calls": rec.calls.get("diffusion", 0),
        }
        rec.remove_all()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return out_dir
