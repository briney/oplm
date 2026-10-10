"""Stage-by-stage parity against the recorded upstream ESMFold2-Fast run (spec §9, §10 M1).

CPU, fp32, dense attention, reference trimul: the same arithmetic upstream used on its CPU
run. Tolerances are initial; Task 12 records the observed maxima in docs/FOLD.md §7 and
tightens them to <= 10x observed.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from safetensors.torch import load_file

from oplm.fold.data.ccd import ReferenceConformers
from oplm.fold.data.featurize import featurize
from oplm.fold.fixtures import (
    FIXTURE_CASES,
    fold_config_from_upstream,
    load_fixture,
    upstream_lm_rows,
)
from oplm.fold.modeling_fold import OplmForFolding

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from oplm.fold.fixtures import FixtureCase

_CASES = [pytest.param(c, id=c.name) for c in FIXTURE_CASES]


@functools.cache
def _manifest(fixtures_dir: Path) -> dict[str, Any]:
    return json.loads((fixtures_dir / "manifest.json").read_text())


def _case_fixture(fixtures_dir: Path, case: FixtureCase) -> dict[str, Tensor]:
    """Load ``case``'s recorded tensors; skip it when an ``OPLM_CASES`` subset left it out."""
    if case.name not in _manifest(fixtures_dir)["cases"]:
        pytest.skip(f"case {case.name} not in the fixture manifest")
    return load_fixture(fixtures_dir, case.name)


@pytest.fixture(scope="module")
def released(fixtures_dir: Path) -> OplmForFolding:
    cfg = fold_config_from_upstream(
        json.loads((fixtures_dir / "config.json").read_text()),
        attention_backend="dense",
        trimul_backend="reference",
        trimul_chunk_size=None,
    )
    model = OplmForFolding(cfg).eval()
    missing, unexpected = model.load_state_dict(
        load_file(str(fixtures_dir / "head.safetensors")), strict=True
    )
    assert not missing and not unexpected
    return model.float()


@pytest.fixture(scope="module")
def conformers(fixtures_dir: Path) -> ReferenceConformers:
    return ReferenceConformers.load(fixtures_dir / "reference_conformers.json")


def _close(a: Tensor, b: Tensor, *, atol: float, rtol: float = 0.0, what: str = "") -> None:
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item()
    torch.testing.assert_close(
        a, b, atol=atol, rtol=rtol, msg=f"{what}: max abs err {err:.3e} (atol {atol})"
    )


def test_head_weights_load_strictly_and_config_matches(released: OplmForFolding) -> None:
    assert released.config.trunk_blocks == 24 and released.config.distogram_bins == 64
    assert len(released.state_dict()) == 1054


@pytest.mark.parametrize("case", _CASES)
def test_featurizer_matches_upstream(
    fixtures_dir: Path, conformers: ReferenceConformers, case
) -> None:
    fx = _case_fixture(fixtures_dir, case)
    f = featurize(case.chains, conformers=conformers)
    for ours, theirs in [
        ("token_index", "token_index"),
        ("residue_index", "residue_index"),
        ("asym_id", "asym_id"),
        ("sym_id", "sym_id"),
        ("entity_id", "entity_id"),
        ("mol_type", "mol_type"),
        ("res_type", "res_type"),
        ("token_mask", "token_attention_mask"),
        ("ref_element", "ref_element"),
        ("ref_space_uid", "ref_space_uid"),
        ("atom_mask", "atom_attention_mask"),
        ("atom_to_token", "atom_to_token"),
        ("distogram_atom_idx", "distogram_atom_idx"),
        ("ref_atom_name_chars", "ref_atom_name_chars"),
    ]:
        assert torch.equal(getattr(f, ours).long(), fx[f"features.{theirs}"].long()), ours
    assert torch.equal(f.token_bonds.bool(), fx["features.token_bonds"].bool())
    _close(f.ref_charge, fx["features.ref_charge"], atol=0, what="ref_charge")
    _close(f.ref_pos, fx["features.ref_pos"], atol=1e-6, what="ref_pos")
    rows = upstream_lm_rows(fx["features.input_ids"], bos=0, eos=2, pad=1)
    ours_rows = [[t for t in row if t != 1] for row in f.lm_input_ids.tolist()]
    assert ours_rows == rows


@pytest.mark.parametrize("case", _CASES)
def test_shim_matches(fixtures_dir: Path, released: OplmForFolding, case) -> None:
    fx = _case_fixture(fixtures_dir, case)
    with torch.no_grad():
        lm_z = released.language_model(fx["lm_hidden_states"].float())
    _close(lm_z, fx["language_model.out.0"], atol=1e-4, rtol=1e-4, what="lm_z")


@pytest.mark.parametrize("case", _CASES)
def test_inputs_embedder_matches(
    fixtures_dir: Path, released: OplmForFolding, conformers, case
) -> None:
    fx = _case_fixture(fixtures_dir, case)
    f = featurize(case.chains, conformers=conformers)
    with torch.no_grad():
        emb = released.input_embedder(f)
    _close(emb.s_inputs, fx["inputs_embedder.out.0"], atol=1e-4, rtol=1e-4, what="x_inputs")
    _close(emb.relpos, fx["rel_pos.out.0"], atol=1e-5, what="relpos")
    _close(emb.bonds, fx["token_bonds.out.0"], atol=1e-5, what="bonds")
    z_init = (
        fx["z_init_1.out.0"][:, :, None]
        + fx["z_init_2.out.0"][:, None]
        + fx["rel_pos.out.0"]
        + fx["token_bonds.out.0"]
    )
    _close(emb.z_init, z_init, atol=1e-4, rtol=1e-4, what="z_init")


@pytest.mark.parametrize("case", _CASES)
def test_recurrence_readout_and_distogram_match(
    fixtures_dir: Path, released: OplmForFolding, conformers, case
) -> None:
    fx = _case_fixture(fixtures_dir, case)
    f = featurize(case.chains, conformers=conformers)
    pair_mask = f.token_mask[:, :, None].float() * f.token_mask[:, None, :].float()
    z_init, lm_z, z0 = (
        fx["z_init_1.out.0"][:, :, None]
        + fx["z_init_2.out.0"][:, None]
        + fx["rel_pos.out.0"]
        + fx["token_bonds.out.0"],
        fx["language_model.out.0"].float(),
        fx["z0"].float(),
    )
    with torch.no_grad():
        refined = released.lm_encoder(lm_z, pair_mask)
        _close(refined, fx["lm_encoder.out.0"], atol=1e-3, rtol=1e-3, what="lm_encoder")
        z, states = released.parcae.run(
            released.folding_trunk,
            lambda _t: z_init.float() + refined,
            z0=z0,
            pair_mask=pair_mask,
            num_loops=case.num_loops + 1,
            return_states=True,
        )
        assert len(states) == case.num_loops + 1
        for i, state in enumerate(states):
            _close(state, fx[f"folding_trunk.out.{i}"], atol=2e-3, rtol=2e-3, what=f"state {i + 1}")
        readout = released.parcae.readout(z, pair_mask)
        _close(
            released.parcae.out_proj(z),
            fx["parcae_readout.out.0"],
            atol=2e-3,
            rtol=2e-3,
            what="readout",
        )
        _close(readout, fx["parcae_coda.out.0"], atol=3e-3, rtol=3e-3, what="coda")
        logits = released.distogram_head(readout + readout.transpose(1, 2))
    _close(logits, fx["distogram_head.out.0"], atol=5e-3, rtol=3e-3, what="distogram")


@pytest.mark.parametrize("case", _CASES)
def test_denoiser_matches(fixtures_dir: Path, released: OplmForFolding, conformers, case) -> None:
    fx = _case_fixture(fixtures_dir, case)
    f = featurize(case.chains, conformers=conformers)
    with torch.no_grad():
        emb = released.input_embedder(f)
        inp = released.structure_head.prepare(
            s_inputs=fx["inputs_embedder.out.0"].float(),
            z_trunk=fx["parcae_coda.out.0"].float(),
            relpos=fx["rel_pos.out.0"].float(),
            atom_features=emb.atom_features,
            rope=emb.rope,
            atom_mask=f.atom_mask,
            atom_to_token=f.atom_to_token,
            token_mask=f.token_mask,
            num_samples=1,
        )
        x_denoised = released.structure_head.denoise(
            fx["diffusion.in.0.x_noisy"].float(), fx["diffusion.in.0.t_hat"].float(), inp
        )
    _close(x_denoised, fx["diffusion.out.0.x_denoised"], atol=5e-2, what="x_denoised (Å)")


@pytest.mark.parametrize("case", _CASES)
def test_sampler_schedule_matches_recorded_t_hat(
    fixtures_dir: Path, released: OplmForFolding, case
) -> None:
    fx = _case_fixture(fixtures_dir, case)
    calls = _manifest(fixtures_dir)["cases"][case.name]["denoiser_calls"]
    sched = released.structure_head.noise_schedule(case.num_steps, torch.device("cpu"))
    cap = released.config.inference_sigma_cap
    sched = torch.nn.functional.pad(sched[sched <= cap], (1, 0), value=cap).tolist()
    gammas = [released.config.gamma_0 if s > released.config.gamma_min else 0.0 for s in sched]
    t_hats = [s * (1 + g) for s, g in zip(sched[:-1], gammas[1:], strict=True)]
    assert len(t_hats) == calls
    for i, t in enumerate(t_hats):
        assert abs(fx[f"diffusion.in.{i}.t_hat"][0].item() - t) < 1e-3 * max(t, 1.0), i


@pytest.mark.parametrize("case", _CASES)
def test_confidence_matches(fixtures_dir: Path, released: OplmForFolding, conformers, case) -> None:
    fx = _case_fixture(fixtures_dir, case)
    f = featurize(case.chains, conformers=conformers)
    with torch.no_grad():
        out = released.confidence_head(
            s_inputs=fx["inputs_embedder.out.0"].float(),
            z=fx["parcae_coda.out.0"].float(),
            relpos=fx["rel_pos.out.0"].float(),
            bonds=fx["token_bonds.out.0"].float(),
            coords=fx["output.sample_atom_coords"].float(),
            distogram_atom_idx=f.distogram_atom_idx,
            token_mask=f.token_mask,
            atom_to_token=f.atom_to_token,
            atom_mask=f.atom_mask,
            asym_id=f.asym_id,
        )
    _close(out.pae_logits, fx["output.pae_logits"], atol=3e-3, rtol=3e-3, what="pae_logits")
    _close(out.pde_logits, fx["output.pde_logits"], atol=3e-3, rtol=3e-3, what="pde_logits")
    _close(out.plddt_logits, fx["output.plddt_logits"], atol=3e-3, rtol=3e-3, what="plddt_logits")
    _close(out.plddt, fx["output.plddt"], atol=1e-3, what="plddt")
    _close(out.ptm, fx["output.ptm"], atol=1e-3, what="ptm")
    _close(out.iptm, fx["output.iptm"], atol=1e-3, what="iptm")
    _close(out.pair_chains_iptm, fx["output.pair_chains_iptm"], atol=1e-3, what="pair_chains_iptm")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _CASES[:1])
def test_gpu_pipeline_tracks_the_cpu_oracle(fixtures_dir: Path, conformers, case) -> None:
    """bf16 autocast + FlexAttention + fused trimul: finite end to end, pair within bf16 drift."""
    fx = _case_fixture(fixtures_dir, case)
    cfg = fold_config_from_upstream(json.loads((fixtures_dir / "config.json").read_text()))
    model = OplmForFolding(cfg).eval().cuda()
    model.load_state_dict(load_file(str(fixtures_dir / "head.safetensors")), strict=True)
    f = featurize(case.chains, conformers=conformers, pad_tokens_to=128)
    hs = torch.zeros(1, 128, cfg.lm_num_hidden_states, cfg.lm_hidden_size)
    L = fx["lm_hidden_states"].shape[1]
    hs[:, :L] = fx["lm_hidden_states"].float()
    z0 = torch.zeros(1, 128, 128, cfg.pair_width)
    z0[:, :L, :L] = fx["z0"].float()
    with torch.no_grad():
        out = model(
            f,
            lm_hidden_states=hs.cuda(),
            z0=z0.cuda(),
            num_loops=case.num_loops + 1,
            num_steps=case.num_steps,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
    assert torch.isfinite(out.coords).all() and torch.isfinite(out.confidence.pae).all()
    ref = fx["parcae_coda.out.0"].float().cuda()
    ours = out.pair[:, :L, :L]
    rel = ((ours - ref).norm() / ref.norm()).item()
    assert rel < 5e-2, f"relative Frobenius error of the final pair: {rel:.3e}"
