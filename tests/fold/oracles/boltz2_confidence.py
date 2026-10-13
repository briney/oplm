# ruff: noqa
# fmt: off
# Vendored from boltz==2.2.1 (https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024
# Jeremy Wohlwend, Gabriele Corso, Saro Passaro): boltz/model/loss/confidencev2.py and boltz/model/layers/confidence_utils.py. Modifications are listed
# in tests/fold/oracles/README.md. Test oracle only; never imported from src/.

import torch
from torch import nn

# boltz/data/const.py: chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}
CHAIN_TYPE_IDS = {name: i for i, name in enumerate(["PROTEIN", "DNA", "RNA", "NONPOLYMER"])}


# ---- from boltz/model/layers/confidence_utils.py


def compute_collinear_mask(v1, v2):
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity,
    resolved_mask=None,
    inference=False,
):
    with torch.amp.autocast("cuda", enabled=False):
        asym_id_token = feats["asym_id"]
        asym_id_atom = torch.bmm(
            feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()
        ).squeeze(-1)

    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    frames_idx_pred = (
        frames_idx_true.clone()
        .repeat_interleave(multiplicity, 0)
        .reshape(B // multiplicity, multiplicity, -1, 3)
    )

    # Iterate through the batch and modify the frames for nonpolymers
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if (
                feats["mol_type"][i, token_idx] != CHAIN_TYPE_IDS["NONPOLYMER"]
                or num_atoms < 3
            ):
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5
            if inference:
                resolved_pair = 1 - (
                    feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                    * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            else:
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                resolved_pair = 1 - (
                    resolved_mask[i][mask_chain_atom.bool()][None, :]
                    * resolved_mask[i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            try:
                frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            except Exception as e:
                print(f"Failed to process {feats['pdb_id']} due to {e}")
            token_idx += num_tokens
            atom_idx += num_atoms

    frames_expanded = pred_atom_coords[
        torch.arange(0, B // multiplicity, 1)[:, None, None, None].to(
            frames_idx_pred.device
        ),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(
            frames_idx_pred.device
        ),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Compute masks for collinearity / overlap
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B // multiplicity, multiplicity, -1)
    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]


# ---- from boltz/model/loss/confidencev2.py


def resolved_loss(
    pred_resolved,
    feats,
    true_coords_resolved_mask,
    token_level_confidence=False,
    multiplicity=1,
    mask_loss=None,
):
    with torch.autocast("cuda", enabled=False):
        if token_level_confidence:
            token_to_rep_atom = feats["token_to_rep_atom"]
            token_to_rep_atom = token_to_rep_atom.repeat_interleave(
                multiplicity, 0
            ).float()
            ref_mask = torch.bmm(
                token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
            ).squeeze(-1)

            pad_mask = feats["token_pad_mask"]
            pad_mask = pad_mask.repeat_interleave(multiplicity, 0).float()
        else:
            ref_mask = true_coords_resolved_mask.float()
            pad_mask = feats["atom_pad_mask"]
            pad_mask = pad_mask.repeat_interleave(multiplicity, 0).float()
        # compute loss
        log_softmax_resolved = torch.nn.functional.log_softmax(
            pred_resolved.float(), dim=-1
        )
        errors = (
            -ref_mask * log_softmax_resolved[:, :, 0]
            - (1 - ref_mask) * log_softmax_resolved[:, :, 1]
        )
        loss = torch.sum(errors * pad_mask, dim=-1) / (
            1e-7 + torch.sum(pad_mask, dim=-1)
        )

        # Average over the batch dimension
        if mask_loss is not None:
            mask_loss = (
                mask_loss.repeat_interleave(multiplicity, 0)
                .reshape(-1, multiplicity)
                .float()
            )
            loss = torch.sum(loss.reshape(-1, multiplicity) * mask_loss) / (
                torch.sum(mask_loss) + 1e-7
            )
        else:
            loss = torch.mean(loss)
    return loss


def get_target_lddt(
    pred_atom_coords,
    feats,
    true_atom_coords,
    true_coords_resolved_mask,
    token_level_confidence=True,
    multiplicity=1,
):
    with torch.autocast("cuda", enabled=False):
        # extract necessary features
        atom_mask = true_coords_resolved_mask

        R_set_to_rep_atom = feats["r_set_to_rep_atom"]
        R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_nucleotide_token = (token_type == CHAIN_TYPE_IDS["DNA"]).float() + (
            token_type == CHAIN_TYPE_IDS["RNA"]
        ).float()

        B = true_atom_coords.shape[0]

        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"].float()
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

        if token_level_confidence:
            true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
            pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

        # compute true lddt
        true_d = torch.cdist(
            true_token_coords if token_level_confidence else true_atom_coords,
            torch.bmm(R_set_to_rep_atom, true_atom_coords),
        )
        pred_d = torch.cdist(
            pred_token_coords if token_level_confidence else pred_atom_coords,
            torch.bmm(R_set_to_rep_atom, pred_atom_coords),
        )

        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )
        pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)

        if token_level_confidence:
            pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
            atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float())
        is_nucleotide_R_element = torch.bmm(
            R_set_to_rep_atom,
            torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1).float()),
        ).squeeze(-1)
        cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
            1, true_d.shape[1], 1
        )
        target_lddt, mask_no_match = lddt_dist(
            pred_d, true_d, pair_mask, cutoff, per_atom=True
        )
        return target_lddt, mask_no_match, atom_mask


def plddt_loss(
    pred_lddt,
    pred_atom_coords,
    feats,
    true_atom_coords,
    true_coords_resolved_mask,
    token_level_confidence=False,
    multiplicity=1,
    mask_loss=None,
    relative_confidence_supervision=False,
    relative_pred_lddt=None,
):
    target_lddt, mask_no_match, atom_mask = get_target_lddt(
        pred_atom_coords=pred_atom_coords,
        feats=feats,
        true_atom_coords=true_atom_coords,
        true_coords_resolved_mask=true_coords_resolved_mask,
        token_level_confidence=token_level_confidence,
        multiplicity=multiplicity,
    )

    num_bins = pred_lddt.shape[-1]
    bin_index = torch.floor(target_lddt * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)
    errors = -1 * torch.sum(
        lddt_one_hot * torch.nn.functional.log_softmax(pred_lddt, dim=-1),
        dim=-1,
    )
    atom_mask = atom_mask.squeeze(-1)
    loss = torch.sum(errors * atom_mask * mask_no_match, dim=-1) / (
        1e-7 + torch.sum(atom_mask * mask_no_match, dim=-1)
    )
    # Average over the batch dimension
    if mask_loss is not None:
        mask_loss = mask_loss.repeat_interleave(multiplicity, 0).reshape(
            -1, multiplicity
        )
        loss = torch.sum(loss.reshape(-1, multiplicity) * mask_loss) / (
            torch.sum(mask_loss) + 1e-7
        )
    else:
        loss = torch.mean(loss)

    rel_loss = 0.0
    if relative_confidence_supervision:
        # relative LDDT loss
        B = true_atom_coords.shape[0]
        relative_target_lddt = target_lddt.view(
            B // multiplicity, multiplicity, 1, -1
        ) - target_lddt.view(B // multiplicity, 1, multiplicity, -1)
        rel_bin_index = torch.floor(
            torch.abs(relative_target_lddt) * num_bins
        ).long() * torch.sign(relative_target_lddt)
        rel_bin_index = torch.clamp(
            rel_bin_index, max=(num_bins - 1), min=-(num_bins - 1)
        ).long() + (num_bins - 1)
        rel_lddt_one_hot = nn.functional.one_hot(
            rel_bin_index, num_classes=2 * num_bins - 1
        )
        rel_errors = -1 * torch.sum(
            rel_lddt_one_hot
            * torch.nn.functional.log_softmax(relative_pred_lddt, dim=-1),
            dim=-1,
        )
        rel_atom_mask = atom_mask.view(B // multiplicity, multiplicity, 1, -1).repeat(
            1, 1, multiplicity, 1
        )
        rel_mask_no_match = mask_no_match.view(
            B // multiplicity, multiplicity, 1, -1
        ).repeat(1, 1, multiplicity, 1)
        rel_loss = torch.sum(rel_errors * rel_atom_mask * rel_mask_no_match, dim=-1) / (
            1e-7 + torch.sum(rel_atom_mask * rel_mask_no_match, dim=-1)
        )

        if mask_loss is not None:
            rel_mask_loss = mask_loss.view(B // multiplicity, multiplicity, 1).repeat(
                1, 1, multiplicity
            )
            rel_loss = torch.sum(rel_loss * rel_mask_loss) / (
                torch.sum(rel_mask_loss) + 1e-7
            )
        else:
            rel_loss = torch.mean(rel_loss)

    return loss, rel_loss


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    # NOTE: the mask is a pairwise mask which should have the identity elements already masked out
    # Compute mask over distances
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    if per_atom:
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    else:
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
        total = torch.sum(dists_to_score, dim=(-1, -2))
        return score, total


def express_coordinate_in_frame(atom_coords, frame_atom_a, frame_atom_b, frame_atom_c):
    batch, multiplicity = atom_coords.shape[0], atom_coords.shape[1]
    batch_indices0 = torch.arange(batch)[:, None, None].to(atom_coords.device)
    batch_indices1 = torch.arange(multiplicity)[None, :, None].to(atom_coords.device)

    # extract frame atoms
    a, b, c = (
        atom_coords[batch_indices0, batch_indices1, frame_atom_a],
        atom_coords[batch_indices0, batch_indices1, frame_atom_b],
        atom_coords[batch_indices0, batch_indices1, frame_atom_c],
    )
    w1 = (a - b) / (torch.norm(a - b, dim=-1, keepdim=True) + 1e-5)
    w2 = (c - b) / (torch.norm(c - b, dim=-1, keepdim=True) + 1e-5)

    # build orthogonal frame
    e1 = (w1 + w2) / (torch.norm(w1 + w2, dim=-1, keepdim=True) + 1e-5)
    e2 = (w2 - w1) / (torch.norm(w2 - w1, dim=-1, keepdim=True) + 1e-5)
    e3 = torch.linalg.cross(e1, e2)

    # NOTE: it is unclear based on what atom of the token the error is computed, here I will use the atom indicated by b (center of frame)

    # project onto frame basis
    d = b[:, :, None, :, :] - b[:, :, :, None, :]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[:, :, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[:, :, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[:, :, :, None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return x_transformed


def get_target_pae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    true_coords_resolved_mask,
    multiplicity=1,
):
    with torch.autocast("cuda", enabled=False):
        # Retrieve frames and resolved masks
        frames_idx_original = feats["frames_idx"]
        mask_frame_true = feats["frame_resolved_mask"]

        # Adjust the frames for nonpolymers after symmetry correction!
        # NOTE: frames of polymers do not change under symmetry!
        frames_idx_true, mask_collinear_true = compute_frame_pred(
            true_atom_coords,
            frames_idx_original,
            feats,
            multiplicity,
            resolved_mask=true_coords_resolved_mask,
        )

        frame_true_atom_a, frame_true_atom_b, frame_true_atom_c = (
            frames_idx_true[:, :, :, 0],
            frames_idx_true[:, :, :, 1],
            frames_idx_true[:, :, :, 2],
        )
        # Compute token coords in true frames
        B, N, _ = true_atom_coords.shape
        true_atom_coords = true_atom_coords.reshape(
            B // multiplicity, multiplicity, -1, 3
        )
        true_coords_transformed = express_coordinate_in_frame(
            true_atom_coords, frame_true_atom_a, frame_true_atom_b, frame_true_atom_c
        )

        # Compute pred frames and mask
        frames_idx_pred, mask_collinear_pred = compute_frame_pred(
            pred_atom_coords, frames_idx_original, feats, multiplicity
        )
        frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c = (
            frames_idx_pred[:, :, :, 0],
            frames_idx_pred[:, :, :, 1],
            frames_idx_pred[:, :, :, 2],
        )
        # Compute token coords in pred frames
        B, N, _ = pred_atom_coords.shape
        pred_atom_coords = pred_atom_coords.reshape(
            B // multiplicity, multiplicity, -1, 3
        )
        pred_coords_transformed = express_coordinate_in_frame(
            pred_atom_coords, frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c
        )

        target_pae = torch.sqrt(
            ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
        )

        # Compute mask for the pae loss
        b_true_resolved_mask = true_coords_resolved_mask[
            torch.arange(B // multiplicity)[:, None, None].to(
                pred_coords_transformed.device
            ),
            frame_true_atom_b,
        ]

        pair_mask = (
            mask_frame_true[:, None, :, None]  # if true frame is invalid
            * mask_collinear_true[:, :, :, None]  # if true frame is invalid
            * mask_collinear_pred[:, :, :, None]  # if pred frame is invalid
            * b_true_resolved_mask[:, :, None, :]  # If atom j is not resolved
            * feats["token_pad_mask"][:, None, :, None]
            * feats["token_pad_mask"][:, None, None, :]
        )
        return target_pae, pair_mask


def pae_loss(
    pred_pae,
    pred_atom_coords,
    feats,
    true_atom_coords,
    true_coords_resolved_mask,
    multiplicity=1,
    max_dist=32.0,
    mask_loss=None,
    relative_confidence_supervision=False,
    relative_pred_pae=None,
):
    target_pae, pair_mask = get_target_pae(
        pred_atom_coords=pred_atom_coords,
        feats=feats,
        true_atom_coords=true_atom_coords,
        true_coords_resolved_mask=true_coords_resolved_mask,
        multiplicity=multiplicity,
    )

    # compute loss
    num_bins = pred_pae.shape[-1]
    bin_index = torch.floor(target_pae * num_bins / max_dist).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    pae_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)
    errors = -1 * torch.sum(
        pae_one_hot
        * torch.nn.functional.log_softmax(pred_pae.reshape(pae_one_hot.shape), dim=-1),
        dim=-1,
    )
    loss = torch.sum(errors * pair_mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(pair_mask, dim=(-2, -1))
    )
    # Average over the batch dimension
    if mask_loss is not None:
        mask_loss = mask_loss.repeat_interleave(multiplicity, 0).reshape(
            -1, multiplicity
        )
        loss = torch.sum(loss.reshape(-1, multiplicity) * mask_loss) / (
            torch.sum(mask_loss) + 1e-7
        )
    else:
        loss = torch.mean(loss)

    rel_loss = 0.0
    if relative_confidence_supervision:
        B, N, _, _ = pred_pae.shape
        rel_target_pae = target_pae.view(
            B // multiplicity, multiplicity, 1, N, N
        ) - target_pae.view(B // multiplicity, 1, multiplicity, N, N)
        rel_bin_index = torch.floor(
            torch.abs(rel_target_pae) * num_bins / max_dist
        ).long() * torch.sign(rel_target_pae)
        rel_bin_index = torch.clamp(
            rel_bin_index, max=(num_bins - 1), min=-(num_bins - 1)
        ).long() + (num_bins - 1)
        rel_pae_one_hot = nn.functional.one_hot(
            rel_bin_index, num_classes=2 * num_bins - 1
        )
        rel_errors = -1 * torch.sum(
            rel_pae_one_hot
            * torch.nn.functional.log_softmax(relative_pred_pae, dim=-1),
            dim=-1,
        )
        rel_mask = pair_mask.view(B // multiplicity, multiplicity, 1, N, N).repeat(
            1, 1, multiplicity, 1, 1
        )
        rel_loss = torch.sum(rel_errors * rel_mask, dim=(-2, -1)) / (
            1e-7 + torch.sum(rel_mask, dim=(-2, -1))
        )

        if mask_loss is not None:
            rel_mask_loss = mask_loss.view(B // multiplicity, multiplicity, 1).repeat(
                1, 1, multiplicity
            )
            rel_loss = torch.sum(rel_loss * rel_mask_loss) / (
                torch.sum(rel_mask_loss) + 1e-7
            )
        else:
            rel_loss = torch.mean(rel_loss)

    return loss, rel_loss


def get_target_pde(
    pred_atom_coords,
    feats,
    true_atom_coords,
    true_coords_resolved_mask,
    multiplicity=1,
):
    with torch.autocast("cuda", enabled=False):
        # extract necessary features
        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0).float()
        token_mask = torch.bmm(
            token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
        ).squeeze(-1)
        mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)

        # compute true pde
        true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
        pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

        true_d = torch.cdist(true_token_coords, true_token_coords)
        pred_d = torch.cdist(pred_token_coords, pred_token_coords)
        target_pde = torch.abs(true_d - pred_d)
        return target_pde, mask


def pde_loss(
    pred_pde,
    pred_atom_coords,
    feats,
    true_atom_coords,
    true_coords_resolved_mask,
    multiplicity=1,
    max_dist=32.0,
    mask_loss=None,
    relative_confidence_supervision=False,
    relative_pred_pde=None,
):
    target_pde, mask = get_target_pde(
        pred_atom_coords=pred_atom_coords,
        feats=feats,
        true_atom_coords=true_atom_coords,
        true_coords_resolved_mask=true_coords_resolved_mask,
        multiplicity=multiplicity,
    )
    # compute loss
    num_bins = pred_pde.shape[-1]
    bin_index = torch.floor(target_pde * num_bins / max_dist).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    pde_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)
    errors = -1 * torch.sum(
        pde_one_hot * torch.nn.functional.log_softmax(pred_pde, dim=-1),
        dim=-1,
    )
    loss = torch.sum(errors * mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(mask, dim=(-2, -1))
    )
    # Average over the batch dimension
    if mask_loss is not None:
        mask_loss = mask_loss.repeat_interleave(multiplicity, 0).reshape(
            -1, multiplicity
        )
        loss = torch.sum(loss.reshape(-1, multiplicity) * mask_loss) / (
            torch.sum(mask_loss) + 1e-7
        )
    else:
        loss = torch.mean(loss)

    rel_loss = 0.0
    if relative_confidence_supervision:
        B, N = target_pde.shape[:2]
        rel_target_pde = target_pde.view(
            B // multiplicity, multiplicity, 1, N, N
        ) - target_pde.view(B // multiplicity, 1, multiplicity, N, N)
        rel_bin_index = torch.floor(
            torch.abs(rel_target_pde) * num_bins / max_dist
        ).long() * torch.sign(rel_target_pde)
        rel_bin_index = torch.clamp(
            rel_bin_index, max=(num_bins - 1), min=-(num_bins - 1)
        ).long() + (num_bins - 1)
        rel_pde_one_hot = nn.functional.one_hot(
            rel_bin_index, num_classes=2 * num_bins - 1
        )
        rel_errors = -1 * torch.sum(
            rel_pde_one_hot
            * torch.nn.functional.log_softmax(relative_pred_pde, dim=-1),
            dim=-1,
        )
        rel_mask = mask.view(B // multiplicity, multiplicity, 1, N, N).repeat(
            1, 1, multiplicity, 1, 1
        )
        rel_loss = torch.sum(rel_errors * rel_mask, dim=(-2, -1)) / (
            1e-7 + torch.sum(rel_mask, dim=(-2, -1))
        )

        if mask_loss is not None:
            rel_mask_loss = mask_loss.view(B // multiplicity, multiplicity, 1).repeat(
                1, 1, multiplicity
            )
            rel_loss = torch.sum(rel_loss * rel_mask_loss) / (
                torch.sum(rel_mask_loss) + 1e-7
            )
        else:
            rel_loss = torch.mean(rel_loss)

    return loss, rel_loss
