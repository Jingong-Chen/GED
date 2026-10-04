"""Inference: reverse diffusion on the candidate set + MST / top-K connectivity projector."""
from __future__ import annotations

import numpy as np
import torch

from ged.candidates import knn_candidates, spatial_mst, spatial_mst_knn
from ged.diffusion import reverse_diffusion

# z-normalisation statistics of log10 (r, x, b) in per-unit (S_base = 100 MVA)
LOG_MEANS = np.array([-2.52, -1.59, -2.89], dtype=np.float32)
LOG_STDS = np.array([0.60, 0.40, 2.24], dtype=np.float32)


def _to_model_pos(pos, pad_n):
    """[0,1]^2 (x=lon, y=lat) -> [-1,1]^2 in (y, x) order, zero-padded to pad_n rows.

    Note: the node transformer has no key-padding mask, so pad_n changes the
    output slightly. The paper used pad_n=1500 for the full state and
    pad_n=256 for 1-degree patches; keep these values to reproduce it.
    """
    n = pos.shape[0]
    pos_norm = pos.copy()
    pos_norm[:, 0] = pos[:, 1] * 2 - 1
    pos_norm[:, 1] = pos[:, 0] * 2 - 1
    pos_pad = torch.zeros(pad_n, 2)
    pos_pad[:n] = torch.from_numpy(pos_norm.astype(np.float32))
    return pos_pad


def _project(pairs, edge_probs, mst_set, n, edge_density):
    """MST edges are forced; the remaining budget is filled by the highest-score candidates."""
    mst_mask = np.array([(int(u), int(v)) in mst_set for u, v in pairs.tolist()], dtype=bool)
    target_k = max(n - 1, int(round(edge_density * n))) if n > 1 else 0
    target_k = min(target_k, pairs.shape[0])
    mst_idx = np.where(mst_mask)[0].tolist()
    non_mst_idx = np.where(~mst_mask)[0].tolist()
    non_mst_idx.sort(key=lambda i: -edge_probs[i])
    keep_idx = mst_idx + non_mst_idx[: max(0, target_k - len(mst_idx))]
    return keep_idx, mst_mask


@torch.no_grad()
def predict_full_texas(model, sample, knn=15, edge_density=1.7, n_samples=5, n_steps=30,
                       max_n=1500, device="cuda", use_mst=True, verbose=True):
    """Single-pass inference on one large graph (the 1,447-bus Texas backbone).

    Candidates are kNN(knn) pairs plus the spatial MST; all of them are scored
    by the model. The MST edges are then forced and the top
    round(edge_density * N) - (N - 1) other candidates are admitted.
    """
    raster = sample["raster"]
    pos = sample["gt_pos"].numpy()
    n = pos.shape[0]
    if n > max_n:
        raise ValueError(f"n={n} > max_n={max_n}")
    pos_pad = _to_model_pos(pos, max_n)

    pairs = knn_candidates(pos, k=knn)
    if use_mst:
        mst_set = spatial_mst_knn(pos)
        existing = {(int(u), int(v)) for u, v in pairs.tolist()}
        extras = [e for e in mst_set if e not in existing]
        if extras:
            pairs = np.concatenate([pairs, np.array(sorted(extras), dtype=np.int64)], axis=0)
        if verbose:
            print(f"  N={n}, candidate pairs={pairs.shape[0]}, MST edges added={len(extras)}", flush=True)
    else:
        mst_set = set()
        if verbose:
            print(f"  N={n}, candidate pairs={pairs.shape[0]}, MST projector disabled", flush=True)

    geo = raster.permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
    pos_b = pos_pad.unsqueeze(0).to(device)
    ii = torch.from_numpy(pairs[:, 0]).long().unsqueeze(0).to(device)
    jj = torch.from_numpy(pairs[:, 1]).long().unsqueeze(0).to(device)

    edge_probs, class_probs, attr_avg = reverse_diffusion(
        model, geo, pos_b, ii, jj, n_samples=n_samples, n_steps=n_steps)
    # Voltage tier = argmax over LV / HV / EHV (classes 1, 2, 3)
    pred_classes_all = 1 + class_probs[:, 1:].argmax(axis=1).astype(np.int64)

    keep_idx, mst_mask = _project(pairs, edge_probs, mst_set, n, edge_density)
    pred_attrs_z = attr_avg[keep_idx]
    return {
        "patch_id": int(sample["patch_id"]),
        "n_nodes": int(n),
        "n_pred_edges": len(keep_idx),
        "pred_edges": pairs[keep_idx],
        "pred_classes": pred_classes_all[keep_idx],
        "candidates": pairs,
        "all_class_probs": class_probs,
        "all_edge_probs": edge_probs,
        "pred_attrs_z": pred_attrs_z,
        "pred_rxb": 10.0 ** (pred_attrs_z * LOG_STDS + LOG_MEANS),
        "edge_prob": edge_probs[keep_idx],
        "gt_pos": pos,
        "patch_meta": sample["patch_meta"],
        "is_mst": mst_mask[keep_idx],
        "bus_id": sample["bus_id"].numpy() if "bus_id" in sample else None,
        "settings": {"knn": knn, "edge_density": edge_density, "n_samples": n_samples,
                     "n_steps": n_steps, "max_n": max_n, "use_mst": bool(use_mst)},
    }


@torch.no_grad()
def predict_patch(model, sample, knn=12, edge_density=1.5, n_samples=5, n_steps=30,
                  pad_n=256, device="cuda", use_mst=True):
    """Inference on one 1-degree patch (used for the MST-projector ablation).

    Only the kNN pairs are scored by the model; MST pairs missing from kNN are
    appended with score 1 and forced when ``use_mst`` is True. Without the MST
    the top round(edge_density * N) pairs by score are kept.
    """
    raster = sample["raster"]
    pos = sample["gt_pos"].numpy()
    n = pos.shape[0]
    pos_pad = _to_model_pos(pos, pad_n)
    pairs = knn_candidates(pos, k=knn)
    if pairs.shape[0] == 0:
        return None

    geo = raster.permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
    pos_b = pos_pad.unsqueeze(0).to(device)
    ii = torch.from_numpy(pairs[:, 0]).long().unsqueeze(0).to(device)
    jj = torch.from_numpy(pairs[:, 1]).long().unsqueeze(0).to(device)
    edge_probs, _, attr_avg = reverse_diffusion(
        model, geo, pos_b, ii, jj, n_samples=n_samples, n_steps=n_steps)

    mst_set = spatial_mst(pos) if (use_mst and n > 1) else set()
    existing = {(int(u), int(v)) for u, v in pairs.tolist()}
    extras = [e for e in mst_set if e not in existing]
    if extras:
        pairs = np.concatenate([pairs, np.array(sorted(extras), dtype=np.int64)], axis=0)
        edge_probs = np.concatenate([edge_probs, np.ones(len(extras), dtype=np.float32)])
        attr_avg = np.concatenate([attr_avg, np.zeros((len(extras), 3), dtype=np.float32)], axis=0)

    keep_idx, mst_mask = _project(pairs, edge_probs, mst_set, n, edge_density)
    pred_attrs_z = attr_avg[keep_idx]
    return {
        "patch_id": int(sample["patch_id"]),
        "n_nodes": int(n),
        "n_pred_edges": len(keep_idx),
        "pred_edges": pairs[keep_idx],
        "pred_attrs_z": pred_attrs_z,
        "pred_rxb": 10.0 ** (pred_attrs_z * LOG_STDS + LOG_MEANS) if len(pred_attrs_z)
        else np.zeros((0, 3), dtype=np.float32),
        "edge_prob": edge_probs[keep_idx],
        "gt_pos": pos,
        "patch_meta": sample["patch_meta"],
        "is_mst": mst_mask[keep_idx] if len(keep_idx) else np.zeros(0, dtype=bool),
    }


def predict_graph(model, sample, **kwargs):
    """Convenience alias for full-graph inference."""
    return predict_full_texas(model, sample, **kwargs)
