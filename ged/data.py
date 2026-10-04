"""Multi-scale training data: 1-, 4-, 7-degree patches and the full-Texas graph.

Dataset classes for the three training stages:
  MultiScaleDataset  stage 1 (topology + impedance + Fiedler)
  PhysDataset        stage 2 (+ per-bus injection P = Pg - Pd for the DC-PF loss)
  ThermalDataset     stage 3 (+ per-pair MVA rating target)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ged.losses import RATE_LOG_MEAN, RATE_LOG_STD
from ged.paths import GLOBAL_FILE, PATCH_1DEG_DIR, PATCH_4DEG_DIR, PATCH_7DEG_DIR, SPLITS_FILE


def voltage_to_tier(v: float) -> int:
    if v >= 500: return 3
    if v >= 230: return 2
    return 1


class MultiScaleDataset(Dataset):
    """Mixes 1° patches, 4° patches, 7° patches, and the global sample.

    Each item samples ``n_pairs`` training pairs from one graph: up to n_pairs/3
    ground-truth branches (labelled by voltage tier) and the rest negatives drawn
    from kNN(15) and spatial-MST non-edges (random pairs as fallback).
    ``n_geo_channels`` slices the raster to its first channels (28 in the paper).
    """
    def __init__(self, files, max_n=1536, n_pairs=384, n_geo_channels=28):
        self.files = files
        self.max_n = max_n
        self.n_pairs = n_pairs
        self.n_geo_channels = n_geo_channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        raster = d["raster"][:, :, : self.n_geo_channels]
        pos = d["gt_pos"].numpy()
        edge_index = d["gt_edge_index"].numpy()
        voltage = d["gt_voltage"].numpy()
        e_attrs = d["gt_edge_attrs"].numpy() if "gt_edge_attrs" in d else np.zeros((edge_index.shape[1], 3), dtype=np.float32)
        n = pos.shape[0]

        if n > self.max_n:
            order = np.argsort(-voltage)[: self.max_n]
            keep = sorted(order.tolist()); keep_set = set(keep)
            old_to_new = {old: new for new, old in enumerate(keep)}
            pos = pos[keep]; voltage = voltage[keep]
            new_edges, new_attrs = [], []
            for k in range(edge_index.shape[1]):
                u, v = int(edge_index[0, k]), int(edge_index[1, k])
                if u in keep_set and v in keep_set:
                    new_edges.append([old_to_new[u], old_to_new[v]])
                    new_attrs.append(e_attrs[k])
            edge_index = np.array(new_edges, dtype=np.int64).T if new_edges else np.zeros((2, 0), dtype=np.int64)
            e_attrs = np.array(new_attrs, dtype=np.float32) if new_attrs else np.zeros((0, 3), dtype=np.float32)
            n = self.max_n

        edge_map = {}
        attr_map = {}
        for k in range(edge_index.shape[1]):
            u, v = int(edge_index[0, k]), int(edge_index[1, k])
            if u == v: continue
            if u > v: u, v = v, u
            tier = voltage_to_tier(max(voltage[u], voltage[v]))
            edge_map[(u, v)] = tier
            attr_map[(u, v)] = e_attrs[k]

        gt_pairs = list(edge_map.keys())
        np.random.shuffle(gt_pairs)
        n_pos = min(len(gt_pairs), self.n_pairs // 3)
        pos_pairs = gt_pairs[:n_pos]
        n_neg = self.n_pairs - n_pos

        neg_pairs, neg_seen = [], set()
        max_unique = n * (n - 1) // 2 - len(edge_map)
        target_neg = min(n_neg, max(0, max_unique))
        if n > 2:
            import networkx as _nx
            from scipy.spatial import cKDTree
            kdt = cKDTree(pos)
            k_neighbors = min(15, n - 1)
            _, neigh = kdt.query(pos, k=k_neighbors + 1)
            cands = []
            for i in range(n):
                for j in neigh[i, 1:]:
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    if (a, b) not in edge_map:
                        cands.append((a, b))
            # Add spatial MST candidates
            if n <= 800:
                mst_g = _nx.Graph()
                for i in range(n):
                    for j in range(i + 1, n):
                        mst_g.add_edge(i, j, weight=float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])))
                for u, v in _nx.minimum_spanning_tree(mst_g, weight="weight").edges():
                    a, b = (int(u), int(v)) if u < v else (int(v), int(u))
                    if (a, b) not in edge_map:
                        cands.append((a, b))
            cands = list(set(cands))
            np.random.shuffle(cands)
            for ab in cands:
                if len(neg_pairs) >= target_neg: break
                if ab in neg_seen: continue
                neg_pairs.append(ab); neg_seen.add(ab)
            tries = 0; max_tries = target_neg * 8 + 64
            while len(neg_pairs) < target_neg and tries < max_tries:
                tries += 1
                i, j = np.random.choice(n, 2, replace=False)
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if (a, b) in edge_map or (a, b) in neg_seen: continue
                neg_pairs.append((a, b)); neg_seen.add((a, b))

        pairs = pos_pairs + neg_pairs
        x_0 = torch.tensor([edge_map.get(p, 0) for p in pairs], dtype=torch.long)
        attrs = torch.zeros(len(pairs), 3, dtype=torch.float32)
        attr_mask = torch.zeros(len(pairs), dtype=torch.bool)
        for k, p in enumerate(pairs):
            if p in attr_map:
                attrs[k] = torch.from_numpy(attr_map[p])
                attr_mask[k] = True

        geo = raster.permute(2, 0, 1).contiguous()
        pos_norm = np.zeros_like(pos)
        pos_norm[:, 0] = pos[:, 1] * 2 - 1
        pos_norm[:, 1] = pos[:, 0] * 2 - 1
        pos_pad = torch.zeros(self.max_n, 2)
        pos_pad[:n] = torch.from_numpy(pos_norm.astype(np.float32))

        i_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return {
            "geo": geo, "pos": pos_pad,
            "i_idx": i_idx, "j_idx": j_idx,
            "x_0": x_0, "attrs": attrs, "attr_mask": attr_mask,
            "n_real": n,
        }


def collate(samples):
    P = max(s["i_idx"].numel() for s in samples)
    B = len(samples)
    # Resize all rasters to same shape (downsample bigger to smaller)
    raster_h = min(s["geo"].shape[-2] for s in samples)
    raster_w = min(s["geo"].shape[-1] for s in samples)
    out = {
        "geo": torch.stack([F.interpolate(s["geo"].unsqueeze(0),
                                            size=(raster_h, raster_w),
                                            mode="bilinear",
                                            align_corners=False).squeeze(0) for s in samples]),
        "pos": torch.stack([s["pos"] for s in samples]),
        "n_real": torch.tensor([s["n_real"] for s in samples]),
    }
    out["i_idx"] = torch.zeros(B, P, dtype=torch.long)
    out["j_idx"] = torch.zeros(B, P, dtype=torch.long)
    out["x_0"]   = torch.zeros(B, P, dtype=torch.long)
    out["attrs"] = torch.zeros(B, P, 3, dtype=torch.float32)
    out["attr_mask"] = torch.zeros(B, P, dtype=torch.bool)
    out["pair_mask"] = torch.zeros(B, P, dtype=torch.bool)
    for b, s in enumerate(samples):
        p = s["i_idx"].numel()
        out["i_idx"][b, :p] = s["i_idx"]
        out["j_idx"][b, :p] = s["j_idx"]
        out["x_0"][b, :p]   = s["x_0"]
        out["attrs"][b, :p] = s["attrs"]
        out["attr_mask"][b, :p] = s["attr_mask"]
        out["pair_mask"][b, :p] = True
    return out


def collect_files(splits_file=SPLITS_FILE, patch_1=PATCH_1DEG_DIR, patch_4=PATCH_4DEG_DIR,
                  patch_7=PATCH_7DEG_DIR, global_file=GLOBAL_FILE):
    """Return (train_files, val_files) mixing 1-, 4-, 7-degree patches and the full-Texas sample.

    1-degree patches follow splits.json. 4- and 7-degree patches are split 80/20 by
    patch id and repeated 3x and 6x in the training list. The full-Texas sample is
    repeated 20x in training and also used once for validation.
    """
    splits_d = json.loads(Path(splits_file).read_text())
    patch_1, patch_4, patch_7 = Path(patch_1), Path(patch_4), Path(patch_7)
    global_file = Path(global_file)
    train_files, val_files = [], []
    # 1° patches according to existing split
    for pid in splits_d["train"]:
        f = patch_1 / f"patch_{pid:04d}.pt"
        if f.exists(): train_files.append(f)
    for pid in splits_d["val"]:
        f = patch_1 / f"patch_{pid:04d}.pt"
        if f.exists(): val_files.append(f)
    # 4° + 7° patches: 80/20 split by patch_id
    for patch_dir, n_repeat in [(patch_4, 3), (patch_7, 6)]:
        if not patch_dir.exists(): continue
        ids = sorted([int(p.stem.split("_")[1]) for p in patch_dir.glob("patch_*.pt")])
        cut = int(len(ids) * 0.8)
        for pid in ids[:cut]:
            f = patch_dir / f"patch_{pid:04d}.pt"
            for _ in range(n_repeat):
                train_files.append(f)
        for pid in ids[cut:]:
            f = patch_dir / f"patch_{pid:04d}.pt"
            val_files.append(f)
    # Global sample: heavily oversample so the model sees it every batch or two
    if global_file.exists():
        for _ in range(20):
            train_files.append(global_file)
        val_files.append(global_file)
    return train_files, val_files


class PhysDataset(MultiScaleDataset):
    """Extends MultiScaleDataset to also load per-bus injection P_inj = Pg - Pd."""

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        sample = super().__getitem__(idx)
        # Pull Pd/Pg before any subsetting; align by sorting to match the parent's keep order
        n = sample["n_real"]
        bus_attrs = d.get("gt_bus_attrs", {})
        Pd = bus_attrs.get("Pd"); Pg = bus_attrs.get("Pg")
        n_full = d["gt_pos"].shape[0]
        if Pd is None: Pd = torch.zeros(n_full)
        if Pg is None: Pg = torch.zeros(n_full)
        p_inj_full = (Pg - Pd).float()
        # If the parent class subset by argsort(-voltage)[:max_n], replicate
        if n_full > self.max_n:
            voltage = d["gt_voltage"].numpy()
            order = np.argsort(-voltage)[: self.max_n]
            keep = sorted(order.tolist())
            p_inj = p_inj_full[keep]
        else:
            p_inj = p_inj_full
        p_inj_pad = torch.zeros(self.max_n)
        p_inj_pad[:n] = p_inj
        sample["p_inj"] = p_inj_pad
        return sample


def phys_collate(samples):
    out = collate(samples)
    out["p_inj"] = torch.stack([s["p_inj"] for s in samples])
    return out


class ThermalDataset(PhysDataset):
    """Add per-pair gt rateA to the batch (only valid for true edges)."""
    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        sample = super().__getitem__(idx)
        # Build edge_idx → rateA lookup
        edge_index = d["gt_edge_index"].numpy()
        rate_full = d.get("gt_edge_rateA")
        if rate_full is None:
            rate_full = torch.ones(edge_index.shape[1]) * 200.0
        # The parent class built `pairs` and stored attr_map by (u, v) sorted.
        # We re-create by reading the pairs from sample's i_idx/j_idx
        ii = sample["i_idx"].tolist(); jj = sample["j_idx"].tolist()
        rate_map = {}
        for k in range(edge_index.shape[1]):
            u, v = int(edge_index[0, k]), int(edge_index[1, k])
            if u == v: continue
            a, b = (u, v) if u < v else (v, u)
            rate_map[(a, b)] = float(rate_full[k])
        rate_per_pair = torch.zeros(len(ii), dtype=torch.float32)
        for p in range(len(ii)):
            u, v = ii[p], jj[p]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in rate_map:
                rate_per_pair[p] = rate_map[(a, b)]
        # z-normalise log10 (only meaningful where mask is True; we still write 0 for non-edges)
        log_r = torch.log10(rate_per_pair.clamp(min=1.0))
        sample["rate_z"] = (log_r - RATE_LOG_MEAN) / RATE_LOG_STD
        return sample


def thermal_collate(samples):
    out = collate(samples)
    out["p_inj"] = torch.stack([s["p_inj"] for s in samples])
    P = out["i_idx"].shape[1]
    B = len(samples)
    out["rate_z"] = torch.zeros(B, P, dtype=torch.float32)
    for b, s in enumerate(samples):
        p = s["rate_z"].numel()
        out["rate_z"][b, :p] = s["rate_z"]
    return out
