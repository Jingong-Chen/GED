"""Build every processed input used by training and evaluation.

Inputs (shipped in data/):
  data/activsg2000_geocoded/{bus,branch,gen}.csv   ACTIVSg2000 with Census-Gazetteer bus coordinates
                                                   (re-create with scripts/00_geocode_activsg.py)
  data/raster/texas_raster_28ch.npz                28-channel Texas raster (214 x 263 x 28)

Outputs (data/processed/):
  activsg2000_texas/          buses with state == TX
  backbone/                   >=115 kV, geocoded, largest connected component (1,447 buses)
  patches_1deg/               155 overlapping 1-degree patches (stride 0.5 deg), 64x64 raster
  splits.json                 124 / 15 / 16 random split of the 1-degree patches (seed 42)
  patches_4deg/, patches_7deg/  4-degree (stride 2, 128x128) and 7-degree (stride 3, 192x192) patches
  global/global.pt            full-Texas sample, 256x256 raster

Usage:
  python scripts/01_prepare_data.py            # everything (about 2-4 minutes on a laptop CPU)
  python scripts/01_prepare_data.py --only global
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from ged.paths import GEOCODED_DIR, PROCESSED_DIR, RASTER_FILE, TX_BBOX

LOG_MEANS = np.array([-2.52, -1.59, -2.89], dtype=np.float32)
LOG_STDS = np.array([0.60, 0.40, 2.24], dtype=np.float32)
V_TIER_IDX = {115.0: 0, 161.0: 1, 230.0: 2, 500.0: 3}
EPS = 1e-6


def normalize_rxb(r, x, b):
    raw = np.array([np.log10(max(r, EPS)), np.log10(max(x, EPS)), np.log10(max(b, EPS))], dtype=np.float32)
    return (raw - LOG_MEANS) / LOG_STDS


def load_raster(path):
    d = np.load(path)
    return d["features"].astype(np.float32)


# --------------------------------------------------------------------------
# 1. Texas filter and >=115 kV backbone
# --------------------------------------------------------------------------
def build_texas(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    bus = pd.read_csv(src / "bus.csv")
    branch = pd.read_csv(src / "branch.csv")
    gen = pd.read_csv(src / "gen.csv")
    tx_bus = bus[bus.state == "TX"].reset_index(drop=True)
    ids = set(tx_bus.bus_id.values)
    tx_branch = branch[branch.from_bus.isin(ids) & branch.to_bus.isin(ids)].reset_index(drop=True)
    gen_col = "bus_id" if "bus_id" in gen.columns else gen.columns[0]
    tx_gen = gen[gen[gen_col].isin(ids)].reset_index(drop=True)
    tx_bus.to_csv(dst / "bus.csv", index=False)
    tx_branch.to_csv(dst / "branch.csv", index=False)
    tx_gen.to_csv(dst / "gen.csv", index=False)
    print(f"[texas] buses {len(bus)} -> {len(tx_bus)}, branches {len(branch)} -> {len(tx_branch)}")


def build_backbone(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    bus = pd.read_csv(src / "bus.csv")
    branch = pd.read_csv(src / "branch.csv")
    gen = pd.read_csv(src / "gen.csv")
    # Names that fail to geocode are placed at the Texas centroid; drop them.
    bus = bus[~((bus.lat == 31.0) & (bus.lon == -99.0))].copy()
    bus = bus[bus.baseKV >= 115].copy()
    keep = set(bus.bus_id.values)
    branch = branch[branch.from_bus.isin(keep) & branch.to_bus.isin(keep)].copy()
    G = nx.Graph()
    G.add_nodes_from(bus.bus_id.values)
    for _, r in branch.iterrows():
        G.add_edge(int(r.from_bus), int(r.to_bus))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main = comps[0]
    bus = bus[bus.bus_id.isin(main)].copy()
    branch = branch[branch.from_bus.isin(main) & branch.to_bus.isin(main)].copy()
    gen = gen[gen.bus_id.isin(main)].copy()
    bus.to_csv(dst / "bus.csv", index=False)
    branch.to_csv(dst / "branch.csv", index=False)
    gen.to_csv(dst / "gen.csv", index=False)
    meta = {
        "filter": "drop centroid-unresolved; baseKV>=115; largest connected component",
        "bus_count": int(len(bus)), "branch_count": int(len(branch)), "gen_count": int(len(gen)),
        "voltage_tiers": {str(int(v)): int((bus.baseKV == v).sum()) for v in sorted(bus.baseKV.unique())},
    }
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[backbone] {len(bus)} buses / {len(branch)} branches / {len(gen)} gens")


# --------------------------------------------------------------------------
# 2. 1-degree patches (64x64 raster, local positions in [0,1]^2)
# --------------------------------------------------------------------------
def build_1deg(backbone: Path, feats: np.ndarray, out_dir: Path, splits_file: Path,
               patch_deg=1.0, stride_deg=0.5, min_buses=8, patch_res=64):
    out_dir.mkdir(parents=True, exist_ok=True)
    bus = pd.read_csv(backbone / "bus.csv")
    branch = pd.read_csv(backbone / "branch.csv")
    gen = pd.read_csv(backbone / "gen.csv")
    H, W, C = feats.shape

    # Patch grid over the padded backbone bounding box
    pad = 0.25
    lat_min = np.floor((bus.lat.min() - pad) * 2) / 2
    lat_max = np.ceil((bus.lat.max() + pad) * 2) / 2
    lon_min = np.floor((bus.lon.min() - pad) * 2) / 2
    lon_max = np.ceil((bus.lon.max() + pad) * 2) / 2
    patches = []
    pid = 0
    for la in np.arange(lat_min, lat_max - patch_deg + 1e-9, stride_deg):
        for lo in np.arange(lon_min, lon_max - patch_deg + 1e-9, stride_deg):
            m = (bus.lat >= la) & (bus.lat < la + patch_deg) & (bus.lon >= lo) & (bus.lon < lo + patch_deg)
            sub = bus[m]
            if len(sub) < min_buses:
                continue
            ids = set(sub.bus_id.values)
            if len(branch[branch.from_bus.isin(ids) & branch.to_bus.isin(ids)]) < 1:
                continue
            patches.append({"patch_id": int(pid), "lat_min": float(la), "lat_max": float(la + patch_deg),
                            "lon_min": float(lo), "lon_max": float(lo + patch_deg)})
            pid += 1

    def crop(p):
        out = np.zeros((patch_res, patch_res, C), dtype=np.float32)
        for i in range(patch_res):
            for j in range(patch_res):
                lat = p["lat_min"] + (i + 0.5) / patch_res * (p["lat_max"] - p["lat_min"])
                lon = p["lon_min"] + (j + 0.5) / patch_res * (p["lon_max"] - p["lon_min"])
                fy = (lat - TX_BBOX["lat_min"]) / (TX_BBOX["lat_max"] - TX_BBOX["lat_min"])
                fx = (lon - TX_BBOX["lon_min"]) / (TX_BBOX["lon_max"] - TX_BBOX["lon_min"])
                if 0 <= fy < 1 and 0 <= fx < 1:
                    out[i, j] = feats[min(int(fy * H), H - 1), min(int(fx * W), W - 1)]
        return out

    # Impedance / rating per bus pair (parallel branches: mean r, x; summed b, rateA)
    edges = branch.copy()
    edges["pair"] = edges.apply(lambda r: tuple(sorted((int(r["from_bus"]), int(r["to_bus"])))), axis=1)
    edges = edges.groupby("pair").agg({"r": "mean", "x": "mean", "b": "sum", "rateA": "sum"}).reset_index()
    edge_attrs = {p: (float(r), float(x), float(b), float(rA))
                  for p, r, x, b, rA in zip(edges["pair"], edges["r"], edges["x"], edges["b"], edges["rateA"])}
    gen_at_bus = gen.groupby("bus_id")["Pg"].sum().to_dict()
    genmax_at_bus = gen.groupby("bus_id")["Pmax"].sum().to_dict()
    bus_lookup = bus.set_index("bus_id").to_dict("index")
    all_coords = bus[["lon", "lat"]].to_numpy()
    all_ids = bus["bus_id"].to_numpy()
    kdt_all = cKDTree(all_coords)

    samples = []
    for p in tqdm(patches, desc="patches_1deg"):
        m = (bus.lat >= p["lat_min"]) & (bus.lat < p["lat_max"]) & \
            (bus.lon >= p["lon_min"]) & (bus.lon < p["lon_max"])
        sub = bus[m].copy().reset_index(drop=True)
        b2l = {bid: i for i, bid in enumerate(sub.bus_id.values)}
        pos = np.stack([(sub.lon.values - p["lon_min"]) / (p["lon_max"] - p["lon_min"]),
                        (sub.lat.values - p["lat_min"]) / (p["lat_max"] - p["lat_min"])], axis=1).astype(np.float32)
        ids = set(sub.bus_id.values)
        sub_br = branch[branch.from_bus.isin(ids) & branch.to_bus.isin(ids)]
        edge_set = set()
        for _, r in sub_br.iterrows():
            a, b = b2l[int(r.from_bus)], b2l[int(r.to_bus)]
            if a != b:
                edge_set.add((a, b) if a < b else (b, a))
        ei = np.array(sorted(edge_set), dtype=np.int64).T
        voltage = sub.baseKV.values.astype(np.float32)
        sample = {
            "patch_id": p["patch_id"],
            "raster": torch.from_numpy(crop(p)),
            "gt_pos": torch.from_numpy(pos),
            "gt_edge_index": torch.from_numpy(ei),
            "gt_voltage": torch.from_numpy(voltage),
            "patch_meta": {"lat_min": p["lat_min"], "lat_max": p["lat_max"],
                           "lon_min": p["lon_min"], "lon_max": p["lon_max"],
                           "n_buses": int(len(sub)), "n_edges": int(ei.shape[1])},
        }
        # Bus ids are recovered by nearest-neighbour lookup of the (float32) local positions.
        n = pos.shape[0]
        node_lon = p["lon_min"] + pos[:, 0] * (p["lon_max"] - p["lon_min"])
        node_lat = p["lat_min"] + pos[:, 1] * (p["lat_max"] - p["lat_min"])
        _, idx = kdt_all.query(np.stack([node_lon, node_lat], axis=1), k=1)
        local_to_gid = all_ids[idx]

        e_attrs = np.zeros((ei.shape[1], 3), dtype=np.float32)
        e_rate = np.zeros((ei.shape[1],), dtype=np.float32)
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            bu, bv = int(local_to_gid[u]), int(local_to_gid[v])
            key = (bu, bv) if bu < bv else (bv, bu)
            if key in edge_attrs:
                r, x, b, rA = edge_attrs[key]
            else:  # tier-median fallback
                v_match = bus[bus.bus_id.isin([bu, bv])]["baseKV"].max()
                tier_ids = bus[bus.baseKV == v_match].bus_id
                te = branch[branch.from_bus.isin(tier_ids) & branch.to_bus.isin(tier_ids)]
                if len(te):
                    r, x = float(te["r"].median()), float(te["x"].median())
                    b, rA = float(te["b"].median()), float(te["rateA"].median())
                else:
                    r, x, b, rA = 0.01, 0.05, 0.01, 200.0
            e_attrs[k] = normalize_rxb(r, x, b)
            e_rate[k] = rA

        attrs = {key: np.zeros((n,), dtype=np.float32) for key in ("Pd", "Qd", "Pg", "Pmax")}
        btype = np.zeros((n,), dtype=np.int64)
        btier = np.zeros((n,), dtype=np.int64)
        for k in range(n):
            gid = int(local_to_gid[k])
            bl = bus_lookup.get(gid)
            if bl is None:
                continue
            attrs["Pd"][k] = float(bl.get("Pd", 0.0))
            attrs["Qd"][k] = float(bl.get("Qd", 0.0))
            btype[k] = int(bl.get("type", 1.0))
            btier[k] = V_TIER_IDX.get(float(bl.get("baseKV", 115.0)), 0)
            attrs["Pg"][k] = float(gen_at_bus.get(gid, 0.0))
            attrs["Pmax"][k] = float(genmax_at_bus.get(gid, 0.0))
        sample["gt_edge_attrs"] = torch.from_numpy(e_attrs)
        sample["gt_edge_rateA"] = torch.from_numpy(e_rate)
        sample["gt_bus_attrs"] = {
            "Pd": torch.from_numpy(attrs["Pd"]), "Qd": torch.from_numpy(attrs["Qd"]),
            "Pg": torch.from_numpy(attrs["Pg"]), "Pmax": torch.from_numpy(attrs["Pmax"]),
            "type": torch.from_numpy(btype), "tier": torch.from_numpy(btier),
        }
        sample["bus_id"] = torch.from_numpy(local_to_gid.astype(np.int64))
        torch.save(sample, out_dir / f"patch_{p['patch_id']:04d}.pt")
        samples.append({"patch_id": p["patch_id"], "n_buses": int(n), "n_edges": int(ei.shape[1])})

    (out_dir / "manifest.json").write_text(json.dumps({
        "n_samples": len(samples), "samples": samples, "raster_shape": [patch_res, patch_res, C],
        "patch_size_deg": patch_deg, "stride_deg": stride_deg, "min_buses": min_buses,
    }, indent=2))
    print(f"[1deg] {len(samples)} patches -> {out_dir}")

    # Random 80/10/10 split by patch id (seed 42)
    ids = sorted(s["patch_id"] for s in samples)
    rng = np.random.RandomState(42)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_train, n_val = int(0.8 * len(ids)), int(0.1 * len(ids))
    splits = {"train": sorted(shuffled[:n_train]),
              "val": sorted(shuffled[n_train:n_train + n_val]),
              "test": sorted(shuffled[n_train + n_val:])}
    splits_file.write_text(json.dumps(splits, indent=2))
    print(f"[splits] train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")


# --------------------------------------------------------------------------
# 3. 4- and 7-degree patches
# --------------------------------------------------------------------------
def build_large_patches(backbone: Path, feats: np.ndarray, out_dir: Path, patch_size, stride,
                        target_hw, min_buses):
    bus = pd.read_csv(backbone / "bus.csv").sort_values("bus_id").reset_index(drop=True)
    branch = pd.read_csv(backbone / "branch.csv")
    gen = pd.read_csv(backbone / "gen.csv")
    pg_at = gen.groupby("bus_id")["Pg"].sum().to_dict()
    pmax_at = gen.groupby("bus_id")["Pmax"].sum().to_dict()
    br = branch.copy()
    br["pair"] = br.apply(lambda r: tuple(sorted((int(r["from_bus"]), int(r["to_bus"])))), axis=1)
    br = br.groupby("pair").agg({"r": "mean", "x": "mean", "b": "sum", "rateA": "sum"}).reset_index()
    edge_imp = {p: (float(r), float(x), float(b), float(rA))
                for p, r, x, b, rA in zip(br["pair"], br["r"], br["x"], br["b"], br["rateA"])}
    H, W, C = feats.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    S, Wlon, N, E = TX_BBOX["lat_min"], TX_BBOX["lon_min"], TX_BBOX["lat_max"], TX_BBOX["lon_max"]
    patches = []
    pid = 0
    for la in tqdm(np.arange(S - 0.5, N - patch_size + 0.01, stride), desc=f"patches_{int(patch_size)}deg"):
        for lo in np.arange(Wlon - 0.5, E - patch_size + 0.01, stride):
            m = (bus.lat >= la) & (bus.lat < la + patch_size) & (bus.lon >= lo) & (bus.lon < lo + patch_size)
            sub = bus[m].copy().reset_index(drop=True)
            if len(sub) < min_buses:
                continue
            ids = set(sub.bus_id.values)
            sub_br = branch[branch.from_bus.isin(ids) & branch.to_bus.isin(ids)]
            if len(sub_br) < 1:
                continue
            b2l = {int(b): i for i, b in enumerate(sub.bus_id.values)}
            pos = np.stack([(sub.lon.values - lo) / patch_size,
                            (sub.lat.values - la) / patch_size], axis=1).astype(np.float32)
            eu, ev, eattr, edge_set = [], [], [], set()
            for _, r in sub_br.iterrows():
                a, b = b2l[int(r.from_bus)], b2l[int(r.to_bus)]
                if a == b:
                    continue
                u, v = (a, b) if a < b else (b, a)
                if (u, v) in edge_set:
                    continue
                edge_set.add((u, v))
                key = tuple(sorted((int(r.from_bus), int(r.to_bus))))
                rr, xx, bb, _ = edge_imp[key] if key in edge_imp else (float(r.r), float(r.x), float(r.b), 0.0)
                eu.append(u)
                ev.append(v)
                eattr.append(normalize_rxb(rr, xx, bb))
            ei = np.stack([np.array(eu), np.array(ev)], axis=0).astype(np.int64) if eu else np.zeros((2, 0), np.int64)
            eattr = np.array(eattr, dtype=np.float32) if eattr else np.zeros((0, 3), dtype=np.float32)

            raster = np.zeros((target_hw, target_hw, C), dtype=np.float32)
            for i in range(target_hw):
                lat = la + (i + 0.5) / target_hw * patch_size
                for j in range(target_hw):
                    lon = lo + (j + 0.5) / target_hw * patch_size
                    fy = (lat - TX_BBOX["lat_min"]) / (TX_BBOX["lat_max"] - TX_BBOX["lat_min"])
                    fx = (lon - TX_BBOX["lon_min"]) / (TX_BBOX["lon_max"] - TX_BBOX["lon_min"])
                    if 0 <= fy < 1 and 0 <= fx < 1:
                        raster[i, j] = feats[min(int(fy * H), H - 1), min(int(fx * W), W - 1)]

            voltage = sub.baseKV.values.astype(np.float32)
            sample = {
                "patch_id": pid,
                "raster": torch.from_numpy(raster),
                "gt_pos": torch.from_numpy(pos),
                "gt_edge_index": torch.from_numpy(ei),
                "gt_voltage": torch.from_numpy(voltage),
                "gt_edge_attrs": torch.from_numpy(eattr),
                "gt_bus_attrs": {
                    "Pd": torch.from_numpy(sub.Pd.values.astype(np.float32)),
                    "Qd": torch.from_numpy(sub.Qd.values.astype(np.float32)),
                    "Pg": torch.from_numpy(np.array([float(pg_at.get(int(b), 0.0)) for b in sub.bus_id],
                                                    dtype=np.float32)),
                    "Pmax": torch.from_numpy(np.array([float(pmax_at.get(int(b), 0.0)) for b in sub.bus_id],
                                                      dtype=np.float32)),
                    "type": torch.from_numpy(sub.type.values.astype(np.int64)),
                    "tier": torch.from_numpy(np.array([V_TIER_IDX.get(float(v), 0) for v in voltage],
                                                      dtype=np.int64)),
                },
                "bus_id": torch.from_numpy(sub["bus_id"].values.astype(np.int64)),
                "patch_meta": {"lat_min": float(la), "lat_max": float(la + patch_size),
                               "lon_min": float(lo), "lon_max": float(lo + patch_size),
                               "n_buses": int(len(sub)), "n_edges": int(ei.shape[1]),
                               "scope": f"{patch_size}deg"},
            }
            torch.save(sample, out_dir / f"patch_{pid:04d}.pt")
            patches.append({"patch_id": pid, "n_buses": int(len(sub)), "n_edges": int(ei.shape[1]),
                            "lat_min": float(la), "lon_min": float(lo)})
            pid += 1
    (out_dir / "manifest.json").write_text(json.dumps({
        "patch_size_deg": patch_size, "stride_deg": stride, "min_buses": min_buses,
        "n_patches": len(patches), "patches": patches, "raster_resolution": target_hw}, indent=2))
    print(f"[{patch_size}deg] {len(patches)} patches -> {out_dir}")


# --------------------------------------------------------------------------
# 4. Full-Texas sample
# --------------------------------------------------------------------------
def build_global(backbone: Path, feats: np.ndarray, out_file: Path, target_hw=256):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    bus = pd.read_csv(backbone / "bus.csv").sort_values("bus_id").reset_index(drop=True)
    branch = pd.read_csv(backbone / "branch.csv")
    gen = pd.read_csv(backbone / "gen.csv")
    br = branch.copy()
    br["pair"] = br.apply(lambda r: tuple(sorted((int(r["from_bus"]), int(r["to_bus"])))), axis=1)
    br = br.groupby("pair").agg({"r": "mean", "x": "mean", "b": "sum", "rateA": "sum"}).reset_index()
    b2l = {int(b): i for i, b in enumerate(bus["bus_id"].values)}
    pos = np.stack([(bus["lon"].values - TX_BBOX["lon_min"]) / (TX_BBOX["lon_max"] - TX_BBOX["lon_min"]),
                    (bus["lat"].values - TX_BBOX["lat_min"]) / (TX_BBOX["lat_max"] - TX_BBOX["lat_min"])],
                   axis=1).astype(np.float32)
    eu, ev, eattr = [], [], []
    for _, row in br.iterrows():
        a, b = row["pair"]
        if a not in b2l or b not in b2l or b2l[a] == b2l[b]:
            continue
        eu.append(b2l[a])
        ev.append(b2l[b])
        eattr.append(normalize_rxb(float(row["r"]), float(row["x"]), float(row["b"])))
    ei = np.stack([np.array(eu), np.array(ev)], axis=0).astype(np.int64)
    voltage = bus["baseKV"].values.astype(np.float32)
    pg_at = gen.groupby("bus_id")["Pg"].sum().to_dict()
    pmax_at = gen.groupby("bus_id")["Pmax"].sum().to_dict()
    H, W, C = feats.shape
    raster = np.zeros((target_hw, target_hw, C), dtype=np.float32)
    for i in range(target_hw):
        yi = min(int((i + 0.5) / target_hw * H), H - 1)
        for j in range(target_hw):
            raster[i, j] = feats[yi, min(int((j + 0.5) / target_hw * W), W - 1)]
    sample = {
        "patch_id": 9999,
        "raster": torch.from_numpy(raster),
        "gt_pos": torch.from_numpy(pos),
        "gt_edge_index": torch.from_numpy(ei),
        "gt_voltage": torch.from_numpy(voltage),
        "gt_edge_attrs": torch.from_numpy(np.array(eattr, dtype=np.float32)),
        "gt_bus_attrs": {
            "Pd": torch.from_numpy(bus["Pd"].values.astype(np.float32)),
            "Qd": torch.from_numpy(bus["Qd"].values.astype(np.float32)),
            "Pg": torch.from_numpy(np.array([float(pg_at.get(int(b), 0.0)) for b in bus["bus_id"]],
                                            dtype=np.float32)),
            "Pmax": torch.from_numpy(np.array([float(pmax_at.get(int(b), 0.0)) for b in bus["bus_id"]],
                                              dtype=np.float32)),
            "type": torch.from_numpy(bus["type"].values.astype(np.int64)),
            "tier": torch.from_numpy(np.array([V_TIER_IDX.get(float(v), 0) for v in voltage], dtype=np.int64)),
        },
        "bus_id": torch.from_numpy(bus["bus_id"].values.astype(np.int64)),
        "patch_meta": {**TX_BBOX, "n_buses": int(pos.shape[0]), "n_edges": int(ei.shape[1]),
                       "scope": "full_texas"},
    }
    torch.save(sample, out_file)
    print(f"[global] {pos.shape[0]} buses, {ei.shape[1]} edges, raster {raster.shape} -> {out_file}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geocoded", type=Path, default=GEOCODED_DIR)
    ap.add_argument("--raster", type=Path, default=RASTER_FILE)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--only", choices=["backbone", "1deg", "large", "global"], nargs="*",
                    help="Run only these stages (default: all). Later stages need the backbone.")
    args = ap.parse_args()
    stages = set(args.only or ["backbone", "1deg", "large", "global"])
    out = args.out
    backbone = out / "backbone"
    if "backbone" in stages:
        build_texas(args.geocoded, out / "activsg2000_texas")
        build_backbone(out / "activsg2000_texas", backbone)
    feats = load_raster(args.raster)
    if "1deg" in stages:
        build_1deg(backbone, feats, out / "patches_1deg", out / "splits.json")
    if "large" in stages:
        build_large_patches(backbone, feats, out / "patches_4deg", 4.0, 2.0, 128, min_buses=30)
        build_large_patches(backbone, feats, out / "patches_7deg", 7.0, 3.0, 192, min_buses=60)
    if "global" in stages:
        build_global(backbone, feats, out / "global" / "global.pt")


if __name__ == "__main__":
    main()
