"""Per-voltage-tier edge counts on the Texas backbone (Table II).

Columns:
  GT          same-tier ACTIVSg2000 backbone branches (parallel circuits counted)
  GT cap C    GT branches whose bus pair lies in the reference candidate set C
              (kNN 12 on lon/lat degrees union the exact Euclidean MST), the
              attainable upper bound reported in the paper
  GED         unique predicted same-tier edges (self-loops excluded unless
              --count_self_loops, which reproduces the numbers printed in the paper)
  d_GT = GED - GT, d_C = GED - GT cap C

    python scripts/07_tier_table.py --ours outputs/ged_full_texas.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from ged.paths import BACKBONE_DIR, OUTPUTS_DIR

TIERS = [115.0, 161.0, 230.0, 500.0]


def reference_candidate_set(bus, k=12):
    """kNN(k) in (lon, lat) degrees plus the exact Euclidean MST, as bus-id pairs."""
    ids = bus["bus_id"].astype(int).to_numpy()
    P = bus[["lon", "lat"]].to_numpy(float)
    _, nn = cKDTree(P).query(P, k=k + 1)
    C = set()
    for i in range(len(ids)):
        for j in nn[i, 1:]:
            a, b = ids[i], ids[int(j)]
            C.add((a, b) if a < b else (b, a))
    G = nx.Graph()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            G.add_edge(ids[i], ids[j], weight=float(np.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1])))
    for u, v in nx.minimum_spanning_tree(G, weight="weight").edges():
        C.add((u, v) if u < v else (v, u))
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--backbone", type=Path, default=BACKBONE_DIR)
    ap.add_argument("--knn", type=int, default=12, help="k of the reference candidate set")
    ap.add_argument("--count_self_loops", action="store_true",
                    help="Also count degenerate (i, i) pairs, as the paper's Table II did (see README)")
    ap.add_argument("--out_json", type=Path, default=OUTPUTS_DIR / "table2_tiers.json")
    args = ap.parse_args()

    bus = pd.read_csv(args.backbone / "bus.csv").sort_values("bus_id").reset_index(drop=True)
    branch = pd.read_csv(args.backbone / "branch.csv")
    kv = dict(zip(bus["bus_id"].astype(int), bus["baseKV"].astype(float)))
    C = reference_candidate_set(bus, k=args.knn)

    gt, gt_c = {t: 0 for t in TIERS}, {t: 0 for t in TIERS}
    for a, b in zip(branch.from_bus.astype(int), branch.to_bus.astype(int)):
        if abs(kv[a] - kv[b]) > 1e-3:
            continue
        gt[kv[a]] += 1
        if ((a, b) if a < b else (b, a)) in C:
            gt_c[kv[a]] += 1

    ours = torch.load(args.ours, weights_only=False)
    bid = np.asarray(ours["bus_id"])
    pred = set()
    for u, v in ours["pred_edges"]:
        a, b = int(bid[u]), int(bid[v])
        if a != b or args.count_self_loops:
            pred.add((a, b) if a < b else (b, a))
    n_self = sum(1 for a, b in pred if a == b)
    ged = {t: 0 for t in TIERS}
    for a, b in pred:
        if abs(kv[a] - kv[b]) <= 1e-3:
            ged[kv[a]] += 1

    if n_self:
        print(f"(counting {n_self} degenerate self-loop pairs, as in the paper's Table II)")
    rows = {}
    print(f"{'Tier':>8} {'GT':>6} {'GT&C':>6} {'GED':>6} {'d_GT':>6} {'d_C':>6}")
    for t in TIERS + ["Total"]:
        g = sum(gt.values()) if t == "Total" else gt[t]
        gc = sum(gt_c.values()) if t == "Total" else gt_c[t]
        o = sum(ged.values()) if t == "Total" else ged[t]
        name = t if t == "Total" else f"{int(t)} kV"
        rows[name] = {"GT": g, "GT_cap_C": gc, "GED": o, "d_GT": o - g, "d_C": o - gc}
        print(f"{name:>8} {g:>6} {gc:>6} {o:>6} {o - g:>+6} {o - gc:>+6}")

    g = np.array([gt[t] for t in TIERS], float)
    p = np.array([ged[t] for t in TIERS], float)
    g, p = g / g.sum(), p / p.sum()
    kl_pg = float((p * np.log(p / g)).sum())
    kl_gp = float((g * np.log(g / p)).sum())
    coverage = sum(gt_c.values()) / max(sum(gt.values()), 1)
    print(f"\nGT coverage of C: {100 * coverage:.1f}%  (500 kV {100 * gt_c[500.0] / gt[500.0]:.1f}%, "
          f"230 kV {100 * gt_c[230.0] / gt[230.0]:.1f}%)")
    print(f"KL(GED || GT) = {kl_pg:.3f}, KL(GT || GED) = {kl_gp:.3f}, mean = {(kl_pg + kl_gp) / 2:.3f}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": rows, "coverage": coverage, "kl_ged_gt": kl_pg,
                                         "kl_gt_ged": kl_gp}, indent=2))
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
