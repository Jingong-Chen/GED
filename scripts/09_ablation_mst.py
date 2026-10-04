"""Connectivity-projector ablation on the 16 held-out 1-degree test patches (Sec. IV-G).

Runs patch-level inference (kNN 12 candidates, 1.5 N edges, 5 chains x 30
steps) with and without the spatial-MST projector and reports mean Birchfield
metrics, mean number of connected components and the number of disconnected
patches. The paper's ablation used the stage-1 weights.

    python scripts/09_ablation_mst.py --ckpt weights/ged_texas_stage1.pt
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
import torch

from ged.inference import predict_patch
from ged.metrics import all_metrics
from ged.model import load_checkpoint
from ged.paths import OUTPUTS_DIR, PATCH_1DEG_DIR, SPLITS_FILE, WEIGHTS_DIR

KEYS = ["degree_w1", "avg_clustering_diff", "mean_path_diff", "diameter_diff",
        "spectral_gap_diff", "edge_length_w1"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=WEIGHTS_DIR / "ged_texas_stage1.pt")
    ap.add_argument("--patch_dir", type=Path, default=PATCH_1DEG_DIR)
    ap.add_argument("--splits", type=Path, default=SPLITS_FILE)
    ap.add_argument("--split", default="test")
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--edge_density", type=float, default=1.5)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--n_steps", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=Path, default=OUTPUTS_DIR / "ablation_mst.json")
    args = ap.parse_args()

    model, _ = load_checkpoint(args.ckpt, device=args.device)
    geo_c = model.node_proj.in_features - 2
    ids = json.loads(args.splits.read_text())[args.split]

    agg = {}
    for label, use_mst in (("GED (MST projector)", True), ("GED w/o MST projector", False)):
        rows, ncc = [], []
        for pid in ids:
            s = torch.load(args.patch_dir / f"patch_{pid:04d}.pt", weights_only=False)
            s["raster"] = s["raster"][:, :, :geo_c]
            res = predict_patch(model, s, knn=args.knn, edge_density=args.edge_density,
                                n_samples=args.n_samples, n_steps=args.n_steps,
                                device=args.device, use_mst=use_mst)
            pos = s["gt_pos"].numpy()
            n = pos.shape[0]
            G_gt, G = nx.Graph(), nx.Graph()
            G_gt.add_nodes_from(range(n))
            G.add_nodes_from(range(n))
            G_gt.add_edges_from(map(tuple, s["gt_edge_index"].numpy().T.tolist()))
            G.add_edges_from(map(tuple, res["pred_edges"].tolist()))
            rows.append(all_metrics(G_gt, G, pos={i: tuple(pos[i]) for i in range(n)}))
            ncc.append(nx.number_connected_components(G))
        agg[label] = {k: float(np.mean([r[k] for r in rows if np.isfinite(r[k])])) for k in KEYS}
        agg[label]["n_components_mean"] = float(np.mean(ncc))
        agg[label]["n_disconnected_patches"] = int(sum(c > 1 for c in ncc))
        agg[label]["n_patches"] = len(ids)
        r = agg[label]
        print(f"{label:<24} " + " ".join(f"{k}={r[k]:.3f}" for k in KEYS)
              + f" | n_cc={r['n_components_mean']:.2f} disconnected={r['n_disconnected_patches']}/{len(ids)}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(agg, indent=2))
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
