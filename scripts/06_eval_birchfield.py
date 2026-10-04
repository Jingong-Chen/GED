"""Birchfield-style graph metrics on the full 1,447-bus Texas backbone (Table I).

Compares GED against the ACTIVSg2000 ground truth and four classical baselines
built on the same bus positions (normalised to the Texas bounding box):
Random kNN (k=3), Geometric radius graph (eps=0.06), Watts-Strogatz (k=4,
p=0.1, seed 42) and Schultz et al. (p=0.2, q=0.075, seed 42). Parallel GT
branches are collapsed (1,992 unique bus pairs).

    python scripts/06_eval_birchfield.py --ours outputs/ged_full_texas_table1.pt
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

from ged.baselines import geometric_eps, random_knn, schultz_2014, watts_strogatz
from ged.metrics import all_metrics
from ged.paths import BACKBONE_DIR, OUTPUTS_DIR, TX_BBOX

METRIC_KEYS = ["degree_w1", "avg_clustering_diff", "mean_path_diff",
               "diameter_diff", "spectral_gap_diff", "edge_length_w1"]
SHORT = {
    "degree_w1": r"$W_1(\text{deg})$", "avg_clustering_diff": r"$|\Delta C|$",
    "mean_path_diff": r"$|\Delta L|$", "diameter_diff": r"$|\Delta D|$",
    "spectral_gap_diff": r"$|\Delta \lambda_2|$", "edge_length_w1": r"$W_1(\ell)$",
}


def load_backbone(backbone_dir):
    bus = pd.read_csv(backbone_dir / "bus.csv").sort_values("bus_id").reset_index(drop=True)
    branch = pd.read_csv(backbone_dir / "branch.csv")
    n = len(bus)
    id2loc = {int(b): i for i, b in enumerate(bus["bus_id"].values)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in zip(branch.from_bus, branch.to_bus):
        u, v = id2loc[int(u)], id2loc[int(v)]
        if u != v:
            G.add_edge(min(u, v), max(u, v))
    pos = np.stack([
        (bus["lon"].values - TX_BBOX["lon_min"]) / (TX_BBOX["lon_max"] - TX_BBOX["lon_min"]),
        (bus["lat"].values - TX_BBOX["lat_min"]) / (TX_BBOX["lat_max"] - TX_BBOX["lat_min"]),
    ], axis=1).astype(np.float32)
    return G, pos, id2loc


def ours_graph(path, n, id2loc):
    ours = torch.load(path, weights_only=False)
    bus_id = np.asarray(ours["bus_id"])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in ours["pred_edges"]:
        a, b = id2loc.get(int(bus_id[u]), -1), id2loc.get(int(bus_id[v]), -1)
        if a >= 0 and b >= 0 and a != b:
            G.add_edge(a, b)
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, required=True, help="Output of 05_infer_full_texas.py")
    ap.add_argument("--backbone", type=Path, default=BACKBONE_DIR)
    ap.add_argument("--geometric_eps", type=float, default=0.06)
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--out_json", type=Path, default=OUTPUTS_DIR / "table1_birchfield.json")
    ap.add_argument("--out_tex", type=Path, default=OUTPUTS_DIR / "table1_birchfield.tex")
    args = ap.parse_args()

    G_gt, pos, id2loc = load_backbone(args.backbone)
    n = len(pos)
    pos_dict = {i: tuple(pos[i]) for i in range(n)}
    print(f"GT: {n} buses, {G_gt.number_of_edges()} unique branches, "
          f"{nx.number_connected_components(G_gt)} component(s)")

    methods = {}
    if not args.skip_baselines:
        methods["Random+kNN"] = random_knn(pos, k=3)
        methods["Geometric"] = geometric_eps(pos, eps=args.geometric_eps)
        methods["Watts-Strogatz"] = watts_strogatz(pos, k=4, p=0.1, rng=np.random.RandomState(42))
        methods["Schultz"] = schultz_2014(pos, p=0.2, q=0.075, rng=np.random.RandomState(42))
    methods["GED"] = ours_graph(args.ours, n, id2loc)

    summary = {}
    for name, G in methods.items():
        m = all_metrics(G_gt, G, pos=pos_dict)
        m["n_components"] = nx.number_connected_components(G)
        summary[name] = m

    print(f"\n{'Method':<16}" + "".join(f"{k:>20}" for k in METRIC_KEYS) + f"{'n_cc':>6}")
    for name, m in summary.items():
        print(f"{name:<16}" + "".join(f"{m[k]:>20.3f}" for k in METRIC_KEYS) + f"{m['n_components']:>6}")

    best = {}
    for k in METRIC_KEYS:
        vals = {nm: summary[nm][k] for nm in summary if np.isfinite(summary[nm][k])}
        best[k] = min(vals, key=vals.get) if vals else None
    lines = [r"\begin{tabular}{l" + "c" * (len(METRIC_KEYS) + 1) + "}", r"\toprule",
             "Method & " + " & ".join(SHORT[k] for k in METRIC_KEYS) + r" & $n_{\text{cc}}$ \\", r"\midrule"]
    for name, m in summary.items():
        cells = []
        for k in METRIC_KEYS:
            cell = f"{m[k]:.3f}"
            cells.append(r"\textbf{" + cell + "}" if best[k] == name else cell)
        lines.append(name + " & " + " & ".join(cells) + f" & {m['n_components']}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines) + "\n")
    args.out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {args.out_json} and {args.out_tex}")


if __name__ == "__main__":
    main()
