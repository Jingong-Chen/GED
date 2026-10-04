"""Fig. 3: GT vs Schultz vs Watts-Strogatz vs GED on the 1,447-bus Texas backbone.

All panels share the same bus positions and voltage-tier colours; edges are
coloured by max(baseKV) of their endpoints. In panel (a), dashed faint lines
are GT branches outside the reference candidate set C (kNN 12 + MST).

    python scripts/figures/fig_baseline_compare.py --ours outputs/ged_full_texas.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as _nx
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree as _cKDTree

from ged.baselines import schultz_2014, watts_strogatz
from ged.paths import BACKBONE_DIR, OUTPUTS_DIR

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--ours", type=Path, default=OUTPUTS_DIR / "ged_full_texas.pt")
_ap.add_argument("--backbone", type=Path, default=BACKBONE_DIR)
_ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "fig_baseline_compare.pdf")
ARGS = _ap.parse_args()
SRC_BB = ARGS.backbone
OUT = ARGS.out
OUT.parent.mkdir(parents=True, exist_ok=True)

V_COLOR = {500: "#332288", 230: "#cc3333", 161: "#e8a93b", 115: "#a8c5e0"}
V_LW = {500: 1.0, 230: 0.6, 161: 0.4, 115: 0.25}
V_Z = {500: 4, 230: 3, 161: 2, 115: 1}

bus = pd.read_csv(SRC_BB / "bus.csv")
bus = bus.sort_values("bus_id").reset_index(drop=True)
N = len(bus)
ids = bus["bus_id"].astype(int).to_numpy()
pos_lonlat = bus[["lon", "lat"]].to_numpy().astype(float)
kvs = bus["baseKV"].astype(float).to_numpy()
bus_kv = dict(zip(ids, kvs))

# Normalised positions for baseline graph generation
pos_norm = pos_lonlat.copy()
pos_norm[:, 0] = (pos_norm[:, 0] - pos_norm[:, 0].min()) / (pos_norm[:, 0].max() - pos_norm[:, 0].min())
pos_norm[:, 1] = (pos_norm[:, 1] - pos_norm[:, 1].min()) / (pos_norm[:, 1].max() - pos_norm[:, 1].min())

# Build candidate set C (kNN ∪ MST) on bus positions for "reachable" mask
print("Building candidate set C (kNN k=12 + MST)...")
pos_for_C = pos_lonlat.copy()  # use lon,lat as the candidate-set space
N_bus = len(ids)
_kdt = _cKDTree(pos_for_C)
_, _knn_idx = _kdt.query(pos_for_C, k=13)
_id_to_local = {b: i for i, b in enumerate(ids)}
C_set = set()
for i in range(N_bus):
    for j in _knn_idx[i, 1:]:
        a, b = (ids[i], ids[int(j)]) if ids[i] < ids[int(j)] else (ids[int(j)], ids[i])
        C_set.add((a, b))
_G = _nx.Graph()
for i in range(N_bus):
    for j in range(i + 1, N_bus):
        d = float(np.hypot(pos_for_C[i, 0] - pos_for_C[j, 0],
                            pos_for_C[i, 1] - pos_for_C[j, 1]))
        _G.add_edge(ids[i], ids[j], weight=d)
for u, v in _nx.minimum_spanning_tree(_G, weight="weight").edges():
    a, b = (u, v) if u < v else (v, u)
    C_set.add((a, b))
print(f"  |C| = {len(C_set)}")

# GT branches  (track which are in C for reachable/unreachable shading)
gt_df = pd.read_csv(SRC_BB / "branch.csv")
gt_edges = []
n_gt_reachable = 0; n_gt_unreachable = 0
for _, br in gt_df.iterrows():
    u, v = int(br.from_bus), int(br.to_bus)
    vu = bus_kv.get(u); vv = bus_kv.get(v)
    if vu is None or vv is None: continue
    a, b = (u, v) if u < v else (v, u)
    in_C = (a, b) in C_set
    gt_edges.append((u, v, int(max(vu, vv)), in_C))
    if in_C: n_gt_reachable += 1
    else:    n_gt_unreachable += 1
print(f"  GT edges: reachable={n_gt_reachable}, unreachable={n_gt_unreachable}")

# Schultz
rng_s = np.random.RandomState(0)
G_s = schultz_2014(pos_norm, p=0.2, q=0.075, rng=rng_s)
schultz_edges = []
for i, j in G_s.edges():
    u, v = ids[i], ids[j]
    schultz_edges.append((u, v, int(max(bus_kv[u], bus_kv[v]))))

# Watts-Strogatz
rng_ws = np.random.RandomState(0)
G_ws = watts_strogatz(pos_norm, k=4, p=0.1, rng=rng_ws)
ws_edges = []
for i, j in G_ws.edges():
    u, v = ids[i], ids[j]
    ws_edges.append((u, v, int(max(bus_kv[u], bus_kv[v]))))

# GED
ged_out = torch.load(ARGS.ours, weights_only=False)
pred = ged_out["pred_edges"]
bid = ged_out["bus_id"]
if hasattr(bid, "numpy"): bid = bid.numpy()
seen = set()
ged_edges = []
for k in range(pred.shape[0]):
    u = int(bid[int(pred[k, 0])]); v = int(bid[int(pred[k, 1])])
    if u == v: continue
    key = (u, v) if u < v else (v, u)
    if key in seen: continue
    seen.add(key)
    vu = bus_kv.get(u); vv = bus_kv.get(v)
    if vu is None or vv is None: continue
    ged_edges.append((u, v, int(max(vu, vv))))

bus_pos = dict(zip(ids, pos_lonlat))


def draw(ax, edges, title, has_reachable_flag=False):
    """edges items: (u, v, kV) or (u, v, kV, in_C)."""
    ax.set_title(title, fontsize=13, pad=4)
    ax.set_xlim(-107.5, -93.0); ax.set_ylim(25.5, 36.7)
    ax.set_aspect(1 / np.cos(np.deg2rad(31)))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.18, lw=0.2)
    for item in edges:
        if has_reachable_flag:
            u, v, t, in_C = item
        else:
            u, v, t = item; in_C = True
        if t not in V_COLOR: continue
        a = bus_pos.get(u); b = bus_pos.get(v)
        if a is None or b is None: continue
        if in_C:
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    color=V_COLOR[t], lw=V_LW[t], alpha=0.80, zorder=V_Z[t])
        else:
            # GT edge outside the candidate set: render very faintly
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    color=V_COLOR[t], lw=V_LW[t] * 0.6, alpha=0.22,
                    linestyle=(0, (2.0, 1.4)), zorder=V_Z[t] - 0.5)
    ax.scatter(pos_lonlat[:, 0], pos_lonlat[:, 1], s=0.4,
               c="#566573", alpha=0.6, zorder=5)


fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
draw(axes[0, 0], gt_edges, "(a) GT", has_reachable_flag=True)
draw(axes[0, 1], schultz_edges, "(b) Schultz")
draw(axes[1, 0], ws_edges, "(c) Watts--Strogatz")
draw(axes[1, 1], ged_edges, "(d) GED")

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

handles = [Line2D([0], [0], color=V_COLOR[v], lw=1.8, label=f"{v} kV") for v in (500, 230, 161, 115)]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.2)

plt.tight_layout(pad=0.4, h_pad=2.5, w_pad=0.4)
plt.subplots_adjust(bottom=0.06)
plt.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
print(f"  GT edges: {len(gt_edges)}, Schultz: {len(schultz_edges)}, WS: {len(ws_edges)}, GED: {len(ged_edges)}")
