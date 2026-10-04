"""Fig. 2: the 28 conditioning channels over Texas, with an example test patch.

    python scripts/figures/fig_raster_channels.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from ged.paths import OUTPUTS_DIR, PATCH_1DEG_DIR, RASTER_FILE, SPLITS_FILE

mpl.rcParams.update({
    "font.family": "serif", "font.size": 6, "axes.titlesize": 6,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "fig_raster_channels.pdf")
ARGS = _ap.parse_args()
OUT = ARGS.out
OUT.parent.mkdir(parents=True, exist_ok=True)
TX_FEAT = RASTER_FILE
PATCH_DIR = PATCH_1DEG_DIR

TX_BBOX = (25.84, -106.65, 36.5, -93.51)  # S, W, N, E

CH_NAMES = [
    "road_density", "road_avg_degree", "road_type_motorway", "road_type_trunk",
    "road_type_primary", "road_type_secondary", "road_type_tertiary", "road_type_residential",
    "building_density", "building_type_residential", "building_type_commercial",
    "building_type_industrial", "building_type_other", "population_density",
    "landuse_residential", "landuse_commercial", "landuse_industrial", "landuse_farmland",
    "landuse_forest", "landuse_water", "landuse_other", "dist_to_plant",
    "nearest_plant_capacity", "plant_density", "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "solar_ghi", "wind_cf", "hydro_proximity", "biomass_density",
    "hifld_tx_proximity", "betweenness_corridor",
]
CH_LABELS = {
    "road_density": "road dens.", "road_avg_degree": "road deg.",
    "road_type_motorway": "motorway", "road_type_trunk": "trunk",
    "road_type_primary": "primary", "road_type_secondary": "secondary",
    "road_type_tertiary": "tertiary", "road_type_residential": "road-res",
    "building_density": "bldg dens.", "building_type_residential": "bldg-res",
    "building_type_commercial": "bldg-com", "building_type_industrial": "bldg-ind",
    "building_type_other": "bldg-oth", "population_density": "population",
    "landuse_residential": "lu-res", "landuse_commercial": "lu-com",
    "landuse_industrial": "lu-ind", "landuse_farmland": "farmland",
    "landuse_forest": "forest", "landuse_water": "water",
    "landuse_other": "lu-oth", "dist_to_plant": "dist-plant",
    "nearest_plant_capacity": "plant cap.", "plant_density": "plant dens.",
    "lat_sin": "lat sin", "lat_cos": "lat cos", "lon_sin": "lon sin", "lon_cos": "lon cos",
    "solar_ghi": "solar GHI", "wind_cf": "wind CF",
    "hydro_proximity": "hydro", "biomass_density": "biomass",
    "hifld_tx_proximity": "HIFLD tx", "betweenness_corridor": "btw corr.",
}

ROAD_GROUP    = set(range(0, 8))    # 0-7 road
BLDG_GROUP    = set(range(8, 13))   # 8-12 buildings
POP_GROUP     = {13}                # 13 population
LAND_GROUP    = set(range(14, 21))  # 14-20 landuse
PLANT_GROUP   = set(range(21, 24))  # 21-23 plants
POSENC_GROUP  = set(range(24, 28))  # 24-27 position
RENEW_GROUP   = set(range(28, 32))  # 28-31 renewables
INFRA_GROUP   = set(range(32, 34))  # 32-33 hifld/btw

CATEGORY_COLOR = {
    "ROAD":   "#1B7A3C",  # green
    "BLDG":   "#C46A1B",  # orange
    "POP":    "#A93226",  # red
    "LAND":   "#2874A6",  # blue
    "PLANT":  "#6C3483",  # purple
    "POSENC": "#566573",  # gray
    "RENEW":  "#B7950B",  # dark yellow
    "INFRA":  "#1A5276",  # dark blue
}


def category_of(k):
    if k in ROAD_GROUP:   return "ROAD"
    if k in BLDG_GROUP:   return "BLDG"
    if k in POP_GROUP:    return "POP"
    if k in LAND_GROUP:   return "LAND"
    if k in PLANT_GROUP:  return "PLANT"
    if k in POSENC_GROUP: return "POSENC"
    if k in RENEW_GROUP:  return "RENEW"
    if k in INFRA_GROUP:  return "INFRA"
    return "POSENC"


def cmap_of(cat):
    return {
        "ROAD":   "Greens",
        "BLDG":   "Oranges",
        "POP":    "Reds",
        "LAND":   "Blues",
        "PLANT":  "Purples",
        "POSENC": "RdBu",
        "RENEW":  "plasma",
        "INFRA":  "magma",
    }.get(cat, "viridis")


d = np.load(TX_FEAT)
feats = d["features"]
H, W, C = feats.shape
S, Wlon, N, E = TX_BBOX
extent_tx = (Wlon, E, S, N)

splits = json.loads(SPLITS_FILE.read_text())
test_ids = splits["test"]
candidates = []
for pid in test_ids:
    s = torch.load(PATCH_DIR / f"patch_{pid:04d}.pt", weights_only=False)
    if 25 <= s["gt_pos"].shape[0] <= 90:
        candidates.append((pid, s))
candidates.sort(key=lambda x: -x[1]["gt_pos"].shape[0])
ex_pid, ex_data = candidates[0]
ex_meta = ex_data["patch_meta"]
ex_bbox = (ex_meta["lat_min"], ex_meta["lon_min"], ex_meta["lat_max"], ex_meta["lon_max"])

N_CHANNELS = 28  # NAPS pre-cursor scope: drop renewables (28-31) + HIFLD/btw (32-33)

fig = plt.figure(figsize=(7.4, 3.15))
gs = fig.add_gridspec(4, 7, hspace=0.30, wspace=0.10,
                      left=0.02, right=0.98, top=0.97, bottom=0.05)

# channel 27 (lon_cos) is skipped: its cell gs[3,6] holds the GT panel
for k in range(N_CHANNELS - 1):
    r, c = k // 7, k % 7
    ax = fig.add_subplot(gs[r, c])
    arr = feats[:, :, k].astype(np.float32)
    cat = category_of(k)
    cmap = cmap_of(cat)
    name = CH_NAMES[k]
    if any(s in name for s in ("density", "population", "capacity")):
        arr = np.log1p(np.clip(arr, 0, None))
    if cat == "POSENC":
        vmin = np.nanpercentile(arr, 2); vmax = np.nanpercentile(arr, 98)
        v_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -v_abs, v_abs
    else:
        vmin = float(np.nanpercentile(arr, 2))
        vmax = float(np.nanpercentile(arr, 99))
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9
    ax.imshow(arr, extent=extent_tx, origin="lower", cmap=cmap,
              vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((ex_bbox[1], ex_bbox[0]),
                           ex_bbox[3] - ex_bbox[1], ex_bbox[2] - ex_bbox[0],
                           fill=False, edgecolor="#FFD300", lw=0.55))
    ax.set_xlim(Wlon, E); ax.set_ylim(S, N)
    label = CH_LABELS.get(name, name)
    ax.set_title(f"{k}. {label}", fontsize=5.6, color=CATEGORY_COLOR[cat], pad=0.8)
    ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[3, 6])
pos = ex_data["gt_pos"].numpy()
ei = ex_data["gt_edge_index"].numpy()
voltage = ex_data["gt_voltage"].numpy()
lat_min, lon_min, lat_max, lon_max = ex_bbox
node_lat = lat_min + pos[:, 1] * (lat_max - lat_min)
node_lon = lon_min + pos[:, 0] * (lon_max - lon_min)
for k in range(ei.shape[1]):
    u, v = int(ei[0, k]), int(ei[1, k])
    L = float(np.hypot(node_lon[u] - node_lon[v], node_lat[u] - node_lat[v]))
    color = "#C0392B" if L > 0.5 else "#E67E22" if L > 0.15 else "#2E86C1"
    ax.plot([node_lon[u], node_lon[v]], [node_lat[u], node_lat[v]],
            color=color, lw=0.45, alpha=0.92)
sizes = np.where(voltage >= 500, 6, np.where(voltage >= 230, 3.5, 1.6))
colors = np.where(voltage >= 500, "#5B2C6F", np.where(voltage >= 230, "#E74C3C", "#2874A6"))
ax.scatter(node_lon, node_lat, s=sizes, c=colors, edgecolors="white",
           linewidths=0.15, alpha=0.95, zorder=5)
ax.add_patch(Rectangle((ex_bbox[1], ex_bbox[0]),
                       ex_bbox[3] - ex_bbox[1], ex_bbox[2] - ex_bbox[0],
                       fill=False, edgecolor="#FFD300", lw=0.8))
ax.set_xlim(ex_bbox[1] - 0.03, ex_bbox[3] + 0.03)
ax.set_ylim(ex_bbox[0] - 0.03, ex_bbox[2] + 0.03)
ax.set_aspect("auto")
ax.set_title("GT", fontsize=5.6, color="black", pad=0.8)
ax.set_xticks([]); ax.set_yticks([])

leg_handles = []
from matplotlib.patches import Patch

for cat in ("ROAD", "BLDG", "POP", "LAND", "PLANT", "POSENC"):
    leg_handles.append(Patch(facecolor=CATEGORY_COLOR[cat], edgecolor="none", label={
        "ROAD": "road (0-7)", "BLDG": "buildings (8-12)", "POP": "population (13)",
        "LAND": "landuse (14-20)", "PLANT": "plants (21-23)", "POSENC": "lat/lon enc. (24-27)",
    }[cat]))
fig.legend(handles=leg_handles, loc="lower center", ncol=6, fontsize=6.0,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.savefig(OUT)
print(f"saved {OUT}")
