#!/usr/bin/env bash
# Reproduce the paper's tables and figures from the released weights.
# Usage: bash scripts/reproduce_paper.sh [--skip_n1]
set -euo pipefail
cd "$(dirname "$0")/.."
N1_FLAG="${1:-}"

python scripts/01_prepare_data.py

# Table I (GED row): stage-1 weights, kNN 18
python scripts/05_infer_full_texas.py --preset table1 --out outputs/ged_full_texas_table1.pt
python scripts/06_eval_birchfield.py --ours outputs/ged_full_texas_table1.pt

# Final model: Table II, Table III, Fig. 3
python scripts/05_infer_full_texas.py --preset final --out outputs/ged_full_texas.pt
python scripts/07_tier_table.py --ours outputs/ged_full_texas.pt --count_self_loops
python scripts/figures/fig_baseline_compare.py --ours outputs/ged_full_texas.pt
python scripts/figures/fig_raster_channels.py

# Sec. IV-G ablation
python scripts/09_ablation_mst.py --ckpt weights/ged_texas_stage1.pt

# Table III (about 1 hour with N-1; pass --skip_n1 for a few seconds)
python scripts/08_ac_powerflow.py --ours outputs/ged_full_texas.pt ${N1_FLAG}
