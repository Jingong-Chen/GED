#!/usr/bin/env bash
# Three-stage training as run for the paper (about 25 minutes on one RTX 5090).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/01_prepare_data.py
python scripts/02_train_base.py      --out runs/stage1_base
python scripts/03_finetune_phys.py   --warm_start runs/stage1_base/best.pt  --out runs/stage2_phys
python scripts/04_finetune_thermal.py --warm_start runs/stage2_phys/best.pt --out runs/stage3_thermal
python scripts/05_infer_full_texas.py --ckpt runs/stage3_thermal/best.pt --knn 15 --out outputs/retrained_full_texas.pt
python scripts/06_eval_birchfield.py --ours outputs/retrained_full_texas.pt
