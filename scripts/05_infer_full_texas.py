"""Single-pass inference on the full 1,447-bus Texas backbone with released weights.

The output .pt holds the predicted edge list (local indices into the global
sample), per-edge voltage tier, impedance, edge scores, the candidate set, and
the ACTIVSg2000 bus ids (key "bus_id").

Presets reproduce the paper's outputs exactly (same GPU / PyTorch builds; see README):
  --preset final   weights/ged_texas.pt,        kNN 15, density 1.7  (Tables II, III, Fig. 3)
  --preset table1  weights/ged_texas_stage1.pt, kNN 18, density 1.7  (Table I GED row)
Both use 5 reverse chains of 30 steps and pad N to 1500.

    python scripts/05_infer_full_texas.py --preset final  --out outputs/ged_full_texas.pt
    python scripts/05_infer_full_texas.py --preset table1 --out outputs/ged_full_texas_table1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from ged.inference import predict_full_texas
from ged.model import load_checkpoint
from ged.paths import GLOBAL_FILE, OUTPUTS_DIR, WEIGHTS_DIR

PRESETS = {
    "final": {"ckpt": WEIGHTS_DIR / "ged_texas.pt", "knn": 15},
    "table1": {"ckpt": WEIGHTS_DIR / "ged_texas_stage1.pt", "knn": 18},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="final")
    ap.add_argument("--ckpt", type=Path, default=None, help="Override the preset checkpoint")
    ap.add_argument("--global_file", type=Path, default=GLOBAL_FILE)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "ged_full_texas.pt")
    ap.add_argument("--knn", type=int, default=None, help="kNN size of the candidate set (preset default)")
    ap.add_argument("--edge_density", type=float, default=1.7,
                    help="Output edge budget round(edge_density * N), MST edges included")
    ap.add_argument("--n_samples", type=int, default=5, help="Independent reverse chains averaged")
    ap.add_argument("--n_steps", type=int, default=30, help="Reverse steps T")
    ap.add_argument("--max_n", type=int, default=1500, help="Node padding length (keep 1500 to reproduce)")
    ap.add_argument("--use_mst", type=int, default=1, help="1 = MST connectivity projector on")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    ckpt_path = args.ckpt or preset["ckpt"]
    knn = args.knn if args.knn is not None else preset["knn"]
    if not Path(args.global_file).exists():
        sys.exit(f"{args.global_file} not found. Run: python scripts/01_prepare_data.py")

    model, ckpt = load_checkpoint(ckpt_path, device=args.device)
    print(f"Loaded {ckpt_path} (stage={ckpt.get('stage', '?')}, epoch={ckpt.get('epoch', '?')})")
    geo_c = model.node_proj.in_features - 2

    sample = torch.load(args.global_file, weights_only=False)
    sample["raster"] = sample["raster"][:, :, :geo_c]
    res = predict_full_texas(model, sample, knn=knn, edge_density=args.edge_density,
                             n_samples=args.n_samples, n_steps=args.n_steps, max_n=args.max_n,
                             device=args.device, use_mst=bool(args.use_mst))
    res["settings"]["ckpt"] = Path(ckpt_path).name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, args.out)
    print(f"Saved {res['n_pred_edges']} edges over {res['n_nodes']} buses -> {args.out}")


if __name__ == "__main__":
    main()
