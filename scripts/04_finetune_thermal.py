"""Stage 3: thermal-aware fine-tuning, warm-started from stage 2 (final GED model).

Attaches a 1-D MVA-rating head to the shared pair trunk and adds
  0.3  * MSE on z-normalised log10 rating (positive pairs)
  0.15 * L_therm = mean max(0, f_uv / R_uv - 0.8)^2, f_uv = 100 b_uv |theta_u - theta_v| MVA
on top of the stage-2 objective. AdamW lr 5e-5, 20 epochs. The released
weights/ged_texas.pt is the best validation epoch (epoch 2, val 1.2263).

    python scripts/04_finetune_thermal.py --warm_start runs/stage2_phys/best.pt --out runs/stage3_thermal
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ged.data import ThermalDataset, collect_files, thermal_collate
from ged.diffusion import categorical_q_sample
from ged.losses import dc_pf_loss, thermal_loss
from ged.model import EdgeDiffusionModel, ThermalModel, fiedler_regularizer
from ged.paths import GLOBAL_FILE, PATCH_1DEG_DIR, PATCH_4DEG_DIR, PATCH_7DEG_DIR, REPO_ROOT, SPLITS_FILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm_start", default=str(REPO_ROOT / "runs" / "stage2_phys" / "best.pt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "stage3_thermal"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_n", type=int, default=1536)
    ap.add_argument("--n_pairs", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--n_geo_channels", type=int, default=28)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--lambda_attr", type=float, default=0.5)
    ap.add_argument("--lambda_fied", type=float, default=0.20)
    ap.add_argument("--lambda_phys", type=float, default=0.10)
    ap.add_argument("--lambda_rate", type=float, default=0.30,
                    help="MSE on predicted rateA (z-normalised log10 MVA)")
    ap.add_argument("--lambda_thermal", type=float, default=0.15)
    ap.add_argument("--thermal_target", type=float, default=0.80,
                    help="Penalise edges whose loading exceeds this fraction (1.0 = at limit)")
    ap.add_argument("--phys_max_n", type=int, default=120)
    ap.add_argument("--class_w", type=float, nargs=4, default=[1.0, 3.0, 5.0, 6.0])
    ap.add_argument("--splits", type=Path, default=SPLITS_FILE)
    ap.add_argument("--patch_1deg", type=Path, default=PATCH_1DEG_DIR)
    ap.add_argument("--patch_4deg", type=Path, default=PATCH_4DEG_DIR)
    ap.add_argument("--patch_7deg", type=Path, default=PATCH_7DEG_DIR)
    ap.add_argument("--global_file", type=Path, default=GLOBAL_FILE)
    ap.add_argument("--stop_on_nan", type=int, default=1,
                    help="Stop when the validation loss becomes NaN (the best checkpoint is unaffected)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    log_path.write_text(f"[thermal] args: {vars(args)}\n")

    train_files, val_files = collect_files(args.splits, args.patch_1deg, args.patch_4deg, args.patch_7deg, args.global_file)
    print(f"[thermal] train={len(train_files)} val={len(val_files)}", flush=True)
    train_ds = ThermalDataset(train_files, max_n=args.max_n, n_pairs=args.n_pairs,
                                n_geo_channels=args.n_geo_channels)
    val_ds = ThermalDataset(val_files, max_n=args.max_n, n_pairs=args.n_pairs,
                              n_geo_channels=args.n_geo_channels)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, collate_fn=thermal_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=thermal_collate)

    base = EdgeDiffusionModel(geo_c=args.n_geo_channels, hidden=args.hidden,
                              n_layers=args.n_layers, n_heads=args.n_heads).to(args.device)
    # Warm start base from the phys checkpoint
    ck = torch.load(args.warm_start, map_location=args.device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    base.load_state_dict(sd, strict=True)
    print(f"[thermal] warm-started base from {args.warm_start} "
          f"(ep={ck.get('epoch','?')}, val={ck.get('best_val','?')})", flush=True)
    model = ThermalModel(base).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_w = torch.tensor(args.class_w, device=args.device, dtype=torch.float32)

    t0 = time.time(); best_val = float("inf")
    for ep in range(args.epochs):
        model.train()
        tr_topo, tr_attr, tr_fied, tr_phys, tr_rate, tr_therm, tr_acc = [], [], [], [], [], [], []
        for batch in train_dl:
            opt.zero_grad()
            geo = batch["geo"].to(args.device); pos = batch["pos"].to(args.device)
            ii = batch["i_idx"].to(args.device); jj = batch["j_idx"].to(args.device)
            x_0 = batch["x_0"].to(args.device); pm = batch["pair_mask"].to(args.device)
            attrs_gt = batch["attrs"].to(args.device)
            attr_mask = batch["attr_mask"].to(args.device)
            n_real = batch["n_real"].to(args.device)
            p_inj = batch["p_inj"].to(args.device)
            rate_gt = batch["rate_z"].to(args.device)
            Bsz, P = x_0.shape
            t = torch.rand(Bsz, device=args.device)
            x_t = categorical_q_sample(x_0, t, num_classes=4)
            cls_logits, attr_pred, rate_pred = model(geo, pos, ii, jj, x_t, t)

            loss_topo_full = F.cross_entropy(
                cls_logits.reshape(-1, 4), x_0.reshape(-1),
                weight=class_w, reduction="none").reshape(Bsz, P)
            loss_topo = (loss_topo_full * pm).sum() / pm.sum().clamp(min=1)

            attr_diff = (attr_pred - attrs_gt) ** 2
            attr_mask_f = attr_mask.float().unsqueeze(-1)
            loss_attr = (attr_diff * attr_mask_f).sum() / attr_mask_f.sum().clamp(min=1) / 3.0

            # Rate regression (only on positive pairs)
            rate_diff = (rate_pred - rate_gt) ** 2
            loss_rate = (rate_diff * attr_mask.float()).sum() / attr_mask.float().sum().clamp(min=1)

            edge_prob = (1.0 - cls_logits.softmax(dim=-1)[..., 0])
            loss_fied = fiedler_regularizer(edge_prob, ii, jj, n_real, eps_target=0.04)
            loss_phys = dc_pf_loss(edge_prob, attr_pred, ii, jj, n_real, p_inj,
                                    max_n=args.phys_max_n)
            loss_thermal = thermal_loss(edge_prob, attr_pred, rate_pred, ii, jj,
                                          n_real, p_inj, max_n=args.phys_max_n,
                                          target=args.thermal_target)

            loss = (loss_topo
                    + args.lambda_attr * loss_attr
                    + args.lambda_fied * loss_fied
                    + args.lambda_phys * loss_phys
                    + args.lambda_rate * loss_rate
                    + args.lambda_thermal * loss_thermal)
            if not torch.isfinite(loss):
                # Skip non-finite batches (most often from thermal/dc-pf
                # numerical instability) instead of letting NaN propagate.
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                tr_topo.append(loss_topo.item()); tr_attr.append(loss_attr.item())
                tr_fied.append(loss_fied.item()); tr_phys.append(loss_phys.item())
                tr_rate.append(loss_rate.item()); tr_therm.append(loss_thermal.item())
                pred = cls_logits.argmax(dim=-1)
                tr_acc.append(((pred == x_0) & pm).float().sum().item() / pm.sum().clamp(min=1).item())

        model.eval()
        v_topo, v_attr, v_fied, v_phys, v_rate, v_therm = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_dl:
                geo = batch["geo"].to(args.device); pos = batch["pos"].to(args.device)
                ii = batch["i_idx"].to(args.device); jj = batch["j_idx"].to(args.device)
                x_0 = batch["x_0"].to(args.device); pm = batch["pair_mask"].to(args.device)
                attrs_gt = batch["attrs"].to(args.device)
                attr_mask = batch["attr_mask"].to(args.device)
                n_real = batch["n_real"].to(args.device)
                p_inj = batch["p_inj"].to(args.device)
                rate_gt = batch["rate_z"].to(args.device)
                Bsz, P = x_0.shape
                t = torch.rand(Bsz, device=args.device)
                x_t = categorical_q_sample(x_0, t, num_classes=4)
                cls_logits, attr_pred, rate_pred = model(geo, pos, ii, jj, x_t, t)
                lt = (F.cross_entropy(cls_logits.reshape(-1, 4), x_0.reshape(-1),
                                       weight=class_w, reduction="none").reshape(Bsz, P) * pm).sum() / pm.sum().clamp(min=1)
                ad = (attr_pred - attrs_gt) ** 2
                am = attr_mask.float().unsqueeze(-1)
                la = (ad * am).sum() / am.sum().clamp(min=1) / 3.0
                rd = ((rate_pred - rate_gt) ** 2 * attr_mask.float()).sum() / attr_mask.float().sum().clamp(min=1)
                ep_prob = (1.0 - cls_logits.softmax(dim=-1)[..., 0])
                lf = fiedler_regularizer(ep_prob, ii, jj, n_real, eps_target=0.04)
                lp = dc_pf_loss(ep_prob, attr_pred, ii, jj, n_real, p_inj, max_n=args.phys_max_n)
                lh = thermal_loss(ep_prob, attr_pred, rate_pred, ii, jj, n_real, p_inj,
                                    max_n=args.phys_max_n, target=args.thermal_target)
                v_topo.append(lt.item()); v_attr.append(la.item())
                v_fied.append(lf.item()); v_phys.append(lp.item())
                v_rate.append(rd.item()); v_therm.append(lh.item())

        val_total = (np.mean(v_topo)
                     + args.lambda_attr * np.mean(v_attr)
                     + args.lambda_fied * np.mean(v_fied)
                     + args.lambda_phys * np.mean(v_phys)
                     + args.lambda_rate * np.mean(v_rate)
                     + args.lambda_thermal * np.mean(v_therm))
        msg = (f"ep{ep:03d} train: topo={np.mean(tr_topo):.4f} attr={np.mean(tr_attr):.4f} "
               f"fied={np.mean(tr_fied):.4f} phys={np.mean(tr_phys):.5f} "
               f"rate={np.mean(tr_rate):.4f} therm={np.mean(tr_therm):.5f} acc={np.mean(tr_acc):.3f} | "
               f"val: total={val_total:.4f} rate={np.mean(v_rate):.4f} therm={np.mean(v_therm):.5f} | "
               f"t={time.time()-t0:.0f}s")
        print(msg, flush=True)
        with log_path.open("a") as fh: fh.write(msg + "\n")

        if args.stop_on_nan and not np.isfinite(val_total):
            print(f"[thermal] validation loss is not finite at ep{ep}; stopping (best checkpoint kept)", flush=True)
            break
        if val_total < best_val:
            best_val = val_total
            torch.save({"args": vars(args), "model_state_dict": model.state_dict(),
                        "epoch": ep, "best_val": best_val,
                        "has_rate_head": True}, out / "best.pt")
            print(f"  [saved best ep{ep} val={best_val:.4f}]", flush=True)

    print(f"[thermal] done, best_val={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
