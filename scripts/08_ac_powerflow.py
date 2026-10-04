"""AC power flow and N-1 on the full 2,000-bus ACTIVSg2000 case (Table III).

Variants (same protocol, see ged/acpf.py):
  GT        ACTIVSg2000 backbone branches (impedance length-scaled like the others)
  GED       predicted same-tier edges, tier-median impedance      (paper row "GED")
  GED-rxb   predicted same-tier edges, learned (r, x, b) head     (optional)
  Schultz   Schultz et al. backbone on the same buses, tier-median impedance

Reported: convergence, % buses in [0.94, 1.06] and [0.95, 1.05] p.u., V_min,
active losses (% of generation), % lines above 100% loading, and N-1 over 200
randomly sampled backbone outages (seed 0): convergence rate and mean number of
new voltage-band violations per outage. The N-1 study takes roughly 15-20
minutes per variant on one CPU core; use --skip_n1 for a quick check.

    python scripts/08_ac_powerflow.py --ours outputs/ged_full_texas.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
from matpowercaseframes import CaseFrames

from ged.acpf import (
    add_per_tier_mst,
    build_full_pp,
    build_gt_branches,
    build_ours_subset,
    build_schultz_subset,
    compute_metrics,
    n1_contingency,
    try_ac,
)
from ged.paths import ACTIVSG_CASE, BACKBONE_DIR, OUTPUTS_DIR

ALL_VARIANTS = ["GT", "GED", "GED-rxb", "Schultz"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, required=True, help="Output of 05_infer_full_texas.py")
    ap.add_argument("--case", type=Path, default=ACTIVSG_CASE)
    ap.add_argument("--backbone", type=Path, default=BACKBONE_DIR)
    ap.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=["GT", "GED", "Schultz"])
    ap.add_argument("--n1_max", type=int, default=200, help="-1 = trip every backbone line")
    ap.add_argument("--skip_n1", action="store_true")
    ap.add_argument("--load_scale", type=float, default=1.0)
    ap.add_argument("--out_json", type=Path, default=OUTPUTS_DIR / "table3_acpf.json")
    args = ap.parse_args()

    bus_csv = args.backbone / "bus.csv"
    bus_bb_df = pd.read_csv(bus_csv).sort_values("bus_id").reset_index(drop=True)
    backbone_buses = set(bus_bb_df["bus_id"].astype(int).tolist())
    cf = CaseFrames(str(args.case))
    bus_df = cf.bus.reset_index()
    bus_v_all = dict(zip(bus_df["BUS_I"].astype(int), bus_df["BASE_KV"].astype(float)))

    variants = []
    for name in args.variants:
        if name == "GT":
            subset = build_gt_branches(cf, backbone_buses, bus_v_all)
        elif name in ("GED", "GED-rxb"):
            subset = build_ours_subset(args.ours, bus_v_all, use_learned_attrs=(name == "GED-rxb"))
            subset, added = add_per_tier_mst(subset, bus_bb_df)
            print(f"  {name}: per-tier MST closure added {added} lines", flush=True)
        else:
            subset = build_schultz_subset(bus_bb_df, bus_v_all)
            subset, added = add_per_tier_mst(subset, bus_bb_df)
            print(f"  {name}: per-tier MST closure added {added} lines", flush=True)
        variants.append((name, subset))

    results = {}
    for name, subset in variants:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        net, _, _, _ = build_full_pp(subset, cf, scale_load=args.load_scale, backbone_csv=bus_csv)
        ok, relax = try_ac(net, max_relax=1)
        info = {"n_buses": len(net.bus), "n_lines": len(net.line), "n_trafos": len(net.trafo),
                "converged": ok, "relax_rounds": relax, "time_s": time.time() - t0}
        if ok:
            info.update(compute_metrics(net, bus_v_all))
            print(f"  V in [0.94,1.06]: {info['v_in_band_pct']:.1f}%  V in [0.95,1.05]: "
                  f"{info['v_in_strict_pct']:.1f}%  Vmin: {info['v_min']:.3f}  "
                  f"Ploss: {info['p_loss_pct']:.1f}%  lines>100%: {info['pct_lines_overloaded']:.1f}%", flush=True)
            for tier, td in info["per_tier"].items():
                print(f"    {tier}: n={td['n']}, V in band={td['v_in_band_pct']:.1f}%", flush=True)
        if ok and not args.skip_n1:
            print(f"  N-1 over {args.n1_max} sampled backbone outages ...", flush=True)
            t1 = time.time()
            n1 = n1_contingency(lambda s=subset: build_full_pp(s, cf, scale_load=1.0, backbone_csv=bus_csv)[0],
                                max_lines=args.n1_max)
            n1["time_s"] = time.time() - t1
            info["n1"] = n1
            print(f"  N-1 converged {n1['convergence_rate_pct']:.1f}%, mean new V violations "
                  f"{n1['mean_new_v_violations_per_n1']:.2f}", flush=True)
        results[name] = info
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=str))

    print("\nVariant   Conv  [0.94,1.06]  [0.95,1.05]  Vmin   Ploss  L>100%  N1-conv  N1-dV")
    for name, r in results.items():
        if not r["converged"]:
            print(f"{name:<9} no")
            continue
        n1 = r.get("n1", {})
        print(f"{name:<9} yes   {r['v_in_band_pct']:>9.1f}%  {r['v_in_strict_pct']:>9.1f}%  {r['v_min']:.3f}  "
              f"{r['p_loss_pct']:>4.1f}%  {r['pct_lines_overloaded']:>5.1f}%  "
              f"{n1.get('convergence_rate_pct', float('nan')):>6.1f}%  "
              f"{n1.get('mean_new_v_violations_per_n1', float('nan')):>5.2f}")
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
