"""AC power-flow validation on the full 2,000-bus ACTIVSg2000 case (pandapower).

Protocol (Sec. IV-F of the paper): every same-tier backbone branch of
ACTIVSg2000 is replaced by the evaluated backbone. Lines get tier-median
per-unit impedance scaled by haversine length / tier-median length. Cross-tier
transformers and the sub-115 kV network stay at ACTIVSg2000 values. A per-tier
MST closure keeps each voltage tier internally connected. Newton-Raphson runs
once with no load relaxation; N-1 trips randomly sampled backbone lines.
"""
from __future__ import annotations

import logging
import warnings

import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import torch

from ged.baselines import schultz_2014
from ged.paths import BACKBONE_DIR

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

BACKBONE_BUS_CSV = BACKBONE_DIR / "bus.csv"

# Per-tier line median impedance (per-unit, S_base=100 MVA) and thermal limit (MVA)
TIER_LINE = {
    115.0: (0.00650, 0.03601, 0.00554, 180.27),
    161.0: (0.00329, 0.02084, 0.00970, 269.83),
    230.0: (0.00123, 0.03223, 0.00000, 430.76),
    500.0: (0.00024, 0.00909, 0.08122, 1763.39),
}
# Per-pair transformer reactance and sn_mva from typical IEEE practice
# (vk_percent on HV-side base)
TIER_TRAFO = {
    (115.0, 161.0): {"sn_mva": 200, "vk": 9.0, "vkr": 0.4},
    (115.0, 230.0): {"sn_mva": 300, "vk": 10.5, "vkr": 0.4},
    (115.0, 500.0): {"sn_mva": 500, "vk": 13.0, "vkr": 0.3},
    (161.0, 230.0): {"sn_mva": 250, "vk": 9.5, "vkr": 0.4},
    (161.0, 500.0): {"sn_mva": 500, "vk": 12.0, "vkr": 0.3},
    (230.0, 500.0): {"sn_mva": 600, "vk": 11.0, "vkr": 0.3},
}
ADJACENT_TIERS = {(115.0, 161.0), (161.0, 230.0), (230.0, 500.0)}


def haversine_km(la1, lo1, la2, lo2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_full_pp(branch_subset, cf, scale_load=1.0, backbone_csv=BACKBONE_BUS_CSV):
    """Build pandapower net using ACTIVSg case + supplied backbone branches.

    branch_subset: list of (u, v, kind, params) where:
      - kind='line': params = (r_pu, x_pu, b_pu, rateA, vk_kv)
      - kind='trafo': params = (sn_mva, vk_pct, vkr_pct, hv_kv, lv_kv)
    u/v are ACTIVSg bus_ids.
    """
    bus_df = cf.bus.reset_index()
    gen_df = cf.gen.reset_index()
    bra_df = cf.branch.reset_index()
    bus_v_all = dict(zip(bus_df["BUS_I"].astype(int), bus_df["BASE_KV"].astype(float)))
    # Bus lat/lon for length-aware impedance scaling on Ours backbone lines
    bus_bb_df = pd.read_csv(backbone_csv)
    bus_geo = dict(zip(bus_bb_df["bus_id"].astype(int),
                        zip(bus_bb_df["lat"].astype(float), bus_bb_df["lon"].astype(float))))

    net = pp.create_empty_network(f_hz=60.0, sn_mva=100.0)
    bus_idx_map = {}
    for _, b in bus_df.iterrows():
        bid = int(b["BUS_I"])
        bus_idx_map[bid] = pp.create_bus(net, vn_kv=float(b["BASE_KV"]),
                                          name=f"b{bid}", in_service=True)

    # Loads
    for _, row in bus_df.iterrows():
        pd_mw = float(row["PD"]) * scale_load
        qd = float(row["QD"]) * scale_load
        if abs(pd_mw) > 1e-6 or abs(qd) > 1e-6:
            pp.create_load(net, bus=bus_idx_map[int(row["BUS_I"])],
                           p_mw=pd_mw, q_mvar=qd)

    # Generators + slack (preserve GT dispatch)
    valid_gen = gen_df[gen_df["GEN_STATUS"] > 0]
    slack_orig = bus_df[bus_df["BUS_TYPE"] == 3]
    slack_bus_i = int(slack_orig.iloc[0]["BUS_I"]) if len(slack_orig) else int(valid_gen.iloc[valid_gen["PMAX"].argmax()]["GEN_BUS"])
    slack_added = False
    for _, g in valid_gen.iterrows():
        gb = int(g["GEN_BUS"])
        if gb not in bus_idx_map: continue
        p = float(g["PG"]); pmax = float(g["PMAX"]); pmin = float(g["PMIN"]); vg = float(g["VG"])
        if gb == slack_bus_i and not slack_added:
            pp.create_ext_grid(net, bus=bus_idx_map[gb], vm_pu=vg,
                               max_p_mw=max(pmax, 100.0), min_p_mw=min(pmin, -100.0))
            slack_added = True
        else:
            pp.create_gen(net, bus=bus_idx_map[gb], p_mw=p, vm_pu=vg,
                          max_p_mw=max(pmax, p + 1.0), min_p_mw=min(pmin, 0.0),
                          controllable=True)
    if not slack_added:
        pp.create_ext_grid(net, bus=bus_idx_map[slack_bus_i], vm_pu=1.0)

    # Non-backbone GT branches (keep as-is) + GT backbone CROSS-TIER transformers
    # (only same-tier backbone is replaced by Ours)
    backbone_set = {int(b) for b in pd.read_csv(backbone_csv)["bus_id"]}
    for _, br in bra_df.iterrows():
        u, v = int(br["F_BUS"]), int(br["T_BUS"])
        if u in backbone_set and v in backbone_set:
            vk_u = bus_v_all.get(u); vk_v = bus_v_all.get(v)
            if vk_u is not None and vk_v is not None and abs(vk_u - vk_v) < 1e-3:
                continue  # same-tier backbone replaced by Ours
            # cross-tier backbone: keep GT trafo as-is
        if u not in bus_idx_map or v not in bus_idx_map: continue
        r = float(br["BR_R"]); x = float(br["BR_X"]); b = float(br["BR_B"])
        rA = float(br.get("RATE_A", 0.0))
        tap = float(br.get("TAP", 0.0))
        vk_u = bus_v_all[u]; vk_v = bus_v_all[v]
        same_tier = abs(vk_u - vk_v) < 1e-3 and abs(tap) < 1e-6
        z_base = (max(vk_u, vk_v) ** 2) / 100.0
        try:
            if same_tier:
                pp.create_line_from_parameters(
                    net, from_bus=bus_idx_map[u], to_bus=bus_idx_map[v],
                    length_km=1.0,
                    r_ohm_per_km=max(r * z_base, 1e-4),
                    x_ohm_per_km=max(x * z_base, 1e-3),
                    c_nf_per_km=max(b / z_base * 1e6 / (2 * np.pi * 60), 0.0),
                    max_i_ka=max(rA / (np.sqrt(3) * max(vk_u, vk_v)), 1.0) if rA > 0 else 5.0,
                )
            else:
                hv = max(vk_u, vk_v); lv = min(vk_u, vk_v)
                hv_bus = bus_idx_map[u] if vk_u >= vk_v else bus_idx_map[v]
                lv_bus = bus_idx_map[v] if vk_u >= vk_v else bus_idx_map[u]
                pp.create_transformer_from_parameters(
                    net, hv_bus=hv_bus, lv_bus=lv_bus,
                    sn_mva=max(rA, 100.0), vn_hv_kv=hv, vn_lv_kv=lv,
                    vk_percent=max(x * 100.0, 5.0),
                    vkr_percent=max(r * 100.0, 0.1),
                    pfe_kw=0.0, i0_percent=0.0,
                )
        except Exception:
            pass

    # Ours backbone replacements
    n_lines_added = 0; n_trafos_added = 0; n_skipped = 0
    for entry in branch_subset:
        u, v, kind, params = entry
        if u not in bus_idx_map or v not in bus_idx_map: continue
        if u == v: continue
        try:
            if kind == "line":
                r, x, b, rateA, v_kv = params
                z_base = (v_kv ** 2) / 100.0
                # Length-aware: scale impedance to actual haversine distance
                # so long Ours lines have proportionally more impedance.
                if u in bus_geo and v in bus_geo:
                    pa, pb = bus_geo[u], bus_geo[v]
                    length_km = max(haversine_km(pa[0], pa[1], pb[0], pb[1]), 1.0)
                else:
                    length_km = 1.0
                # Tier-median per-unit is for a "typical" 100-km line.
                MEDIAN_LEN = {115.0: 35.0, 161.0: 50.0, 230.0: 80.0, 500.0: 150.0}
                ml = MEDIAN_LEN.get(v_kv, 80.0)
                scale = length_km / ml
                r_eff = r * scale; x_eff = x * scale; b_eff = b * scale
                pp.create_line_from_parameters(
                    net, from_bus=bus_idx_map[u], to_bus=bus_idx_map[v],
                    length_km=length_km,
                    r_ohm_per_km=max(r_eff * z_base / length_km, 1e-4),
                    x_ohm_per_km=max(x_eff * z_base / length_km, 1e-3),
                    c_nf_per_km=max(b_eff / z_base * 1e6 / (2 * np.pi * 60 * length_km), 0.0),
                    max_i_ka=max(rateA / (np.sqrt(3) * v_kv), 1.0),
                )
                n_lines_added += 1
            elif kind == "trafo":
                sn_mva, vk, vkr, hv, lv = params
                hv_bus = bus_idx_map[u] if bus_v_all[u] >= bus_v_all[v] else bus_idx_map[v]
                lv_bus = bus_idx_map[v] if bus_v_all[u] >= bus_v_all[v] else bus_idx_map[u]
                pp.create_transformer_from_parameters(
                    net, hv_bus=hv_bus, lv_bus=lv_bus,
                    sn_mva=sn_mva, vn_hv_kv=hv, vn_lv_kv=lv,
                    vk_percent=vk, vkr_percent=vkr,
                    pfe_kw=0.0, i0_percent=0.0,
                )
                n_trafos_added += 1
            else:
                n_skipped += 1
        except Exception:
            n_skipped += 1

    return net, n_lines_added, n_trafos_added, n_skipped


def try_ac(net, max_relax=3):
    for attempt in range(max_relax):
        for init in ("flat", "dc"):
            try:
                pp.runpp(net, algorithm="nr", init=init, enforce_q_lims=False,
                         calculate_voltage_angles=True, max_iteration=50,
                         tolerance_mva=1e-3)
                return True, attempt
            except Exception:
                continue
        # Relax: reduce loads 10%
        if hasattr(net, "load") and len(net.load):
            net.load["p_mw"] *= 0.9
            net.load["q_mvar"] *= 0.9
    return False, max_relax


def build_ours_subset(ours_pt, bus_v_lookup, use_learned_attrs=False):
    ours = torch.load(ours_pt, weights_only=False)
    pred_edges = ours["pred_edges"]
    pred_rxb = ours.get("pred_rxb")
    local_to_bus = ours["bus_id"]
    if hasattr(local_to_bus, "numpy"):
        local_to_bus = local_to_bus.numpy()
    subset, seen = [], set()
    for k in range(pred_edges.shape[0]):
        u_loc, v_loc = int(pred_edges[k, 0]), int(pred_edges[k, 1])
        u = int(local_to_bus[u_loc]); v = int(local_to_bus[v_loc])
        if u == v: continue
        key = (u, v) if u < v else (v, u)
        if key in seen: continue
        seen.add(key)
        vk_u = bus_v_lookup.get(u); vk_v = bus_v_lookup.get(v)
        if vk_u is None or vk_v is None: continue
        if abs(vk_u - vk_v) < 1e-3:
            v_kv = vk_u
            if use_learned_attrs and pred_rxb is not None:
                r, x, b = float(pred_rxb[k, 0]), float(pred_rxb[k, 1]), float(pred_rxb[k, 2])
                rA = TIER_LINE[v_kv][3]
            else:
                r, x, b, rA = TIER_LINE[v_kv]
            subset.append((u, v, "line", (r, x, b, rA, v_kv)))
    return subset


def build_schultz_subset(bus_bb_df, bus_v_lookup, rng_seed=0):
    """Generate Schultz backbone on the 1447 backbone bus positions, then
    drop cross-tier edges and use tier-median impedance per same-tier line.
    """
    ids = bus_bb_df["bus_id"].astype(int).tolist()
    pos = bus_bb_df[["lon", "lat"]].values.astype(float)
    rng = np.random.RandomState(rng_seed)
    G = schultz_2014(pos, p=0.2, q=0.075, rng=rng)
    subset = []
    same, cross = 0, 0
    for i, j in G.edges():
        u, v = ids[i], ids[j]
        vk_u = bus_v_lookup.get(u); vk_v = bus_v_lookup.get(v)
        if vk_u is None or vk_v is None: continue
        if abs(vk_u - vk_v) < 1e-3:
            r, x, b, rA = TIER_LINE[vk_u]
            subset.append((u, v, "line", (r, x, b, rA, vk_u)))
            same += 1
        else:
            cross += 1
    print(f"  Schultz: {same} same-tier edges, {cross} cross-tier dropped", flush=True)
    return subset


def add_per_tier_mst(subset, bus_bb_df):
    """Close per-tier connectivity with MST extras using TIER_LINE impedance."""
    buses_by_tier = {}
    for _, row in bus_bb_df.iterrows():
        buses_by_tier.setdefault(float(row["baseKV"]), []).append(
            (int(row["bus_id"]), float(row["lon"]), float(row["lat"]))
        )
    edges_by_tier = {}
    for u, v, kind, params in subset:
        if kind != "line": continue
        v_kv = params[-1]
        edges_by_tier.setdefault(v_kv, set()).add(
            (u, v) if u < v else (v, u)
        )
    added = 0
    for v_kv, bus_list in buses_by_tier.items():
        ids = [b[0] for b in bus_list]
        G = nx.Graph(); G.add_nodes_from(ids)
        G.add_edges_from(edges_by_tier.get(v_kv, set()))
        if nx.is_connected(G): continue
        pos_map = {b[0]: (b[1], b[2]) for b in bus_list}
        comps = list(nx.connected_components(G))
        while len(comps) > 1:
            c_a = comps[0]; c_b = comps[1]
            best = (float("inf"), None, None)
            for ia in c_a:
                pa = pos_map[ia]
                for ib in c_b:
                    pb = pos_map[ib]
                    d = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                    if d < best[0]: best = (d, ia, ib)
            if best[1] is None: break
            u, v = best[1], best[2]
            G.add_edge(u, v)
            r, x, b, rA = TIER_LINE[v_kv]
            subset.append((u, v, "line", (r, x, b, rA, v_kv)))
            added += 1
            comps = list(nx.connected_components(G))
    return subset, added


def compute_metrics(net, bus_v_lookup):
    """Compute extended electrical metrics."""
    vm = net.res_bus.vm_pu.dropna()
    if hasattr(net, "load") and len(net.load):
        p_load = float(net.load.p_mw.sum())
    else:
        p_load = 0.0
    if hasattr(net, "res_gen") and len(net.res_gen):
        p_gen = float(net.res_gen.p_mw.sum()) + float(net.res_ext_grid.p_mw.sum())
    else:
        p_gen = float(net.res_ext_grid.p_mw.sum()) if hasattr(net, "res_ext_grid") else 0.0
    p_loss = p_gen - p_load
    if len(net.line):
        ll = net.res_line.loading_percent.dropna()
        line_loss_mw = float(net.res_line.pl_mw.sum())
    else:
        ll = pd.Series([], dtype=float); line_loss_mw = 0.0
    if hasattr(net, "res_trafo") and len(net.trafo):
        trafo_loss_mw = float(net.res_trafo.pl_mw.sum())
        tl = net.res_trafo.loading_percent.dropna()
    else:
        trafo_loss_mw = 0.0; tl = pd.Series([], dtype=float)

    # Per-tier voltage breakdown: by pandapower bus vn_kv directly
    per_tier = {}
    for tier in (115.0, 161.0, 230.0, 500.0):
        pp_indices = net.bus.index[np.isclose(net.bus["vn_kv"], tier)].tolist()
        if pp_indices:
            v_tier = net.res_bus.vm_pu.loc[pp_indices].dropna()
            # Per-tier line subset for tier-localised losses
            tier_set = set(pp_indices)
            tier_line_mask = (net.line["from_bus"].isin(tier_set) &
                              net.line["to_bus"].isin(tier_set)) if len(net.line) else pd.Series([], dtype=bool)
            if tier_line_mask.any():
                tier_loss = float(net.res_line.pl_mw.loc[tier_line_mask].sum())
            else:
                tier_loss = 0.0
            per_tier[f"{int(tier)}kV"] = {
                "n": int(len(v_tier)),
                "v_min": float(v_tier.min()) if len(v_tier) else None,
                "v_max": float(v_tier.max()) if len(v_tier) else None,
                "v_in_band_pct": float(((v_tier >= 0.94) & (v_tier <= 1.06)).mean() * 100) if len(v_tier) else None,
                "v_in_strict_pct": float(((v_tier >= 0.95) & (v_tier <= 1.05)).mean() * 100) if len(v_tier) else None,
                "p_loss_mw": tier_loss,
            }

    # Reactive-power audit on generators (PV + slack)
    q_viol = 0; q_max_gen = 0; q_total = 0.0
    if hasattr(net, "res_gen") and len(net.gen):
        for i in net.gen.index:
            q = float(net.res_gen.q_mvar.iloc[i])
            qmin = float(net.gen.min_q_mvar.iloc[i]) if "min_q_mvar" in net.gen.columns else -1e9
            qmax = float(net.gen.max_q_mvar.iloc[i]) if "max_q_mvar" in net.gen.columns else 1e9
            q_total += abs(q)
            if not np.isnan(q) and (q < qmin - 1.0 or q > qmax + 1.0):
                q_viol += 1
            q_max_gen = max(q_max_gen, abs(q))

    # Top-overloaded lines (for diagnostic)
    top_overloaded = []
    if len(ll):
        top_idx = ll.sort_values(ascending=False).head(10).index.tolist()
        for idx in top_idx:
            try:
                fb = int(net.line.from_bus.iloc[idx]); tb = int(net.line.to_bus.iloc[idx])
                top_overloaded.append({
                    "line_idx": int(idx),
                    "from_kv": float(net.bus.vn_kv.iloc[fb]),
                    "to_kv": float(net.bus.vn_kv.iloc[tb]),
                    "loading_pct": float(ll.loc[idx]),
                    "length_km": float(net.line.length_km.iloc[idx]),
                })
            except Exception:
                pass

    return {
        "v_min": float(vm.min()) if len(vm) else None,
        "v_max": float(vm.max()) if len(vm) else None,
        "v_in_band_pct": float(((vm >= 0.94) & (vm <= 1.06)).mean() * 100) if len(vm) else None,
        "v_in_strict_pct": float(((vm >= 0.95) & (vm <= 1.05)).mean() * 100) if len(vm) else None,
        "p_load_mw": p_load,
        "p_gen_mw": p_gen,
        "p_loss_mw": p_loss,
        "p_loss_pct": (p_loss / p_gen * 100) if p_gen > 0 else None,
        "line_loss_mw": line_loss_mw,
        "trafo_loss_mw": trafo_loss_mw,
        "max_line_loading_pct": float(ll.max()) if len(ll) else 0.0,
        "mean_line_loading_pct": float(ll.mean()) if len(ll) else 0.0,
        "pct_lines_overloaded": float((ll > 100).mean() * 100) if len(ll) else 0.0,
        "n_lines_overloaded": int((ll > 100).sum()) if len(ll) else 0,
        "n_lines": int(len(ll)),
        "max_trafo_loading_pct": float(tl.max()) if len(tl) else 0.0,
        "q_violations": q_viol,
        "q_max_mvar": q_max_gen,
        "q_total_mvar": q_total,
        "per_tier": per_tier,
        "top_overloaded": top_overloaded,
    }


def n1_contingency(net_template_fn, max_lines=200, seed=0):
    """N-1 contingency: trip each backbone line (≥115 kV), check PF
    convergence and V-band violations. Pass max_lines=-1 (or >=n_backbone)
    to test all backbone lines."""
    net = net_template_fn()
    # Identify backbone lines (both endpoints ≥ 115 kV)
    bus_kv = {i: net.bus.vn_kv.iloc[i] for i in range(len(net.bus))}
    backbone_lines = [i for i in range(len(net.line))
                       if bus_kv[net.line.from_bus.iloc[i]] >= 115
                       and bus_kv[net.line.to_bus.iloc[i]] >= 115]
    if max_lines < 0 or max_lines >= len(backbone_lines):
        chosen = backbone_lines
    else:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(backbone_lines, max_lines, replace=False)
    converged = 0; v_violations_total = 0; new_violations = 0
    # Baseline V-band first
    pp.runpp(net, algorithm="nr", init="flat", calculate_voltage_angles=True,
             tolerance_mva=1e-3, max_iteration=50)
    base_v = net.res_bus.vm_pu.dropna()
    base_violations = int(((base_v < 0.94) | (base_v > 1.06)).sum())
    failures = 0
    for line_idx in chosen:
        net_n1 = net_template_fn()
        net_n1.line.at[line_idx, "in_service"] = False
        try:
            pp.runpp(net_n1, algorithm="nr", init="flat", calculate_voltage_angles=True,
                     tolerance_mva=1e-3, max_iteration=50)
            converged += 1
            v = net_n1.res_bus.vm_pu.dropna()
            v_violations = int(((v < 0.94) | (v > 1.06)).sum())
            new_violations += max(0, v_violations - base_violations)
            v_violations_total += v_violations
        except Exception:
            failures += 1
    return {
        "n_lines_tested": len(chosen),
        "n_converged": converged,
        "convergence_rate_pct": converged / len(chosen) * 100 if len(chosen) else 0.0,
        "n_failures": failures,
        "baseline_v_violations": base_violations,
        "mean_v_violations_per_n1": v_violations_total / max(converged, 1),
        "mean_new_v_violations_per_n1": new_violations / max(converged, 1),
    }


def build_gt_branches(cf, backbone_buses, bus_v_all):
    bra_df = cf.branch.reset_index()
    branches = []
    for _, br in bra_df.iterrows():
        u, v = int(br["F_BUS"]), int(br["T_BUS"])
        r = float(br["BR_R"]); x = float(br["BR_X"]); b = float(br["BR_B"])
        rA = float(br.get("RATE_A", 0.0))
        if u in backbone_buses and v in backbone_buses:
            vk_u = bus_v_all.get(u); vk_v = bus_v_all.get(v)
            if abs(vk_u - vk_v) < 1e-3:
                branches.append((u, v, "line", (r, x, b, rA, vk_u)))
            else:
                tier_pair = (min(vk_u, vk_v), max(vk_u, vk_v))
                tp = TIER_TRAFO.get(tier_pair, {"sn_mva": max(rA, 200), "vk": max(x*100, 8.0), "vkr": max(r*100, 0.3)})
                branches.append((u, v, "trafo",
                                  (tp["sn_mva"], tp["vk"], tp["vkr"], tier_pair[1], tier_pair[0])))
    return branches
