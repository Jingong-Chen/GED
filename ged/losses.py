"""Physics-informed training losses on the soft DC power-flow Laplacian.

dc_pf_loss:   mean (b_uv (theta_u - theta_v))^2 with B~ theta = P_bal, b_uv = s_uv / x_uv
thermal_loss: mean max(0, |f_uv| / R_uv - target)^2 with f_uv = 100 b_uv |theta_u - theta_v| MVA
"""
from __future__ import annotations

import torch

ATTR_LOG_MEAN = torch.tensor([-2.52, -1.59, -2.89])
ATTR_LOG_STD = torch.tensor([0.60, 0.40, 2.24])

# log10(MVA rating) statistics used to z-normalise the rating head
RATE_LOG_MEAN = 2.468
RATE_LOG_STD = 0.304


def dc_pf_loss(edge_prob, attr_pred, ii, jj, n_real, p_inj, max_n=120,
               attr_mean=ATTR_LOG_MEAN, attr_std=ATTR_LOG_STD):
    """DC PF angle-difference penalty (see module docstring).

    edge_prob: (B, P): predicted P(edge present)
    attr_pred: (B, P, 3): z-normalised log10(r,x,b)
    ii, jj: (B, P): pair indices
    n_real: (B,): N per patch
    p_inj: (B, N_max): Pg - Pd in MW

    Returns scalar mean penalty. Skips batches with N > max_n or N < 3.
    """
    device = edge_prob.device
    Bsz, P = edge_prob.shape
    am = attr_mean.to(device); asd = attr_std.to(device)
    # Recover x_pu
    log10_x = attr_pred[..., 1] * asd[1] + am[1]
    x_pu = (10.0 ** log10_x).clamp(min=1e-3, max=10.0)
    susc = edge_prob / x_pu  # b_uv weighted by edge probability

    losses = []
    for b in range(Bsz):
        N = int(n_real[b].item())
        if N < 3 or N > max_n:
            continue
        i_b = ii[b][:P]; j_b = jj[b][:P]
        s_b = susc[b][:P]
        # Keep only edges whose endpoints are real
        valid = (i_b < N) & (j_b < N) & (i_b != j_b)
        if not valid.any():
            continue
        i_v = i_b[valid]; j_v = j_b[valid]; s_v = s_b[valid]
        # Build NxN Laplacian B
        B_mat = torch.zeros(N, N, device=device, dtype=s_v.dtype)
        B_mat.index_put_((i_v, j_v), -s_v, accumulate=True)
        B_mat.index_put_((j_v, i_v), -s_v, accumulate=True)
        B_mat.index_put_((i_v, i_v), s_v, accumulate=True)
        B_mat.index_put_((j_v, j_v), s_v, accumulate=True)
        # Tikhonov regularization for invertibility (handles disconnected
        # components and zero rows at small edge_prob)
        B_reg = B_mat + 1e-2 * torch.eye(N, device=device, dtype=s_v.dtype)
        # Balance: project injection to be slack-free
        p = p_inj[b, :N].to(s_v.dtype) / 100.0  # MW → per-unit (S_base=100)
        p_bal = p - p.mean()
        try:
            theta = torch.linalg.solve(B_reg, p_bal)
        except Exception:
            continue
        # Per-edge angle differences weighted by susceptance (flow proxy)
        theta_diff = theta[i_v] - theta[j_v]
        flow_proxy = s_v * theta_diff
        # L2 penalty on the flow proxy: softly discourages topologies that
        # require large angle swings (≈ high line loading)
        losses.append((flow_proxy ** 2).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def thermal_loss(edge_prob, attr_pred, rate_pred, ii, jj, n_real, p_inj,
                 max_n=120, target=0.8,
                 attr_mean=ATTR_LOG_MEAN, attr_std=ATTR_LOG_STD,
                 rate_mean=RATE_LOG_MEAN, rate_std=RATE_LOG_STD):
    """Penalty on |flow|/rate > target.  Uses DC-PF angle solve to get
    flows, predicted x for susceptance, and predicted rateA for the
    thermal rating.  `target` < 1 keeps a safety margin.
    """
    device = edge_prob.device
    Bsz, P = edge_prob.shape
    am = attr_mean.to(device); asd = attr_std.to(device)
    # Recover actual x_pu and rateA from z-normalised predictions
    log10_x = attr_pred[..., 1] * asd[1] + am[1]
    x_pu = (10.0 ** log10_x).clamp(min=1e-3, max=10.0)
    # Clamp z-normalised rate to ±3 std to prevent overflow when training
    # diverges (gives rate range ~17-4180 MVA, covers realistic transmission).
    log10_rate = rate_pred.clamp(min=-3.0, max=3.0) * rate_std + rate_mean
    rate_mva = (10.0 ** log10_rate).clamp(min=10.0, max=5000.0)
    susc = edge_prob / x_pu

    losses = []
    for b in range(Bsz):
        N = int(n_real[b].item())
        if N < 3 or N > max_n:
            continue
        i_b = ii[b][:P]; j_b = jj[b][:P]
        s_b = susc[b][:P]; r_b = rate_mva[b][:P]
        valid = (i_b < N) & (j_b < N) & (i_b != j_b)
        if not valid.any():
            continue
        i_v = i_b[valid]; j_v = j_b[valid]
        s_v = s_b[valid]; r_v = r_b[valid]
        # Soft Laplacian
        B_mat = torch.zeros(N, N, device=device, dtype=s_v.dtype)
        B_mat.index_put_((i_v, j_v), -s_v, accumulate=True)
        B_mat.index_put_((j_v, i_v), -s_v, accumulate=True)
        B_mat.index_put_((i_v, i_v), s_v, accumulate=True)
        B_mat.index_put_((j_v, j_v), s_v, accumulate=True)
        B_reg = B_mat + 1e-2 * torch.eye(N, device=device, dtype=s_v.dtype)
        p_bus = p_inj[b, :N].to(s_v.dtype) / 100.0  # MW → per-unit
        p_bal = p_bus - p_bus.mean()
        try:
            theta = torch.linalg.solve(B_reg, p_bal)
        except Exception:
            continue
        theta_diff = theta[i_v] - theta[j_v]
        # Flow in per-unit ≈ b * Δθ, convert to MVA: * S_base (100 MVA)
        flow_pu = s_v * theta_diff
        flow_mva = (flow_pu.abs() * 100.0)
        # Loading ratio: flow / rate, want ≤ target
        loading = flow_mva / r_v
        # Squared hinge penalty
        overload = torch.relu(loading - target)
        losses.append((overload ** 2).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()
