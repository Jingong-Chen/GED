"""GED reverse network: categorical edge denoiser with impedance and rating heads.

The network scores candidate bus pairs (u, v). Inputs are a geographic raster
G (C x H x W), bus positions P in [-1, 1]^2, the noisy class x_t of every
candidate pair, and the noise level t. Outputs are 4-class logits
{NONE, LV (115/161 kV), HV (230 kV), EHV (500 kV)} and a z-normalised
(log10 r, log10 x, log10 b) triplet per pair. ``ThermalModel`` adds a scalar
MVA-rating head on the shared pair trunk (stage-3 fine-tuning).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmb(nn.Module):
    """Sinusoidal embedding of the noise level t in [0, 1]."""

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=t.dtype)
                          * (10.0 / max(half - 1, 1)))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EdgeDiffusionModel(nn.Module):
    """Categorical edge diffusion denoiser + per-pair impedance regression."""

    NUM_CLASSES = 4
    ATTR_DIM = 3  # log r, log x, log b (z-normalised)

    def __init__(self, geo_c=28, pos_dim=2, hidden=192, n_layers=4, n_heads=6):
        super().__init__()
        self.t_emb = TimestepEmb(64)
        self.node_proj = nn.Linear(pos_dim + geo_c, hidden)
        self.edge_class_emb = nn.Embedding(self.NUM_CLASSES, 32)
        in_pair = 2 * hidden + 32 + 64 + 3 + geo_c
        layers = [nn.Linear(in_pair, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.shared = nn.Sequential(*layers)
        self.class_head = nn.Linear(hidden, self.NUM_CLASSES)
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        self.attr_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.ATTR_DIM),
        )

        self.node_attn_layers = nn.ModuleList()
        self.node_norms1 = nn.ModuleList()
        self.node_ffns = nn.ModuleList()
        self.node_norms2 = nn.ModuleList()
        for _ in range(n_layers):
            self.node_attn_layers.append(nn.MultiheadAttention(
                hidden, n_heads, batch_first=True, dropout=0.0,
            ))
            self.node_norms1.append(nn.LayerNorm(hidden))
            self.node_ffns.append(nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(),
                nn.Linear(hidden * 2, hidden),
            ))
            self.node_norms2.append(nn.LayerNorm(hidden))

    def encode_nodes(self, geo, pos):
        """Bilinearly sample the raster at each bus and run the node transformer."""
        grid = torch.stack([pos[..., 1], pos[..., 0]], dim=-1).unsqueeze(2)
        geo_at = F.grid_sample(geo, grid, mode="bilinear",
                               padding_mode="border", align_corners=False)
        geo_at = geo_at.squeeze(-1).transpose(1, 2)
        h = self.node_proj(torch.cat([pos, geo_at], dim=-1))
        for attn, n1, ffn, n2 in zip(self.node_attn_layers, self.node_norms1,
                                     self.node_ffns, self.node_norms2):
            a, _ = attn(h, h, h, need_weights=False)
            h = n1(h + a)
            h = n2(h + ffn(h))
        return h

    def pair_features(self, geo, pos, i_idx, j_idx, x_t, t):
        """Shared pair trunk activations h_uv (B, P, hidden)."""
        B = pos.shape[0]
        node_h = self.encode_nodes(geo, pos)
        b_idx = torch.arange(B, device=pos.device).unsqueeze(1).expand_as(i_idx)
        hi = node_h[b_idx, i_idx]
        hj = node_h[b_idx, j_idx]
        pi = pos[b_idx, i_idx]
        pj = pos[b_idx, j_idx]
        dx = pj - pi
        length = dx.norm(dim=-1, keepdim=True)
        mid = 0.5 * (pi + pj)
        grid = torch.stack([mid[..., 1], mid[..., 0]], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(geo, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        class_e = self.edge_class_emb(x_t)
        t_e = self.t_emb(t).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        feat = torch.cat([hi, hj, class_e, t_e, length, dx, sampled], dim=-1)
        return self.shared(feat)

    def forward(self, geo, pos, i_idx, j_idx, x_t, t):
        """
        Args:
          geo:   (B, C, H, W) raster
          pos:   (B, N, 2) bus positions in [-1, 1] as (y, x) = (2*lat_n - 1, 2*lon_n - 1)
          i_idx, j_idx: (B, P) candidate endpoints
          x_t:   (B, P) noisy class in {0, 1, 2, 3}
          t:     (B,) noise level in [0, 1]
        Returns:
          class_logits (B, P, 4), attr_pred (B, P, 3)
        """
        h = self.pair_features(geo, pos, i_idx, j_idx, x_t, t)
        return self.class_head(h), self.attr_head(h)


class ThermalModel(nn.Module):
    """EdgeDiffusionModel plus a 1-D MVA-rating head on the shared pair trunk."""

    def __init__(self, base: EdgeDiffusionModel):
        super().__init__()
        self.base = base
        hidden = base.attr_head[-1].in_features
        self.rate_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-initialised output so the rating starts at the tier median (z = 0).
        nn.init.zeros_(self.rate_head[-1].weight)
        nn.init.zeros_(self.rate_head[-1].bias)

    def forward(self, geo, pos, ii, jj, x_t, t):
        h = self.base.pair_features(geo, pos, ii, jj, x_t, t)
        return self.base.class_head(h), self.base.attr_head(h), self.rate_head(h).squeeze(-1)


def fiedler_regularizer(edge_prob, i_idx, j_idx, n_real, eps_target=0.10):
    """max(0, eps - lambda_2(normalised soft Laplacian))^2, averaged over the batch."""
    B, P = edge_prob.shape
    device = edge_prob.device
    losses = []
    for b in range(B):
        n = int(n_real[b].item())
        if n < 4:
            continue
        i = i_idx[b].clamp(max=n - 1)
        j = j_idx[b].clamp(max=n - 1)
        p = edge_prob[b]
        A = torch.zeros(n, n, device=device, dtype=p.dtype)
        A[i, j] = p
        A[j, i] = p
        A = A.clamp(min=0.0, max=1.0)
        d = A.sum(dim=1)
        D_inv_sqrt = torch.where(d > 1e-6, d.pow(-0.5), torch.zeros_like(d))
        L_norm = torch.eye(n, device=device, dtype=p.dtype) - \
            (D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0))
        L_norm = 0.5 * (L_norm + L_norm.transpose(0, 1))
        try:
            eigvals = torch.linalg.eigvalsh(L_norm)
        except RuntimeError:
            continue
        lam2 = eigvals[1] if eigvals.numel() >= 2 else eigvals[0]
        losses.append(torch.relu(eps_target - lam2).pow(2))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


DEFAULT_ARCH = {"geo_c": 28, "hidden": 192, "n_layers": 4, "n_heads": 6}


def load_checkpoint(path, device="cpu", with_rate_head=False):
    """Load a GED checkpoint.

    Accepts the released format ({"arch", "model_state_dict", optional
    "rate_head_state_dict"}) as well as raw training checkpoints written by
    scripts 02-04 (where stage-3 keys are prefixed with ``base.``).

    Returns (model, ckpt_dict). ``model`` is an EdgeDiffusionModel, or a
    ThermalModel when ``with_rate_head`` is True and a rating head is present.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    arch = dict(DEFAULT_ARCH)
    arch.update(ckpt.get("arch", {}))
    args = ckpt.get("args") or ckpt.get("train_args") or {}
    for k_arch, k_args in (("geo_c", "n_geo_channels"), ("hidden", "hidden"),
                           ("n_layers", "n_layers"), ("n_heads", "n_heads")):
        if "arch" not in ckpt and k_args in args:
            arch[k_arch] = args[k_args]

    sd = ckpt["model_state_dict"]
    rate_sd = ckpt.get("rate_head_state_dict")
    if any(k.startswith("base.") for k in sd):
        rate_sd = {k[len("rate_head."):]: v for k, v in sd.items() if k.startswith("rate_head.")}
        sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}

    base = EdgeDiffusionModel(geo_c=arch["geo_c"], hidden=arch["hidden"],
                              n_layers=arch["n_layers"], n_heads=arch["n_heads"])
    base.load_state_dict(sd, strict=True)
    model = base
    if with_rate_head and rate_sd:
        model = ThermalModel(base)
        model.rate_head.load_state_dict(rate_sd, strict=True)
    model.to(device).eval()
    return model, ckpt
