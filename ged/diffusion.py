"""Forward corruption and reverse sampling for 4-class categorical edge diffusion."""
from __future__ import annotations

import numpy as np
import torch


def categorical_q_sample(x_0, t, num_classes=4):
    """Forward kernel q(x_t | x_0) = (1 - t) * delta(x_0) + t * Uniform(K).

    With probability t a pair is replaced by a uniformly random class.
    x_0: (B, P) long, t: (B,) float in [0, 1].
    """
    mix = torch.rand_like(x_0.float()) < t.unsqueeze(-1)
    rand_class = torch.randint(0, num_classes, x_0.shape, device=x_0.device)
    return torch.where(mix, rand_class, x_0)


@torch.no_grad()
def reverse_diffusion(model, geo, pos_b, ii, jj, n_samples=5, n_steps=30, num_classes=4):
    """Run the reverse process and average the final-step outputs over samples.

    For each sample s (seeded with torch.manual_seed(s)):
      x_T ~ Uniform{0..K-1} on every candidate pair,
      for k = 0..T-1 with t_k = 1 - k/T:
        logits = model(x_t, t_k); x~ ~ Cat(softmax(logits));
        x_t <- x~ with probability max(1 - t_k, 0.05), else keep x_t.
    The softmax of the final step is kept.

    Returns numpy arrays (P,), (P, K), (P, 3): mean edge score s = 1 - pi_NONE,
    mean class probabilities, and mean z-normalised impedance prediction.
    """
    device = geo.device
    P = ii.shape[1]
    edge_probs = np.zeros(P, dtype=np.float32)
    class_probs = np.zeros((P, num_classes), dtype=np.float32)
    attr_avg = np.zeros((P, 3), dtype=np.float32)
    for s in range(n_samples):
        torch.manual_seed(s)
        x_t = torch.randint(0, num_classes, (1, P), device=device)
        schedule = np.linspace(1.0, 0.0, n_steps + 1)
        last_probs = None
        last_attrs = None
        for k in range(n_steps):
            t_val = float(schedule[k])
            t = torch.full((1,), t_val, device=device)
            cls_logits, attr_pred = model(geo, pos_b, ii, jj, x_t, t)[:2]
            probs = cls_logits.softmax(dim=-1)
            last_probs = probs
            last_attrs = attr_pred
            keep = max(1 - t_val, 0.05)
            sampled = torch.multinomial(probs.reshape(-1, num_classes), 1).reshape(1, -1)
            x_t = torch.where(torch.rand_like(x_t.float()) < keep, sampled, x_t)
        edge_probs += (1.0 - last_probs[0, :, 0]).cpu().numpy()
        class_probs += last_probs[0].cpu().numpy()
        attr_avg += last_attrs[0].cpu().numpy()
    edge_probs /= n_samples
    class_probs /= n_samples
    attr_avg /= n_samples
    return edge_probs, class_probs, attr_avg
