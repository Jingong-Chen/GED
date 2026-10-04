"""Birchfield et al. grid-validation metrics for synthetic transmission grids.

Metrics:
  1. Degree distribution distance (W1 between empirical degree CDFs)
  2. Average clustering coefficient
  3. Mean shortest path length (over connected components)
  4. Diameter (over largest connected component)
  5. Spectral gap (second-smallest eigenvalue of normalized Laplacian)

Plus convenience helpers:
  - edge_length_w1: W1 distance between edge-length distributions
  - all_metrics(G_gt, G_pred): produce comparison dict
"""
from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance


def degree_w1(G_gt: nx.Graph, G_pred: nx.Graph) -> float:
    d_gt = [d for _, d in G_gt.degree()]
    d_pr = [d for _, d in G_pred.degree()]
    if not d_gt or not d_pr:
        return float("inf")
    return float(wasserstein_distance(d_gt, d_pr))


def avg_clustering(G: nx.Graph) -> float:
    if G.number_of_nodes() < 3:
        return 0.0
    return float(nx.average_clustering(G))


def mean_shortest_path(G: nx.Graph) -> float:
    """Mean of average_shortest_path_length over connected components weighted by size."""
    if G.number_of_nodes() == 0:
        return float("nan")
    if not nx.is_connected(G):
        comps = [G.subgraph(c) for c in nx.connected_components(G) if len(c) > 1]
        if not comps:
            return float("nan")
        total = sum(c.number_of_nodes() for c in comps)
        return float(sum(nx.average_shortest_path_length(c) * c.number_of_nodes() for c in comps) / total)
    return float(nx.average_shortest_path_length(G))


def diameter(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        return int(nx.diameter(G.subgraph(gcc)))
    return int(nx.diameter(G))


def spectral_gap(G: nx.Graph) -> float:
    """Algebraic connectivity = second-smallest normalized Laplacian eigenvalue."""
    if G.number_of_nodes() < 2:
        return 0.0
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        H = G.subgraph(gcc)
    else:
        H = G
    if H.number_of_nodes() < 2:
        return 0.0
    L = nx.normalized_laplacian_matrix(H).astype(float).toarray()
    eigs = np.linalg.eigvalsh(L)
    eigs.sort()
    return float(eigs[1]) if len(eigs) >= 2 else 0.0


def edge_length_w1(G_gt: nx.Graph, G_pred: nx.Graph, pos: dict) -> float:
    """W1 between edge-length distributions. `pos` maps node_id -> (x, y) in same coords."""
    def lengths(G):
        out = []
        for u, v in G.edges():
            if u in pos and v in pos:
                a, b = pos[u], pos[v]
                out.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
        return out
    a = lengths(G_gt); b = lengths(G_pred)
    if not a or not b:
        return float("inf")
    return float(wasserstein_distance(a, b))


def all_metrics(G_gt: nx.Graph, G_pred: nx.Graph, pos: dict | None = None) -> Dict[str, float]:
    out = {
        "degree_w1": degree_w1(G_gt, G_pred),
        "avg_clustering_gt": avg_clustering(G_gt),
        "avg_clustering_pred": avg_clustering(G_pred),
        "avg_clustering_diff": abs(avg_clustering(G_gt) - avg_clustering(G_pred)),
        "mean_path_gt": mean_shortest_path(G_gt),
        "mean_path_pred": mean_shortest_path(G_pred),
        "mean_path_diff": abs(mean_shortest_path(G_gt) - mean_shortest_path(G_pred)),
        "diameter_gt": diameter(G_gt),
        "diameter_pred": diameter(G_pred),
        "diameter_diff": abs(diameter(G_gt) - diameter(G_pred)),
        "spectral_gap_gt": spectral_gap(G_gt),
        "spectral_gap_pred": spectral_gap(G_pred),
        "spectral_gap_diff": abs(spectral_gap(G_gt) - spectral_gap(G_pred)),
        "n_nodes_gt": G_gt.number_of_nodes(),
        "n_edges_gt": G_gt.number_of_edges(),
        "n_nodes_pred": G_pred.number_of_nodes(),
        "n_edges_pred": G_pred.number_of_edges(),
    }
    if pos is not None:
        out["edge_length_w1"] = edge_length_w1(G_gt, G_pred, pos)
    return out


def _self_test():
    g1 = nx.barabasi_albert_graph(50, 3, seed=0)
    g2 = nx.barabasi_albert_graph(50, 3, seed=1)
    g3 = nx.erdos_renyi_graph(50, 0.1, seed=0)
    m12 = all_metrics(g1, g2)
    m13 = all_metrics(g1, g3)
    print("BA <-> BA (similar):")
    print(f"  deg_w1={m12['degree_w1']:.3f}, clust_diff={m12['avg_clustering_diff']:.4f}, path_diff={m12['mean_path_diff']:.3f}, diam_diff={m12['diameter_diff']}")
    print("BA <-> ER (different):")
    print(f"  deg_w1={m13['degree_w1']:.3f}, clust_diff={m13['avg_clustering_diff']:.4f}, path_diff={m13['mean_path_diff']:.3f}, diam_diff={m13['diameter_diff']}")
    assert m13["degree_w1"] > m12["degree_w1"], "ER vs BA should differ more than BA vs BA"
    print("[ok] self-test passed")


if __name__ == "__main__":
    _self_test()
