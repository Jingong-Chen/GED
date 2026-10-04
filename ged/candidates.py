"""Candidate-pair set C = kNN(k) union Euclidean MST, and the MST used by the projector."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def knn_candidates(pos, k=12):
    """Undirected pairs (i < j) from each node's k nearest neighbours. Returns (P, 2) int64."""
    n = pos.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(k, n - 1)
    kdt = cKDTree(pos)
    _, neigh = kdt.query(pos, k=k + 1)
    pairs = set()
    for i in range(n):
        for j in neigh[i, 1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            pairs.add((a, b))
    return np.array(sorted(pairs), dtype=np.int64)


def spatial_mst(pos):
    """Exact Euclidean MST over the complete graph (used for patches, N <= a few hundred)."""
    n = pos.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])))
    T = nx.minimum_spanning_tree(G, weight="weight")
    return {(min(u, v), max(u, v)) for u, v in T.edges()}


def spatial_mst_knn(pos, k_graph=40):
    """Euclidean MST computed on a kNN(40) graph, bridging components if needed.

    Used for the full 1,447-bus state, where the complete graph is large. This
    is the exact routine used to produce the paper's full-Texas outputs.
    """
    kdt = cKDTree(pos)
    k = min(k_graph, pos.shape[0] - 1)
    _, neigh = kdt.query(pos, k=k + 1)
    G = nx.Graph()
    G.add_nodes_from(range(pos.shape[0]))
    for i in range(pos.shape[0]):
        for j in neigh[i, 1:]:
            j = int(j)
            d = float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]))
            G.add_edge(i, j, weight=d)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for c_a in range(len(comps)):
            for c_b in range(c_a + 1, len(comps)):
                best = (float("inf"), None, None)
                for i in comps[c_a]:
                    for j in comps[c_b]:
                        d = float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]))
                        if d < best[0]:
                            best = (d, i, j)
                if best[1] is not None:
                    G.add_edge(best[1], best[2], weight=best[0])
    T = nx.minimum_spanning_tree(G, weight="weight")
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in T.edges()}
