"""Classical baselines for transmission-grid generation.

All baselines take node positions (N, 2) in [0,1] and return a NetworkX Graph
with N nodes and edges chosen by the baseline rule.

  - random_knn(k):      connect each node to k nearest neighbors (deterministic)
  - geometric_eps(eps): connect pairs within Euclidean distance eps
  - watts_strogatz(k, p): Watts-Strogatz ring rewiring over a radial node ordering
  - schultz_2014:       MST + detour + random long-range edges (Schultz et al. 2014)
  - oracle_gt:          identity (returns GT): sanity ceiling
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def random_knn(pos: np.ndarray, k: int = 3, rng: np.random.RandomState | None = None) -> nx.Graph:
    n = pos.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n < 2:
        return G
    k = min(k, n - 1)
    kdt = cKDTree(pos)
    _, neigh = kdt.query(pos, k=k + 1)
    for i in range(n):
        for j in neigh[i, 1:]:
            u, v = (i, int(j)) if i < int(j) else (int(j), i)
            G.add_edge(u, v)
    return G


def geometric_eps(pos: np.ndarray, eps: float = 0.15) -> nx.Graph:
    n = pos.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n < 2:
        return G
    kdt = cKDTree(pos)
    pairs = kdt.query_pairs(eps)
    G.add_edges_from(pairs)
    return G


def watts_strogatz(pos: np.ndarray, k: int = 4, p: float = 0.1,
                   rng: np.random.RandomState | None = None) -> nx.Graph:
    """Canonical WS over node IDs ordered by position (radial sweep)."""
    n = pos.shape[0]
    if n < 4:
        return random_knn(pos, k=min(k, n - 1) if n > 1 else 0)
    center = pos.mean(0)
    ang = np.arctan2(pos[:, 1] - center[1], pos[:, 0] - center[0])
    order = np.argsort(ang)
    seed = rng.randint(0, 2 ** 31) if rng is not None else 42
    ws = nx.watts_strogatz_graph(n, min(k, n - 1), p, seed=seed)
    relabel = {i: int(order[i]) for i in range(n)}
    return nx.relabel_nodes(ws, relabel)


def schultz_2014(pos: np.ndarray, k0: int = 1, p: float = 0.2, q: float = 0.075,
                 rng: np.random.RandomState | None = None) -> nx.Graph:
    """Schultz / Heitzig / Kurths 2014 random growth model (simplified variant).

    Step 1: spatial MST as backbone.
    Step 2: with probability p add closest non-tree neighbor for each node.
    Step 3: with probability q add a random long-range edge per node.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    n = pos.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n < 2:
        return G
    full = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            full.add_edge(i, j, weight=float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])))
    T = nx.minimum_spanning_tree(full, weight="weight")
    G.add_edges_from(T.edges())
    kdt = cKDTree(pos)
    for i in range(n):
        if rng.random() < p:
            _, neigh = kdt.query(pos[i], k=min(4, n))
            for j in neigh[1:]:
                if not G.has_edge(i, int(j)):
                    G.add_edge(i, int(j))
                    break
    for i in range(n):
        if rng.random() < q:
            j = int(rng.randint(0, n))
            if i != j and not G.has_edge(i, j):
                G.add_edge(i, j)
    return G


def oracle_gt(pos: np.ndarray, edge_index: np.ndarray) -> nx.Graph:
    n = pos.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, k]), int(edge_index[1, k]))
    return G


BASELINES = {
    "random_knn_k3": lambda pos, **_: random_knn(pos, k=3),
    "geometric_eps_0.15": lambda pos, **_: geometric_eps(pos, eps=0.15),
    "watts_strogatz_k4_p0.1": lambda pos, rng=None, **_: watts_strogatz(pos, k=4, p=0.1, rng=rng),
    "schultz_2014": lambda pos, rng=None, **_: schultz_2014(pos, p=0.2, q=0.075, rng=rng),
}


def _self_test():
    np.random.seed(0)
    pos = np.random.rand(30, 2)
    for name, fn in BASELINES.items():
        rng = np.random.RandomState(0)
        G = fn(pos, rng=rng)
        print(f"  {name}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, connected={nx.is_connected(G)}")
    print("[ok] baselines self-test passed")


if __name__ == "__main__":
    _self_test()
