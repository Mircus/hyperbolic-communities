from __future__ import annotations
import networkx as nx
import numpy as np
from scipy import sparse

def graph_laplacian(G: nx.Graph, normalized: bool = True):
    n = G.number_of_nodes()
    idx = {v:i for i, v in enumerate(G.nodes())}
    rows, cols, data = [], [], []
    deg = np.zeros(n, dtype=float)
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        w = float(d.get("weight", 1.0))
        rows += [i, j]; cols += [j, i]; data += [-w, -w]
        deg[i] += w; deg[j] += w
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n,n))
    D = sparse.diags(deg)
    L = D - (-A)
    if normalized:
        dinv = 1.0/np.sqrt(np.maximum(deg, 1e-12))
        Dinv = sparse.diags(dinv)
        L = Dinv @ L @ Dinv
    return L
