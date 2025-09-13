from __future__ import annotations
from typing import Dict, List, Tuple
from scipy import sparse
from .boundary_operators import boundary_operators

def build_hodge_laplacians(simplices: Dict[int, List[Tuple[int,...]]]):
    B = boundary_operators(simplices)
    L = {}
    max_k = max(simplices.keys()) if simplices else 0
    for k in range(0, max_k+1):
        down = B.get(k, None)
        up = B.get(k+1, None)
        Ldown = down.T @ down if down is not None else None
        Lup = up @ up.T if up is not None else None
        if Ldown is None and Lup is None:
            n = len(simplices.get(k, []))
            L[k] = sparse.csr_matrix((n, n))
        elif Ldown is None:
            L[k] = Lup.tocsr()
        elif Lup is None:
            L[k] = Ldown.tocsr()
        else:
            L[k] = (Ldown + Lup).tocsr()
    return L
