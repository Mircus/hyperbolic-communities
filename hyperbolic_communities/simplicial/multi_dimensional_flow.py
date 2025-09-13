from __future__ import annotations
from typing import Dict
from scipy import sparse
from .boundary_operators import boundary_operators

def build_coupling_operator(simplices: Dict[int, list], k: int, j: int) -> sparse.csr_matrix:
    B = boundary_operators(simplices)
    n_k = len(simplices.get(k, []))
    n_j = len(simplices.get(j, []))
    if k == j:
        return sparse.eye(n_k, n_j, format="csr")
    if k == j-1:
        return abs(B[j]).tocsr()
    if k == j+1:
        return abs(B[k]).T.tocsr()
    return sparse.csr_matrix((n_k, n_j))
