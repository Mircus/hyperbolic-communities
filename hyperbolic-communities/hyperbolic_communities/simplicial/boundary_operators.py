from __future__ import annotations
from typing import Dict, List, Tuple
from scipy import sparse

def boundary_matrix(faces_km1: List[Tuple[int,...]], faces_k: List[Tuple[int,...]]):
    idx_km1 = {face:i for i, face in enumerate(faces_km1)}
    rows, cols, data = [], [], []
    for j, s in enumerate(faces_k):
        s = list(s)
        for r in range(len(s)):
            face = tuple(s[:r] + s[r+1:])
            i = idx_km1.get(face, None)
            if i is None: continue
            sign = (-1)**r
            rows.append(i); cols.append(j); data.append(sign)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(faces_km1), len(faces_k)))

def boundary_operators(simplices: Dict[int, List[Tuple[int,...]]]):
    B = {}
    if not simplices:
        return B
    maxk = max(simplices.keys())
    for k in range(1, maxk+1):
        B[k] = boundary_matrix(simplices.get(k-1, []), simplices.get(k, []))
    return B
