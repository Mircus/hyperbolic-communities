from __future__ import annotations
from typing import Iterable, Dict, List, Tuple
import numpy as np
from itertools import combinations

class SimplicialComplex:
    def __init__(self, simplices: Iterable[Iterable[int]], max_dim: int | None = None) -> None:
        self.simplices = {}
        all_s = []
        for s in simplices:
            s = tuple(sorted(set(s)))
            if len(s) == 0: 
                continue
            all_s.append(s)
            for k in range(1, len(s)+1):
                for face in combinations(s, k):
                    self.simplices.setdefault(k-1, []).append(tuple(face))
        for k in list(self.simplices.keys()):
            self.simplices[k] = sorted(set(self.simplices[k]))
        self.max_dim = max(self.simplices.keys()) if max_dim is None else max_dim

    @staticmethod
    def from_clique_complex(adj: np.ndarray, max_dim: int = 2) -> "SimplicialComplex":
        n = adj.shape[0]
        simplices = [(i,) for i in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i,j] > 0 or adj[j,i] > 0:
                    simplices.append((i,j))
        if max_dim >= 2:
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i,j] <= 0: continue
                    for k in range(j+1, n):
                        if adj[i,k] > 0 and adj[j,k] > 0:
                            simplices.append((i,j,k))
        return SimplicialComplex(simplices, max_dim=max_dim)

    def faces(self, k: int) -> List[Tuple[int,...]]:
        return self.simplices.get(k, [])

    def n_faces(self, k: int) -> int:
        return len(self.faces(k))
