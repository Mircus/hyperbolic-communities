import numpy as np
from scipy import sparse

def max_generalized_eig(L: sparse.spmatrix, M: sparse.spmatrix) -> float:
    n = L.shape[0]
    x = np.random.default_rng(0).normal(size=(n,))
    x /= np.linalg.norm(x)
    Minv = sparse.linalg.inv(M.tocsc())
    A = Minv @ L
    lam = 0.0
    for _ in range(50):
        y = A @ x
        lam = float(np.dot(x, y))
        yn = np.linalg.norm(y)
        if yn < 1e-12:
            break
        x = y / yn
    return max(lam, 0.0)

def cfl_dt(L: sparse.spmatrix, M: sparse.spmatrix) -> float:
    mu_max = max_generalized_eig(L, M)
    if mu_max <= 0:
        return 1.0
    return (2.0**0.5) / (mu_max**0.5)
