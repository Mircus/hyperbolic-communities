from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from scipy import sparse

Array = np.ndarray
SpMat = sparse.spmatrix

class HyperbolicFlow:
    """Node-level hyperbolic flow:
        M ddot{c} + alpha M dot{c} + 2 L c = f(t)
    Leapfrog-like update with linear damping (mass-lumped allowed).
    """
    def __init__(self, L: SpMat, M: Optional[SpMat] = None, alpha: float = 0.1,
                 f: Optional[Callable[[float, Array], Array]] = None, dt: float = 0.1, seed: int = 0) -> None:
        self.L = L.tocsr()
        n = L.shape[0]
        self.M = (M if M is not None else sparse.eye(n)).tocsr()
        self.alpha = float(alpha)
        self.f = f
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        # diagonal mass detection
        self._M_diag = None
        if self.M.nnz == n and (self.M - sparse.diags(self.M.diagonal())).nnz == 0:
            self._M_diag = self.M.diagonal()

    def step(self, c_prev: Array, c_cur: Array, t: float) -> Array:
        dt = self.dt
        a = self.alpha
        f_t = self.f(t, c_cur) if self.f is not None else 0.0
        rhs = 2 * c_cur - (1 - a * dt / 2.0) * c_prev + dt ** 2 * (-2 * (self.L @ c_cur) + f_t)
        denom = (1 + a * dt / 2.0)
        if self._M_diag is not None:
            if rhs.ndim == 2:
                return rhs / (denom * self._M_diag)[:, None]
            return rhs / (denom * self._M_diag)
        A = (denom) * self.M
        sol = sparse.linalg.spsolve(A, rhs)
        if rhs.ndim == 2 and sol.ndim == 1:
            sol = sol.reshape(-1, 1)
        return sol

    def simulate(self, c0: Array, c1: Optional[Array] = None, T: float = 10.0, record: bool = True):
        nsteps = int(np.round(T / self.dt))
        c0 = np.asarray(c0)
        if c1 is None:
            c1 = c0.copy()
        traj = []
        t = 0.0
        c_prev, c_cur = c0, c1
        if record:
            traj.append(c_prev.copy())
            traj.append(c_cur.copy())
        for k in range(1, nsteps):
            t = (k + 0) * self.dt
            c_next = self.step(c_prev, c_cur, t)
            if record:
                traj.append(c_next.copy())
            c_prev, c_cur = c_cur, c_next
        times = np.linspace(0, nsteps * self.dt, len(traj))
        return times, np.stack(traj, axis=0)

class MultiDimHyperbolicFlow:
    """Multi-dimensional cochain flow with coupling:
        M_k ddot{c}^{(k)} + alpha_k M_k dot{c}^{(k)} + 2 L_k c^{(k)} = F^{(k)}(c^{(j)}, t)
    Vectorized leapfrog update.
    """
    def __init__(self, Ls: Dict[int, SpMat], Ms: Optional[Dict[int, SpMat]] = None,
                 alphas: Optional[Dict[int, float]] = None, coupling_ops: Optional[Dict[tuple, SpMat]] = None,
                 betas: Optional[Dict[tuple, float]] = None, dt: float = 0.1) -> None:
        self.Ls = {k: Ls[k].tocsr() for k in Ls}
        self.Ms = {k: (Ms[k].tocsr() if (Ms and k in Ms and Ms[k] is not None) else sparse.eye(Ls[k].shape[0])).tocsr() for k in Ls}
        self._M_diag = {k: (self.Ms[k].diagonal() if (self.Ms[k].nnz == self.Ms[k].shape[0] and (self.Ms[k] - sparse.diags(self.Ms[k].diagonal())).nnz == 0) else None) for k in Ls}
        self.alphas = {k: (alphas[k] if alphas and k in alphas else 0.1) for k in Ls}
        self.coupling_ops = coupling_ops or {}
        self.betas = betas or {}
        self.dt = float(dt)

    def _force(self, states):
        F = {k: 0.0 for k in self.Ls}
        for (k, j), I in self.coupling_ops.items():
            beta = self.betas.get((k, j), 0.0)
            if beta == 0.0: continue
            F[k] = F[k] + beta * (I @ states[j])
        return F

    def step(self, prev, cur, t: float):
        dt = self.dt
        F = self._force(cur)
        nxt = {}
        for k in self.Ls.keys():
            a = self.alphas[k]
            L = self.Ls[k]
            M = self.Ms[k]
            Mdiag = self._M_diag[k]
            c_prev = prev[k]
            c_cur = cur[k]
            rhs = 2 * c_cur - (1 - a * dt / 2.0) * c_prev + dt ** 2 * (-2 * (L @ c_cur) + F[k])
            denom = (1 + a * dt / 2.0)
            if Mdiag is not None:
                if rhs.ndim == 2:
                    c_next = rhs / (denom * Mdiag)[:, None]
                else:
                    c_next = rhs / (denom * Mdiag)
            else:
                A = denom * M
                c_next = sparse.linalg.spsolve(A, rhs)
                if rhs.ndim == 2 and c_next.ndim == 1:
                    c_next = c_next.reshape(-1, 1)
            nxt[k] = c_next
        return nxt

    def simulate(self, init_prev, init_cur, T: float = 10.0, record: bool = True):
        nsteps = int(np.round(T / self.dt))
        if record:
            traj = {k: [init_prev[k].copy(), init_cur[k].copy()] for k in self.Ls}
        prev = {k: init_prev[k] for k in self.Ls}
        cur = {k: init_cur[k] for k in self.Ls}
        for s in range(1, nsteps):
            t = (s + 0) * self.dt
            nxt = self.step(prev, cur, t)
            if record:
                for k in self.Ls:
                    traj[k].append(nxt[k].copy())
            prev, cur = cur, nxt
        times = np.linspace(0, nsteps * self.dt, len(next(iter(traj.values()))))
        out = {k: np.stack(traj[k], axis=0) for k in self.Ls}
        return times, out
