import numpy as np
from scipy import sparse

def time_average(traj: np.ndarray, window: int = 10) -> np.ndarray:
    if traj.ndim == 3:
        traj = traj[..., 0]
    n = traj.shape[0]
    w0 = max(0, n - window)
    return traj[w0:].mean(axis=0)

def modal_energy(traj: np.ndarray, L: sparse.spmatrix, dt: float) -> float:
    v = np.diff(traj, axis=0) / dt
    e_kin = 0.5 * (v**2).sum()
    e_pot = 0.5 * np.einsum("ti,ij,tj->", traj, L.toarray(), traj)
    return float(e_kin + e_pot)
