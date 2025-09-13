import numpy as np

def suggest_damping(spectrum, target_q=0.8) -> float:
    lam = np.asarray(spectrum)
    lam = lam[lam > 1e-9]
    if lam.size == 0:
        return 0.1
    mid = np.median(lam)
    alpha = np.sqrt(2*mid) / max(target_q, 1e-3)
    return float(alpha)
