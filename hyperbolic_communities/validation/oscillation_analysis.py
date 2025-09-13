import numpy as np
def phase_locking_value(ph):
    z = np.exp(1j*ph)
    return float(np.abs(z.mean(axis=1)).mean())
