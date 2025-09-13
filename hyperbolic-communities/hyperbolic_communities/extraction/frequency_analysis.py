import numpy as np

def dominant_frequencies(traj: np.ndarray, dt: float, top_k: int = 1):
    T, N, M = traj.shape
    x = traj.mean(axis=2)
    x = x - x.mean(axis=0, keepdims=True)
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(T, d=dt)
    power = np.abs(X)
    if power.shape[0] > 0:
        power[0, :] = 0.0
    dom_idx = power.argmax(axis=0)
    dom_freqs = freqs[dom_idx]
    return dom_freqs, dom_idx
