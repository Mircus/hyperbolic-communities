import numpy as np
import matplotlib.pyplot as plt

def plot_community_trajectories(times, traj):
    T, N, M = traj.shape
    avg = traj.mean(axis=1)
    plt.figure()
    for m in range(M):
        plt.plot(times, avg[:, m], label=f"channel {m}")
    plt.xlabel("time"); plt.ylabel("avg amplitude")
    plt.legend(); plt.tight_layout()
    return plt.gcf()
