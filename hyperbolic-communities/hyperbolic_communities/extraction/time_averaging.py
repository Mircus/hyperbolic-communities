def time_average_membership(traj, window=10):
    T = traj.shape[0]
    return traj[max(0,T-window):].mean(axis=0)
