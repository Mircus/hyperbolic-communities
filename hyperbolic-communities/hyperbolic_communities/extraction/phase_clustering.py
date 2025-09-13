import numpy as np
from sklearn.cluster import SpectralClustering

def phase_similarity_clustering(phases: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    T, N = phases.shape
    diff = phases[:, :, None] - phases[:, None, :]
    S = np.cos(diff).mean(axis=0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans", random_state=0)
    labels = sc.fit_predict(S)
    return labels
