from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.signal import hilbert

from .networkx_integration import graph_laplacian
from ..core.hyperbolic_flow import HyperbolicFlow
from ..extraction.frequency_analysis import dominant_frequencies
from ..extraction.phase_clustering import phase_similarity_clustering

class HyperbolicGraphCommunities:
    def __init__(self, G: nx.Graph, damping: float = 0.1, dt: float = 0.1, m: int = 2, seed: int = 0) -> None:
        self.G = G
        self.L = graph_laplacian(G, normalized=True)
        self.alpha = damping
        self.dt = dt
        self.m = m
        self.rng = np.random.default_rng(seed)

    def _init_states(self):
        n = self.G.number_of_nodes()
        c0 = self.rng.normal(size=(n, self.m)) * 0.1
        c1 = c0.copy()
        return c0, c1

    def detect_communities(self, periods: float = 10.0, method: str = "frequency_phase_hybrid") -> dict:
        c0, c1 = self._init_states()
        flow = HyperbolicFlow(L=self.L, M=None, alpha=self.alpha, dt=self.dt)
        t, traj = flow.simulate(c0, c1, T=periods, record=True)
        result = {"times": t, "traj": traj}
        if method in ("frequency", "frequency_phase_hybrid", "frequency_phase"):
            freqs, dom_idx = dominant_frequencies(traj, self.dt, top_k=1)
            result["dominant_freqs"] = freqs
        if method in ("phase", "frequency_phase_hybrid", "frequency_phase"):
            analytic = hilbert(traj[:, :, 0], axis=0)
            phases = np.angle(analytic)
            labels = phase_similarity_clustering(phases, n_clusters=self.m)
            result["labels"] = labels
        return result
