import numpy as np
from scipy import sparse
from hyperbolic_communities.core.hyperbolic_flow import HyperbolicFlow

def test_simple_flow_runs():
    n = 10
    L = sparse.eye(n)
    flow = HyperbolicFlow(L=L, alpha=0.1, dt=0.05)
    c0 = np.zeros((n,1))
    t, traj = flow.simulate(c0, c0.copy(), T=1.0, record=True)
    assert traj.shape[0] > 0
