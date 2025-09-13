import matplotlib.pyplot as plt
from hyperbolic_communities.utils.data_io import load_karate
from hyperbolic_communities.graph.graph_communities import HyperbolicGraphCommunities
from hyperbolic_communities.visualization.temporal_plots import plot_community_trajectories

G = load_karate()
hgc = HyperbolicGraphCommunities(G, damping=0.15, dt=0.1, m=2)
result = hgc.detect_communities(periods=10.0, method="frequency_phase_hybrid")
fig = plot_community_trajectories(result["times"], result["traj"])
plt.show()
