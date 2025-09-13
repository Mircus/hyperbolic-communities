import numpy as np
import networkx as nx

def oscillatory_sbm(n_blocks=2, block_size=16, p_in=0.2, p_out=0.05, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [block_size]*n_blocks
    probs = np.full((n_blocks,n_blocks), p_out, dtype=float)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    labels = []
    for b in range(n_blocks):
        labels += [b]*block_size
    return G, np.array(labels)
