import numpy as np
def community_stability(L):
    T,N=L.shape
    if T<2: return 1.0
    return float(1.0-(L[1:]!=L[:-1]).mean())
