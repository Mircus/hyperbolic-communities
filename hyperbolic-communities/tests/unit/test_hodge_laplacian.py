from hyperbolic_communities.simplicial.simplicial_complex import SimplicialComplex
from hyperbolic_communities.simplicial.hodge_laplacian import build_hodge_laplacians

def test_hodge_shapes():
    simplices = [(0,), (1,), (2,), (0,1), (1,2), (0,2), (0,1,2)]
    K = SimplicialComplex(simplices)
    Ls = build_hodge_laplacians(K.simplices)
    assert 0 in Ls and 1 in Ls and 2 in Ls
