import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree


def dmap_nn_dist(X, r=1, k=5):
    """
    Performs the diffusion Map algorithm for non-linear manifold learning
    Input arguments:
        High dimensional data X
        Max. Neighborhood distance r
        Number of output eigenvectors k
    Returns k eigenvalues and eigenvectors of embedding space
    """
    #Kernel formulation
    kd_tree = KDTree(X)
    sdm = kd_tree.sparse_distance_matrix(kd_tree, r)
    W = sdm.tocsr()
    max = W.max(axis=None)
    epsilon = 0.05*max
    W = -W.power(2)/epsilon
    W.data = (W.expm1()).data + 1

    #Normalization
    I = sp.identity(len(X))
    P = sp.diags(W.sum(axis=1).A1,0)
    Pinv = sp.linalg.spsolve(P,I)
    K = (Pinv.dot(W)).dot(Pinv)
    Q = sp.diags(K.sum(axis=1).A1,0)
    Qinv = sp.linalg.spsolve(Q,I)
    Qnor = Qinv.sqrt()
    T = (Qnor.dot(K)).dot(Qnor)
    
    #Eigenvalues and vector computation
    vals,vec = sp.linalg.eigs(T,k,which='LM')
    vals = np.power(vals, 1/(2*epsilon))
    vec = Qnor.dot(vec)
    
    return vals, vec