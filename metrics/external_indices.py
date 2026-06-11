import numpy as np
from scipy.spatial.distance import cdist


def _crisp2fuzzy(P, n_clusters):
    """
    Convert a crisp partition into a fuzzy membership matrix.

    Parameters
    ----------
    P : ndarray of shape (n_samples,) into n_clusters
        Cluster labels.

    n_clusters : int
        Number of clusters.

    Returns
    -------
    ndarray of shape (n_samples, n_clusters)
        Matrix with binary memberships.
    """
    N = len(P)
    P_fuzzy = np.zeros((N, n_clusters))
    for i in range(N):
        cluster = P[i]
        P_fuzzy[i, cluster] = 1
    return P_fuzzy


def hullermeier_index(U, P):
    """
    Compute the Hüllermeier index between a fuzzy partition
    and a reference crisp partition.

    Parameters
    ----------
    U : ndarray of shape (n_samples, n_clusters)
        Fuzzy membership matrix.

    P : ndarray of shape (n_samples,)
        Reference cluster labels.

    Returns
    -------
    float
        Hüllermeier index. Values closer to 1 indicate
        greater agreement between partitions.

    References
    ----------
    Hullermeier, Eyke, et al.
    Comparing fuzzy partitions: A generalization of the rand index and related measures.
    IEEE Transactions on Fuzzy Systems 20.3 (2011): 546-556.
    """

    n_samples, n_clusters = U.shape

    Pf = _crisp2fuzzy(P=P, n_clusters=n_clusters)

    # matrices of distances
    D1 = cdist(U, U, "cityblock") / 2
    D2 = cdist(Pf, Pf, "cityblock") / 2

    # selecting the values of the lower triangle matrix (can be the upper triangle because the matix is symetric)
    idx_lower = np.tril_indices(n_samples, k=-1)
    D1 = D1[idx_lower]
    D2 = D2[idx_lower]

    num = np.abs(D1 - D2).sum()
    den = (n_samples / 2) * (n_samples - 1)
    return 1 - (num / den)
