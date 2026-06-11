import numpy as np
import pandas as pd
from utils import get_boundaries, check_interval_table
from scipy.spatial.distance import pdist


def caputo_estimator_co_clustering(X, batch_size=20000, random_state=26):
    """
    Estimate the Gaussian Kernel width parameter for interval-valued data using Caputo's heuristic.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Interval-valued dataset.

    batch_size : int, default=20000
        Maximum number of interval bounds processed in each batch for large datasets.

    random_state : int, default=26
        Seed for the random number generator.

    Returns
    -------
    float
        Estimated width parameter.

    References
    ----------
    .. [1] Caputo, B., Sim, K., Furesjö, F., and Smola, A. (2002).
           "Appearance-Based Object Recognition Using SVMs: Which Kernel Should I Use?"
           NIPS Workshop on Statistical Methods for Computational
           Experiments in Visual Processing and Computer Vision.
    """

    check_interval_table(X)
    Xl, Xu = get_boundaries(interval_table=X)
    if isinstance(Xl, pd.DataFrame) and isinstance(Xu, pd.DataFrame):
        Xl = Xl.to_numpy()
        Xu = Xu.to_numpy()

    n = np.size(Xl)
    Xl_flat = Xl.reshape(-1, 1)
    Xu_flat = Xu.reshape(-1, 1)
    if n <= batch_size:
        dist = pdist(Xl_flat, "sqeuclidean") + pdist(Xu_flat, "sqeuclidean")
        dist = dist[dist > 0.0]
        return np.quantile(dist, [0.1, 0.9]).sum() / 4
    else:
        np.random.seed(random_state)
        np.random.shuffle(Xl_flat)
        np.random.shuffle(Xu_flat)
        num_batches = (n + batch_size - 1) // batch_size
        rs_batches = np.zeros(num_batches)
        for i in range(num_batches):
            print(f"\rProgress: {i + 1}/{num_batches}", end="")
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n)
            Xl_i = Xl_flat[start_idx:end_idx]
            Xu_i = Xu_flat[start_idx:end_idx]
            dist = pdist(Xl_i, "sqeuclidean") + pdist(Xu_i, "sqeuclidean")
            dist = dist[dist > 0.0]
            rs_batches[i] = np.quantile(dist, [0.1, 0.9]).sum() / 4
        return rs_batches.mean()


def random_block_prototypes(
    X_lower,
    X_upper,
    n_sample_clusters,
    n_feature_clusters,
    random_state=100,
):
    """
    Initialize block prototypes by randomly selecting interval values from the dataset.

    Parameters
    ----------
    X_lower : ndarray of shape (n_samples, n_features)
        Lower bounds of the observations.

    X_upper : ndarray of shape (n_samples, n_features)
        Upper bounds of the observations.

    n_sample_clusters : int
        Number of sample clusters.

    n_feature_clusters : int
        Number of feature clusters.

    random_state : int, default=100
        Seed used to initialize the random number generator.

    Returns
    -------
    C_lower : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Lower bounds of the prototypes.

    C_upper : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Upper bounds of the prototypes.

    Raises
    ------
    ValueError
        If there are not enough unique interval values to initialize the prototypes.

    """

    rng = np.random.default_rng(random_state)

    Xl_flat = X_lower.flatten()
    Xu_flat = X_upper.flatten()
    jointX = np.vstack([Xl_flat, Xu_flat])
    Xnew = np.unique(jointX, axis=1)
    n_unique = Xnew.shape[1]

    if n_unique < (n_sample_clusters * n_feature_clusters):
        raise ValueError(
            f"Not enough unique values in X ({n_unique}) "
            f"to initialize {n_sample_clusters * n_feature_clusters} distinct prototypes."
        )

    idx = rng.integers(0, n_unique, size=n_sample_clusters * n_feature_clusters)

    C_lower = Xnew[0, idx].reshape((n_sample_clusters, n_feature_clusters))
    C_upper = Xnew[1, idx].reshape((n_sample_clusters, n_feature_clusters))
    return C_lower, C_upper


def squared_distances(X_lower, X_upper, C_lower, C_upper):
    """
    Compute squared distances between interval-valued observations and
    interval-valued prototypes.

    Parameters
    ----------
    X_lower : ndarray of shape (n_samples, n_features)
        Lower bounds of the observations.

    X_upper : ndarray of shape (n_samples, n_features)
        Upper bounds of the observations.

    C_lower : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Lower bounds of the prototypes.

    C_upper : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Upper bounds of the prototypes.

    Returns
    -------
    ndarray of shape (n_sample_clusters, n_feature_clusters, n_samples, n_features)
        Squared distances between observations and prototypes.
    """

    Xl = X_lower[None, None, :, :]
    Xu = X_upper[None, None, :, :]
    Cl = C_lower[:, :, None, None]
    Cu = C_upper[:, :, None, None]
    return (Xl - Cl) ** 2 + (Xu - Cu) ** 2


def kernel_array(X_lower, X_upper, C_lower, C_upper, sigma2):
    """
    Compute the Gaussian kernel values between interval-valued
    observations and prototypes.

    Parameters
    ----------
    X_lower : ndarray of shape (n_samples, n_features)
        Lower bounds of the observations.

    X_upper : ndarray of shape (n_samples, n_features)
        Upper bounds of the observations.

    C_lower : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Lower bounds of the prototypes.

    C_upper : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Upper bounds of the prototypes.

    sigma2 : float
        Gaussian kernel width parameter.

    Returns
    -------
    ndarray of shape (n_sample_clusters, n_feature_clusters, n_samples, n_features)
        Gaussian kernel values between observations and prototypes.
    """
    const = -1 / (2 * sigma2)
    D2 = squared_distances(X_lower, X_upper, C_lower, C_upper)
    return np.exp(const * D2)


def random_fuzzy_partitions(
    n_sample_clusters,
    n_feature_clusters,
    n_samples,
    n_features,
    random_state=26,
):
    """
    Generate random fuzzy partitions for samples and features.

    Parameters
    ----------
    n_sample_clusters : int
        Number of sample clusters.

    n_feature_clusters : int
        Number of feature clusters.

    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    random_state : int, default=100
        Seed for the random number generator.

    Returns
    -------
    G : ndarray of shape (n_samples, n_sample_clusters)
        Sample membership matrix.

    H : ndarray of shape (n_features, n_feature_clusters)
        Feature membership matrix.
    """

    rng = np.random.default_rng(random_state)

    G = rng.uniform(size=(n_samples, n_sample_clusters))
    G /= G.sum(axis=1, keepdims=True)

    H = rng.uniform(size=(n_features, n_feature_clusters))
    H /= H.sum(axis=1, keepdims=True)

    return G, H


def _update_membership_single(
    dk,
    previous_degree,
    exponent,
    eps=1e-10,
):
    """
    Update a fuzzy membership vector from a distance vector.

    Parameters
    ----------
    dk : ndarray of shape (n_clusters,)
        Distances to the clusters prototypes.

    previous_degree : ndarray of shape (n_clusters,)
        Membership degrees from the previous iteration.

    exponent : float
        Membership update exponent.

    eps : float, default=1e-10
        Threshold used to identify near-zero distances.

    Returns
    -------
    ndarray of shape (n_clusters,)
        Updated membership degrees.
    """

    n_clusters = len(dk)
    near_zero = dk < eps

    if np.any(near_zero):
        y = np.zeros(n_clusters)
        d_zeros = dk[near_zero]
        y[near_zero] = d_zeros / np.sum(d_zeros)
        return y
    else:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inv_dk = dk ** (-exponent)
        idx_inf = np.isinf(inv_dk)
        idx_fin = ~idx_inf
        n_inf = idx_inf.sum()

        if n_inf == n_clusters:
            return previous_degree

        inv_dk_fin = inv_dk[idx_fin]
        denom = inv_dk_fin.sum()

        if denom <= 0.0 or not np.isfinite(denom):
            return previous_degree

        if n_inf == 0:
            return inv_dk / denom
        else:
            remaining_mass = 1.0 - previous_degree[idx_inf].sum()
            if remaining_mass > 0.0:
                previous_degree[idx_fin] = remaining_mass * (inv_dk_fin / denom)
            else:
                previous_degree[idx_fin] = 0.0
            return previous_degree


def update_sample_memberships(dist_array, G, H_raised_n, m):
    """
    Update the sample membership matrix.

    Parameters
    ----------
    dist_array : ndarray of shape (n_sample_clusters, n_feature_clusters, n_samples, n_features)
        Distances between observations and prototypes.

    G : ndarray of shape (n_samples, n_sample_clusters)
        Current sample membership matrix.

    H_raised_n : ndarray of shape (n_features, n_feature_clusters)
        Feature membership matrix raised to the fuzzifier parameter ``n``.

    m : float
        Fuzzifier parameter for the sample partition.

    Returns
    -------
    ndarray of shape (n_samples, n_sample_clusters)
        Updated sample membership matrix.
    """

    exponent = 1.0 / (m - 1.0)
    Dik = np.einsum("ph,khnp->nk", H_raised_n, dist_array)
    return np.array([_update_membership_single(d, u, exponent) for d, u in zip(Dik, G)])


def update_feature_memberships(dist_array, H, G_raised_m, n):
    """
    Update the feature membership matrix.

    Parameters
    ----------
    dist_array : ndarray of shape (n_sample_clusters, n_feature_clusters, n_samples, n_features)
        Distances between observations and prototypes.

    H : ndarray of shape (n_features, n_feature_clusters)
        Current feature membership matrix.

    G_raised_m : ndarray of shape (n_samples, n_sample_clusters)
        Sample membership matrix raised to the fuzzifier parameter ``m``.

    n : float
        Fuzzifier parameter for the feature partition.

    Returns
    -------
    ndarray of shape (n_features, n_feature_clusters)
        Updated feature membership matrix.
    """
    exponent = 1.0 / (n - 1.0)
    Djh = np.einsum("nk,khnp->ph", G_raised_m, dist_array)
    return np.array([_update_membership_single(d, v, exponent) for d, v in zip(Djh, H)])


def compute_fuzzy_objective(dist_array, G_raised_m, H_reised_n):
    """
    Compute the fuzzy objective function value.

    Parameters
    ----------
    dist_array : ndarray of shape (n_sample_clusters, n_feature_clusters, n_samples, n_features)
        Distances between observations and prototypes.

    G_raised_m : ndarray of shape (n_samples, n_sample_clusters)
        Sample membership matrix raised to the fuzzifier parameter ``m``.

    H_reised_n : ndarray of shape (n_features, n_feature_clusters)
        Feature membership matrix raised to the fuzzifier parameter ``n``.

    Returns
    -------
    float
        Value of the fuzzy objective function.
    """

    n_sample_clusters, n_feature_clusters, _, _ = dist_array.shape
    Jc = 0.0
    for o in range(n_sample_clusters):
        Go = G_raised_m[:, o]  # (n_samples,)
        for v in range(n_feature_clusters):
            Hv = H_reised_n[:, v]  # (n_features,)
            Jc += Go @ dist_array[o, v] @ Hv

    return Jc


def is_nonincreasing(J_list):
    """
    Check whether a sequence is monotonically non-increasing.

    Parameters
    ----------
    J_list : list of float
        Sequence of objective function values.

    Returns
    -------
    bool
        True if the sequence is monotonically non-increasing,
        False otherwise.
    """

    J_sorted = J_list.copy()
    J_sorted.sort(reverse=True)
    if J_sorted == J_list:
        return True
    else:
        return False
