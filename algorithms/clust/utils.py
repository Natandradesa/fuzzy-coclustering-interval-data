import numpy as np

from utils.base import (
    get_boundaries,
    check_interval_table,
)
from scipy.spatial.distance import pdist


def caputo_estimator(X):
    """
    Estimate the Gaussian Kernel width parameter for interval-valued data using Caputo's heuristic.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Interval-valued dataset.

    Returns
    -------
    float
        Estimated Gaussian kernel width parameter.
    """

    check_interval_table(X)
    Xl, Xu = get_boundaries(interval_table=X)
    Dl = pdist(X=Xl, metric="sqeuclidean")
    Du = pdist(X=Xu, metric="sqeuclidean")
    D = Dl + Du
    Q = np.quantile(D[D > 0.0], q=[0.1, 0.9])
    return sum(Q) / 4.0


class BaseFuzzy:
    def initialize_prototypes(
        self,
        X_lower,
        X_upper,
        n_clusters,
        random_state=None,
    ):
        """
        Initialize prototypes by randomly selecting observations.

        Parameters
        ----------
        X_lower : ndarray of shape (n_samples, n_features)
            Lower bounds of the observations.

        X_upper : ndarray of shape (n_samples, n_features)
            Upper bounds of the observations.

        n_clusters : int
            Number of clusters.

        random_state : int, default=None
            Seed for the random number generator.

        Returns
        -------
        lower_prototypes : ndarray of shape (n_clusters, n_features)
            Lower bounds of the initialized prototypes.

        upper_prototypes : ndarray of shape (n_clusters, n_features)
            Upper bounds of the initialized prototypes.
        """

        np.random.seed(random_state)
        jointX = np.concatenate([X_lower, X_upper], axis=1)
        Xnew = np.unique(jointX, axis=0)
        nNew = Xnew.shape[0]
        Ksort = np.random.choice(nNew, n_clusters, replace=False)
        lower_prototypes = X_lower[Ksort]
        upper_prototypes = X_upper[Ksort]
        return lower_prototypes, upper_prototypes

    def random_fuzzy_partition(
        self,
        n_clusters,
        n_samples,
        random_state=26,
    ):
        """
        Generate random fuzzy partitions for samples and features.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.

        n_samples : int
            Number of samples.

        random_state : int, default=26
            Seed for the random number generator.

        Returns
        -------
        G : ndarray of shape (n_samples, n_clusters)
            Sample membership matrix.

        """

        rng = np.random.default_rng(random_state)
        G = rng.uniform(size=(n_samples, n_clusters))
        G /= G.sum(axis=1, keepdims=True)
        return G

    def _updater_fuzzy_membership(
        self,
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
            y[near_zero] = 1.0 / np.sum(near_zero)
            # d_zeros = dk[near_zero]
            # y[near_zero] = d_zeros/np.sum(d_zeros)
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

    def fuzzy_assignment_rule(self, dist_matrix, G, m):
        """
        Update the sample membership matrix.

        Parameters
        ----------
        dist_matrix : matrix of shape (n_samples, n_clusters)
            Distances between observations and prototypes.

        G : ndarray of shape (n_samples, n_clusters)
            Current sample membership matrix.

        m : float
            Fuzzifier parameter satisfying ``m > 1``.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            Updated sample membership matrix.
        """

        exponent = 1.0 / (m - 1.0)
        return np.array(
            [
                self._updater_fuzzy_membership(d, u, exponent)
                for d, u in zip(dist_matrix, G)
            ]
        )

    def isdecreasing(self, J):
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
        J_sorted = J.copy()
        J_sorted.sort(reverse=True)
        if J_sorted == J:
            return True
        else:
            return False
