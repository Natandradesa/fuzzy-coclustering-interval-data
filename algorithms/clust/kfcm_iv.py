import numpy as np
import pandas as pd
from utils.base import (
    get_boundaries,
    from_boundaries_to_interval_table,
    check_interval_table,
)
from sklearn.utils import check_array, check_random_state
from warnings import warn
from scipy.spatial.distance import cdist
from algorithms.clust.utils import BaseFuzzy


class KFCM_IV(BaseFuzzy):
    """
    Kernel Fuzzy C-Means for Interval Data

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    m : float
        Fuzzifier parameter satisfying ``m > 1``.

    sigma2 : float
        Width parameter of the Gaussian kernel satisfying ``sigma2 > 0``.

    epsilon : float, default=1e-5
        Convergence tolerance.

    max_iter : int, default=100
        Maximum number of iterations.

    n_init : int, default=1
        Number of random initializations.

    lowest_denominator : float, default=1e-100
        Small value used to avoid numerical instability.

    dp : int, default=4
        Number of decimal places used when constructing interval prototypes.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Learned membership matrix.

    prototypes_ : DataFrame or ndarray of shape (n_clusters, n_features)
        Learned interval-valued prototypes.

    lower_prototypes_ : ndarray  of shape (n_clusters, n_features)
        Lower bounds of the prototypes.

    upper_prototypes_ : ndarray  of shape (n_clusters, n_features)
        Upper bounds of the prototypes.

    objective_function_value_ : float
        Objective function value.

    objective_function_history_ : list of float
        Objective function values throughout the optimization.

    n_iter_ : int
        Number of iterations performed.

    References
    ----------
    .. [1] Pimentel, B. A., da Costa, A. F., & de Souza, R. M. (2011, June).
           "Kernel-based fuzzy clustering of interval data."
           2011 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2011)
           (pp. 497-501). IEEE.
    """

    def __init__(
        self,
        n_clusters,
        m,
        sigma2,
        epsilon=1e-5,
        max_iter=100,
        n_init=1,
        lowest_denominator=1e-100,
        dp=4,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.sigma2 = sigma2
        self.max_iter = max_iter
        self.n_init = n_init
        self.ld = lowest_denominator
        self.dp = dp

        self.memberships_ = None
        self.prototypes_ = None
        self.lower_prototypes_ = None
        self.upper_prototypes_ = None
        self.objective_function_value_ = np.inf
        self.objective_function_history_ = None
        self.n_iter_ = None
        self.initial_memberships_ = None
        self.dist_matrix_ = None
        self.variable_names = None

        if self.m <= 1.0:
            raise ValueError(f"m must be greater than 1. Got m={self.m}.")

        if self.sigma2 <= 0.0:
            raise ValueError(f"sigma2 must be positive. Got sigma2={self.sigma2}.")

    def fit(self, X, random_state=0):
        """
        Fit the KFCM-IV model to an interval-valued dataset.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Interval-valued dataset.

        random_state : int, default=0
            Seed for the random number generator.

        Returns
        -------
        self
            Fitted estimator.
        """

        random_state = check_random_state(random_state)

        check_array(
            X,
            accept_sparse=False,
            dtype=None,
            order=None,
            copy=False,
            ensure_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=self.n_clusters,
            ensure_min_features=1,
            estimator=None,
        )

        check_interval_table(X)

        X_lower, X_upper = get_boundaries(interval_table=X)
        if isinstance(X, pd.DataFrame):
            self.variable_names = list(X.columns)
            X_lower = X_lower.to_numpy().astype(float)
            X_upper = X_upper.to_numpy().astype(float)
        else:
            X_lower = X_lower.astype(float)
            X_upper = X_upper.astype(float)

        self.n_samples, self.n_features = X_lower.shape

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        J_best = np.inf
        for seed in seeds:
            rs = self._fit_single(
                X_lower=X_lower,
                X_upper=X_upper,
                random_state=seed,
            )
            J = rs["J"]
            if np.isnan(J):
                raise ValueError("matrix may contain negative or unexpected NaN values")

            if J < J_best:
                J_best = J
                best_rs = rs

        self.memberships_ = best_rs["G"]
        self.prototypes_ = best_rs["C"]
        self.lower_prototypes_ = best_rs["C_lower"]
        self.upper_prototypes_ = best_rs["C_upper"]
        self.objective_function_value_ = best_rs["J"]
        self.objective_function_history_ = best_rs["J_list"]
        self.n_iter_ = best_rs["iterations"]
        self.initial_memberships_ = best_rs["initial_G"]
        self.dist_matrix_ = best_rs["dist"]

        if super().isdecreasing(self.objective_function_history_) == False:
            warn("Objective function did not converge monotonically")

        return self

    def _updater_prototypes(
        self,
        X_lower,
        X_upper,
        G_raised_m,
        kernel_matrix,
        C_lower,
        C_upper,
    ):
        """
        Update prototype bounds.

        Parameters
        ----------
        X_lower : ndarray of shape (n_samples, n_features)
            Lower bounds of the observations.

        X_upper : ndarray of shape (n_samples, n_features)
            Upper bounds of the observations.

        G_raised_m : ndarray of shape (n_samples, n_clusters)
            Membership matrix raised to the exponent ``m``.

        kernel_matrix : ndarray of shape (n_samples, n_clusters)
            Kernel matrix between observations and prototypes.

        C_lower : ndarray of shape (n_clusters, n_features)
            Current lower bounds of the prototypes.

        C_upper : ndarray of shape (n_clusters, n_features)
            Current upper bounds prototype bounds.

        Returns
        -------
        C_lower : ndarray of shape (n_clusters, n_features)
            Updated lower prototype bounds.

        C_upper : ndarray of shape (n_clusters, n_features)
            Updated upper prototype bounds.
        """

        for k in range(self.n_clusters):
            Gmk = G_raised_m[:, k] * kernel_matrix[:, k]
            if Gmk.sum() > self.ld:
                C_lower[k] = np.average(a=X_lower, axis=0, weights=Gmk)
                C_upper[k] = np.average(a=X_upper, axis=0, weights=Gmk)
        return C_lower, C_upper

    def _fit_single(self, X_lower, X_upper, random_state):
        """
        Run a single initialization of the algorithm.

        Parameters
        ----------
        X_lower : ndarray of shape (n_samples, n_features)
            Lower bounds of the observations.

        X_upper : ndarray of shape (n_samples, n_features)
            Upper bounds of the observations.

        random_state : int
            Seed for the random number generator.

        Returns
        -------
        dict
            Dictionary containing the optimization results, including the
            membership matrix, prototypes, objective function value,
            objective function history, number of iterations, and distance
            matrix.
        """

        # Initialization
        G = self.random_fuzzy_partition(
            n_clusters=self.n_clusters,
            n_samples=self.n_samples,
            random_state=random_state,
        )
        initial_G = G
        G_raised_m = G**self.m
        C_lower, C_upper = self.initialize_prototypes(
            X_lower=X_lower,
            X_upper=X_upper,
            n_clusters=self.n_clusters,
            random_state=random_state,
        )

        SED = cdist(XA=X_lower, XB=C_lower, metric="sqeuclidean") + cdist(
            XA=X_upper, XB=C_upper, metric="sqeuclidean"
        )
        kernel_matrix = np.exp((-1 / (2 * self.sigma2)) * SED)

        J = np.inf
        J_list = []

        # iterative step
        for it in range(1, self.max_iter + 1):
            C_lower, C_upper = self._updater_prototypes(
                X_lower=X_lower,
                X_upper=X_upper,
                G_raised_m=G_raised_m,
                kernel_matrix=kernel_matrix,
                C_lower=C_lower,
                C_upper=C_upper,
            )

            SED = cdist(XA=X_lower, XB=C_lower, metric="sqeuclidean") + cdist(
                XA=X_upper, XB=C_upper, metric="sqeuclidean"
            )

            kernel_matrix = np.exp((-1 / (2 * self.sigma2)) * SED)
            Dist = 2.0 - 2.0 * kernel_matrix

            G = self.fuzzy_assignment_rule(dist_matrix=Dist, G=G, m=self.m)
            G_raised_m = G**self.m
            J_old = J
            J = np.sum(G_raised_m * Dist)
            J_list.append(J)
            if np.abs(J - J_old) < self.epsilon:
                break

        C = from_boundaries_to_interval_table(boundaries=[C_lower, C_upper], dp=self.dp)
        if self.variable_names is not None:
            C = pd.DataFrame(C, columns=self.variable_names)

        return {
            "G": G,
            "C": C,
            "C_lower": C_lower,
            "C_upper": C_upper,
            "J": J,
            "J_list": J_list,
            "iterations": it,
            "dist": Dist,
            "initial_G": initial_G,
        }
