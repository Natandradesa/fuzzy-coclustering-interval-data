import numpy as np
import pandas as pd
from utils import (
    get_boundaries,
    from_boundaries_to_interval_table,
    check_interval_table,
)
from algorithms.co_clust.utils_co import (
    random_fuzzy_partitions,
    random_block_prototypes,
    kernel_array,
    update_sample_memberships,
    update_feature_memberships,
    compute_fuzzy_objective,
    is_nonincreasing,
)
from sklearn.utils import check_array, check_random_state
from warnings import warn


class IKFDK_O:
    """
    Interval Kernel Fuzzy Double K-means algorithm in the Original space.

    Parameters
    ----------
    n_sample_clusters : int
        Number of sample clusters.

    n_feature_clusters : int
        Number of feature clusters.

    m : float
        Fuzzifier parameter for samples satisfying ``m > 1``.

    n : float
        Fuzzifier parameter for features satisfying ``n > 1``.

    sigma2 : float
        Width parameter of the Gaussian kernel satisfying ``sigma2 > 0``.

    eps : float, default=1e-5
        Convergence tolerance.

    max_iter : int, default=100
        Maximum number of iterations.

    n_init : int, default=1
        Number of random initializations.

    lowest_denominator : float, default=1e-10
        Small value used to avoid numerical instability.

    dp : int, default=4
        Number of decimal places used when constructing interval prototypes.

    Attributes
    ----------
    sample_memberships_ : ndarray of shape (n_samples, n_sample_clusters)
        Learned membership matrix for the samples.

    feature_memberships_ : ndarray of shape (n_features, n_feature_clusters)
        Learned membership matrix for the features.

    prototypes_ : ndarray of shape (n_sample_clusters, n_feature_clusters)
        Learned interval-valued prototypes.

    lower_prototypes_ : ndarray  of shape (n_sample_clusters, n_feature_clusters)
        Lower bounds of the prototypes.

    upper_prototypes_ : ndarray  of shape (n_sample_clusters, n_feature_clusters)
        Upper bounds of the prototypes.

    objective_function_value_ : float
        Objective function value.

    objective_function_history_ : list of float
        Objective function values throughout the optimization.

    n_iter_ : int
        Number of iterations performed.

    """

    def __init__(
        self,
        n_sample_clusters,
        n_feature_clusters,
        m,
        n,
        sigma2,
        eps=1e-5,
        max_iter=100,
        n_init=1,
        lowest_denominator=1e-10,
        dp=4,
    ):

        self.n_sample_clusters = n_sample_clusters
        self.n_feature_clusters = n_feature_clusters
        self.m = m
        self.n = n
        self.sigma2 = sigma2
        self.eps = eps
        self.max_iter = max_iter
        self.n_init = n_init
        self.ld = lowest_denominator
        self.dp = dp

        if self.m <= 1.0:
            raise ValueError(f"m must be greater than 1. Got m={self.m}.")

        if self.n <= 1.0:
            raise ValueError(f"n must be greater than 1. Got n={self.n}.")

        if self.sigma2 <= 0.0:
            raise ValueError(f"sigma2 must be positive. Got sigma2={self.sigma2}.")

        self.sample_memberships_ = None
        self.feature_memberships_ = None
        self.prototypes_ = None
        self.lower_prototypes_ = None
        self.upper_prototypes_ = None
        self.objective_function_value_ = np.inf
        self.objective_function_history_ = None
        self.n_iter_ = None
        self.initial_sample_memberships_ = None
        self.initial_feature_memberships_ = None

    def fit(self, X, random_state=0):
        """
        Fit the IKFDK-O model to an interval-valued dataset.

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
            ensure_min_samples=self.n_sample_clusters,
            ensure_min_features=self.n_feature_clusters,
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

        self.sample_memberships_ = best_rs["G"]
        self.feature_memberships_ = best_rs["H"]
        self.lower_prototypes_ = best_rs["C_lower"]
        self.upper_prototypes_ = best_rs["C_upper"]
        self.objective_function_value_ = best_rs["J"]
        self.objective_function_history_ = best_rs["J_list"]
        self.n_iter_ = best_rs["iterations"]
        self.initial_sample_memberships_ = best_rs["initial_G"]
        self.initial_feature_memberships_ = best_rs["initial_H"]

        self.prototypes_ = from_boundaries_to_interval_table(
            boundaries=[self.lower_prototypes_, self.upper_prototypes_], dp=self.dp
        )

        if is_nonincreasing(self.objective_function_history_) == False:
            warn("Objective function did not converge monotonically")

        return self

    def _update_prototypes(
        self, X_lower, X_upper, G_raised_m, H_raised_n, kernel_array, C_lower, C_upper
    ):
        """
        Update prototype bounds.

        Parameters
        ----------
        X_lower : ndarray of shape (n_samples, n_features)
            Lower bounds of the observations.

        X_upper : ndarray of shape (n_samples, n_features)
            Upper bounds of the observations.

        G_raised_m : ndarray of shape (n_samples, n_sample_clusters)
            Sample membership matrix raised to the exponent ``m``.

        H_raised_n : ndarray of shape (n_features, n_feature_clusters)
            Feature membership matrix raised to the exponent ``n``.

        kernel_array : ndarray of shape (n_sample_clusters, n_feature_clusters, n_samples, n_features)
            Kernel array between observations and prototypes.

        C_lower : ndarray of shape (n_clusters, n_features)
            Current lower bounds of the prototypes.

        C_upper : ndarray of shape (n_clusters, n_features)
            Current upper bounds prototype bounds.

        Returns
        -------
        C_lower : ndarray of shape (n_sample_clusters, n_feature_clusters)
            Updated lower prototype bounds.

        C_upper : ndarray of shape (n_sample_clusters, n_feature_clusters)
            Updated upper prototype bounds.
        """

        C_lower_new = np.empty_like(C_lower)
        C_upper_new = np.empty_like(C_upper)
        for o in range(self.n_sample_clusters):
            g_o = G_raised_m[:, o]
            for v in range(self.n_feature_clusters):
                h_v = H_raised_n[:, v]
                KA_ov = kernel_array[o, v]
                denom = g_o @ KA_ov @ h_v
                if denom > self.ld:
                    C_lower_new[o, v] = (g_o @ (KA_ov * X_lower) @ h_v) / denom
                    C_upper_new[o, v] = (g_o @ (KA_ov * X_upper) @ h_v) / denom
                else:
                    C_lower_new[o, v] = C_lower[o, v]
                    C_upper_new[o, v] = C_upper[o, v]

        return C_lower_new, C_upper_new

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
            membership matrix for samples and features, prototypes, objective function value,
            objective function history and number of iterations.
        """

        # Initialization
        G, H = random_fuzzy_partitions(
            n_sample_clusters=self.n_sample_clusters,
            n_feature_clusters=self.n_feature_clusters,
            n_samples=self.n_samples,
            n_features=self.n_features,
            random_state=random_state,
        )
        initial_G = G
        initial_H = H
        G_raised_m = G**self.m
        H_raised_n = H**self.n

        C_lower, C_upper = random_block_prototypes(
            X_lower=X_lower,
            X_upper=X_upper,
            n_sample_clusters=self.n_sample_clusters,
            n_feature_clusters=self.n_feature_clusters,
            random_state=random_state,
        )

        KA = kernel_array(
            X_lower=X_lower,
            X_upper=X_upper,
            C_lower=C_lower,
            C_upper=C_upper,
            sigma2=self.sigma2,
        )

        J_list = []
        J = np.inf

        # iterative step

        for it in range(1, self.max_iter + 1):
            C_lower, C_upper = self._update_prototypes(
                X_lower=X_lower,
                X_upper=X_upper,
                G_raised_m=G_raised_m,
                H_raised_n=H_raised_n,
                kernel_array=KA,
                C_lower=C_lower,
                C_upper=C_upper,
            )

            KA = kernel_array(
                X_lower=X_lower,
                X_upper=X_upper,
                C_lower=C_lower,
                C_upper=C_upper,
                sigma2=self.sigma2,
            )
            dist_array = 2.0 - 2.0 * KA

            G = update_sample_memberships(
                dist_array=dist_array, G=G, H_raised_n=H_raised_n, m=self.m
            )
            G_raised_m = G**self.m
            H = update_feature_memberships(
                dist_array=dist_array, H=H, G_raised_m=G_raised_m, n=self.n
            )
            H_raised_n = H**self.n
            J_old = J
            J = compute_fuzzy_objective(
                dist_array=dist_array, G_raised_m=G_raised_m, H_reised_n=H_raised_n
            )
            J_list.append(J)

            if np.abs(J - J_old) < self.eps:
                break

        return {
            "G": G,
            "H": H,
            "C_lower": C_lower,
            "C_upper": C_upper,
            "J": J,
            "J_list": J_list,
            "iterations": it,
            "initial_G": initial_G,
            "initial_H": initial_H,
        }
