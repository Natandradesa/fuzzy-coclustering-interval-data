# import numpy as np
# import pandas as pd
# from utils import get_boundaries, check_interval_table
# from .utils_co import (random_fuzzy_partitions, fuzzy_assignment_rule_for_objects,
#                        fuzzy_assignment_rule_for_variables, computeJ_fuzzy, isdecreasing)

# from sklearn.utils import check_array, check_random_state
# from warnings import warn

# EPS = 1e-100

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


class IKFDK_K:
    """
    Interval Kernel Fuzzy Double K-means algorithm in the Kernel space.

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
        self.objective_function_value_ = np.inf
        self.objective_function_history_ = None
        self.n_iter_ = None
        self.initial_sample_memberships_ = None
        self.initial_feature_memberships_ = None

    def fit(self, X, random_state=0):
        """
        Fit the IKFDK-K model to an interval-valued dataset.

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
        self.objective_function_value_ = best_rs["J"]
        self.objective_function_history_ = best_rs["J_list"]
        self.n_iter_ = best_rs["iterations"]
        self.initial_sample_memberships_ = best_rs["initial_G"]
        self.initial_feature_memberships_ = best_rs["initial_H"]

        if is_nonincreasing(self.objective_function_history_) == False:
            warn("Objective function did not converge monotonically")

        return self

    # def _compute_distances_old(self, X_lower, X_upper, Um, Vn, D_old):
    #     N, P = self.N, self.P
    #     K, H = self.K, self.H
    #     D = D_old.copy()

    #     inv_2sigma2 = 1.0 / (2.0 * self.sigma2)

    #     for k in range(K):
    #         u_k = Um[:, k]
    #         Nk = u_k.sum()

    #         for h in range(H):
    #             v_h = Vn[:, h]
    #             Nh = v_h.sum()
    #             den = Nk * Nh

    #             if den < EPS:
    #                 continue

    #             A_kh = np.zeros((N, P))

    #             for i in range(N):
    #                 for j in range(P):
    #                     diff = (X_lower[i, j] - X_lower) ** 2 + (
    #                         X_upper[i, j] - X_upper
    #                     ) ** 2
    #                     K_ij = np.exp(-diff * inv_2sigma2)
    #                     A_kh[i, j] = np.sum(u_k[:, None] * v_h[None, :] * K_ij)

    #             c_kh = np.sum(u_k[:, None] * v_h[None, :] * A_kh)
    #             D[k, h] = 1.0 - 2.0 / den * A_kh + c_kh / (den**2)
    #     return D

    def _compute_distances(self, X_lower, X_upper, G_raised_m, H_raised_n, D_old):
        D = D_old.copy()

        inv_2sigma2 = 1.0 / (2.0 * self.sigma2)

        for o in range(self.n_sample_clusters):
            g_o = G_raised_m[:, o]
            No = g_o.sum()

            for v in range(self.n_feature_clusters):
                h_v = H_raised_n[:, v]
                Nv = h_v.sum()
                den = No * Nv

                if den < self.ld:
                    continue

                A_ov = np.zeros((self.n_samples, self.n_features))

                for i in range(self.n_samples):
                    for d in range(self.n_features):
                        diff = (X_lower[i, d] - X_lower) ** 2 + (
                            X_upper[i, d] - X_upper
                        ) ** 2
                        K_id = np.exp(-diff * inv_2sigma2)
                        A_ov[i, d] = np.sum(g_o[:, None] * h_v[None, :] * K_id)

                c_ov = np.sum(g_o[:, None] * h_v[None, :] * A_ov)
                D[o, v] = 1.0 - (2.0 / den) * A_ov + c_ov / (den**2)
        return D

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

        dist_array = np.random.uniform(
            size=(
                self.n_sample_clusters,
                self.n_feature_clusters,
                self.n_samples,
                self.n_features,
            )
        )  # Used only in division by zero at the first iteration.

        J_list = []
        J = np.inf

        # iterative step

        for it in range(1, self.max_iter + 1):
            dist_array = self._compute_distances(
                X_lower=X_lower,
                X_upper=X_upper,
                G_raised_m=G_raised_m,
                H_raised_n=H_raised_n,
                D_old=dist_array,
            )

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
            "J": J,
            "J_list": J_list,
            "iterations": it,
            "initial_G": initial_G,
            "initial_H": initial_H,
        }
