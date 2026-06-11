from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from metrics import (
    hullermeier_index,
    adjusted_rand_index_coclustering,
)


def clustering_metrics(y_true, memberships):
    """
    Compute clustering metrics.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True labels.

    memberships : ndarray of shape (n_samples, n_sample_clusters) or
    (n_features, n_feature_clusters)

        Fuzzy membership matrix.

    Returns
    -------
    tuple
        (ARI, HUL)
    """

    y_pred = np.argmax(memberships, axis=1)

    ari = adjusted_rand_score(
        labels_true=y_true,
        labels_pred=y_pred,
    )

    hul = hullermeier_index(
        U=memberships,
        P=y_true,
    )

    return ari, hul


def save_results(results, output_file):
    """
    Save benchmark results.

    Parameters
    ----------
    results : pandas.DataFrame
        Results table.

    output_file : str or Path
        Destination file.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_file,
        index=False,
    )


# Clustering


def _single_clustering_run(
    random_state,
    model_row,
    model_col,
    synthetic_dataset,
    shape,
):

    n_samples, n_features = shape

    X, y_row, y_col = synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    mdl_row = copy.deepcopy(model_row)
    mdl_col = copy.deepcopy(model_col)

    t0 = time.time()

    mdl_row.fit(
        X=X,
        random_state=random_state,
    )

    mdl_col.fit(
        X=X.T,
        random_state=random_state,
    )

    elapsed = time.time() - t0

    ari_row, hul_row = clustering_metrics(
        y_true=y_row,
        memberships=mdl_row.memberships_,
    )

    ari_col, hul_col = clustering_metrics(
        y_true=y_col,
        memberships=mdl_col.memberships_,
    )

    y_row_pred = np.argmax(
        mdl_row.memberships_,
        axis=1,
    )

    y_col_pred = np.argmax(
        mdl_col.memberships_,
        axis=1,
    )

    cari = adjusted_rand_index_coclustering(
        z_true=y_row,
        z_pred=y_row_pred,
        w_true=y_col,
        w_pred=y_col_pred,
    )

    return {
        "ARI": ari_row,
        "ARI_col": ari_col,
        "HUL": hul_row,
        "HUL_col": hul_col,
        "CARI": cari,
        "TIME": elapsed,
    }


def run_clustering(
    model_row,
    model_col,
    synthetic_dataset,
    n_rep=100,
    shape=(60, 60),
    n_jobs=-1,
    random_state=0,
):

    rng = np.random.default_rng(
        seed=random_state,
    )

    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=n_rep,
        dtype=np.int32,
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
    )(
        delayed(_single_clustering_run)(
            random_state=seed,
            model_row=model_row,
            model_col=model_col,
            synthetic_dataset=synthetic_dataset,
            shape=shape,
        )
        for seed in seeds
    )

    return pd.DataFrame(results)


# Co-clustering


def _single_coclustering_run(
    random_state,
    model,
    synthetic_dataset,
    shape,
):

    n_samples, n_features = shape

    X, y_row, y_col = synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    mdl = copy.deepcopy(model)

    t0 = time.time()

    mdl.fit(
        X=X,
        random_state=random_state,
    )

    elapsed = time.time() - t0

    ari_row, hul_row = clustering_metrics(
        y_true=y_row,
        memberships=mdl.sample_memberships_,
    )

    ari_col, hul_col = clustering_metrics(
        y_true=y_col,
        memberships=mdl.feature_memberships_,
    )

    y_row_pred = np.argmax(
        mdl.sample_memberships_,
        axis=1,
    )

    y_col_pred = np.argmax(
        mdl.feature_memberships_,
        axis=1,
    )

    cari = adjusted_rand_index_coclustering(
        z_true=y_row,
        z_pred=y_row_pred,
        w_true=y_col,
        w_pred=y_col_pred,
    )

    return {
        "ARI": ari_row,
        "ARI_col": ari_col,
        "HUL": hul_row,
        "HUL_col": hul_col,
        "CARI": cari,
        "TIME": elapsed,
    }


def run_coclustering(
    model,
    synthetic_dataset,
    n_rep=100,
    shape=(60, 60),
    n_jobs=-1,
    random_state=0,
):

    rng = np.random.default_rng(
        seed=random_state,
    )

    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=n_rep,
        dtype=np.int32,
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
    )(
        delayed(_single_coclustering_run)(
            random_state=seed,
            model=model,
            synthetic_dataset=synthetic_dataset,
            shape=shape,
        )
        for seed in seeds
    )

    return pd.DataFrame(results)
