import numpy as np
import sys
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import from_boundaries_to_interval_table


def generate_data(
    n_samples, n_features, averages, stds, unif_interval=(0, 1), random_state=None
):

    averages = np.asarray(averages)
    stds = np.asarray(stds)

    if averages.shape != stds.shape:
        raise ValueError("averages and stds must have the same shape")

    n_sample_clusters, n_feature_clusters = averages.shape
    rng = np.random.default_rng(random_state)

    base_row, rem_row = divmod(n_samples, n_sample_clusters)
    nb_objects = np.array(
        [base_row + 1] * rem_row + [base_row] * (n_sample_clusters - rem_row), dtype=int
    )

    base_col, rem_col = divmod(n_features, n_feature_clusters)
    nb_features = np.array(
        [base_col + 1] * rem_col + [base_col] * (n_feature_clusters - rem_col),
        dtype=int,
    )

    row_bounds = np.concatenate(([0], np.cumsum(nb_objects)))
    col_bounds = np.concatenate(([0], np.cumsum(nb_features)))

    labels_objects = np.repeat(np.arange(n_sample_clusters), nb_objects)
    labels_variables = np.repeat(np.arange(n_feature_clusters), nb_features)

    X = np.empty((n_samples, n_features), dtype=float)

    for k in range(n_sample_clusters):
        ik = slice(row_bounds[k], row_bounds[k + 1])
        for h in range(n_feature_clusters):
            jh = slice(col_bounds[h], col_bounds[h + 1])
            X[ik, jh] = rng.normal(
                loc=averages[k, h],
                scale=stds[k, h],
                size=(nb_objects[k], nb_features[h]),
            )

    low, high = unif_interval
    gamma = rng.uniform(low=low, high=high, size=(n_samples, n_features))

    X_lower = X - gamma
    X_upper = X + gamma

    X_interval = from_boundaries_to_interval_table(boundaries=(X_lower, X_upper))
    return (
        X_interval,
        labels_objects.astype(np.int64),
        labels_variables.astype(np.int64),
    )


from functools import partial

# Synthetic Dataset 1

averages_1 = np.array(
    [
        [1.0, 2.0, 0.0],
        [3.0, 2.1, 0.0],
        [5.0, 2.2, 0.0],
    ]
)

stds_1 = np.array(
    [
        [0.5, 1.0, 5.0],
        [0.5, 1.0, 5.0],
        [0.5, 1.0, 5.0],
    ]
)

synthetic_dataset1 = partial(
    generate_data,
    averages=averages_1,
    stds=stds_1,
    unif_interval=(0, 2),
)


# Synthetic Dataset 2

averages_2 = np.array(
    [
        [1.0, 3.0, 0.0],
        [2.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)

stds_2 = np.array(
    [
        [1.0, 1.0, 5.0],
        [1.0, 1.0, 5.0],
        [5.0, 5.0, 5.0],
    ]
)

synthetic_dataset2 = partial(
    generate_data,
    averages=averages_2,
    stds=stds_2,
    unif_interval=(0, 2),
)
