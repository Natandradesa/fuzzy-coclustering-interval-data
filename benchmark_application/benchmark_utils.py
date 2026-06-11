from pathlib import Path
import copy
import time
import sys

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from metrics import hullermeier_index


def numerical_class(y):
    return np.unique(y, return_inverse=True)[1]


def read_dataset(path):
    df = pd.read_csv(path, index_col=0)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"])
    return X, y


def get_metrics(y_true, memberships, elapsed_time, objective_function_value):
    y_pred = np.argmax(memberships, axis=1)

    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "HUL": hullermeier_index(U=memberships, P=y_true),
        "TIME": elapsed_time,
        "J": objective_function_value,
    }


def run_n_times(
    X,
    y,
    model,
    n_runs=100,
    random_state=100,
    n_jobs=1,
):

    seeds = random_state + np.arange(n_runs)

    def _single_run(seed):

        mdl = copy.deepcopy(model)

        start = time.time()
        fitted_model = mdl.fit(X=X, random_state=seed)
        elapsed = time.time() - start

        if hasattr(fitted_model, "memberships_"):
            memberships = fitted_model.memberships_
        else:
            memberships = fitted_model.sample_memberships_

        return get_metrics(
            y_true=y,
            memberships=memberships,
            elapsed_time=elapsed,
            objective_function_value=fitted_model.objective_function_value_,
        )

    results = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
    )(delayed(_single_run)(seed) for seed in seeds)

    return pd.DataFrame(results)
