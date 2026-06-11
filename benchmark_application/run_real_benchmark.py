from pathlib import Path

import pandas as pd

from benchmark_utils import (
    read_dataset,
    numerical_class,
    run_n_times,
)

from algorithms import (
    IFCM,
    KFCM_IV,
    IFDK,
    IKFDK_O,
    IKFDK_K,
)

# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

M = 1.1
N = 1.1
N_INIT = 1
N_RUNS = 100
N_JOBS = 2


DATASETS = [
    "abalone",
    "car_models",
    "horses",
    "fish",
    "fungi",
    "temperature",
]

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT.parent / "interval_datasets"
RESULTS_DIR = ROOT / "results"

# --------------------------------------------------


def create_models(
    n_clusters,
    sigma2_clustering,
    sigma2_coclustering,
):

    return {
        "IFCM": IFCM(
            n_clusters=n_clusters,
            m=M,
            n_init=N_INIT,
        ),
        "KFCM_IV": KFCM_IV(
            n_clusters=n_clusters,
            m=M,
            sigma2=sigma2_clustering,
            n_init=N_INIT,
        ),
        "IFDK": IFDK(
            n_sample_clusters=n_clusters,
            n_feature_clusters=n_clusters,
            m=M,
            n=N,
            n_init=N_INIT,
        ),
        "IKFDK_O": IKFDK_O(
            n_sample_clusters=n_clusters,
            n_feature_clusters=n_clusters,
            m=M,
            n=N,
            sigma2=sigma2_coclustering,
            n_init=N_INIT,
        ),
        "IKFDK_K": IKFDK_K(
            n_sample_clusters=n_clusters,
            n_feature_clusters=n_clusters,
            m=M,
            n=N,
            sigma2=sigma2_coclustering,
            n_init=N_INIT,
        ),
    }


def main():
    CAPUTO_ESTIMATIVES = pd.read_csv("caputo_estimatives.csv", index_col=0)

    RESULTS_DIR.mkdir(exist_ok=True)

    for dataset_name in DATASETS:

        print(f"\nDataset: {dataset_name}")

        X, y = read_dataset(DATA_DIR / f"{dataset_name}.txt")

        y = numerical_class(y)

        n_clusters = len(set(y))
        sigma2_clust = CAPUTO_ESTIMATIVES.loc[dataset_name, "clust"]
        sigma2_co_clust = CAPUTO_ESTIMATIVES.loc[dataset_name, "co_clust"]

        models = create_models(
            n_clusters=n_clusters,
            sigma2_clustering=sigma2_clust,
            sigma2_coclustering=sigma2_co_clust,
        )

        dataset_dir = RESULTS_DIR / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        for method_name, model in models.items():

            print(f"  -> {method_name}")

            results = run_n_times(
                X=X,
                y=y,
                model=model,
                n_runs=N_RUNS,
                random_state=100,
                n_jobs=N_JOBS,
            )

            results.to_csv(
                dataset_dir / f"{method_name}.txt",
                index=False,
            )


if __name__ == "__main__":
    main()
