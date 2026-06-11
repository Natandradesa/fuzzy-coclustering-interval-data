from pathlib import Path

import pandas as pd

from benchmark_utils import read_dataset
from algorithms.clust.utils import caputo_estimator
from algorithms.co_clust.utils_co import (
    caputo_estimator_co_clustering,
)

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT.parent / "interval_datasets"

ESTIMATIVES_PATH = ROOT / "caputo_estimatives.csv"

DATASETS = [
    "abalone",
    "car_models",
    "horses",
    "fish",
    "fungi",
    "temperature",
]

width_estimatives = pd.DataFrame(
    index=DATASETS,
    columns=["clust", "co_clust"],
    dtype=float,
)

for dataset_name in DATASETS:

    print(f"Dataset: {dataset_name}")

    X, _ = read_dataset(DATA_DIR / f"{dataset_name}.txt")

    width_estimatives.loc[
        dataset_name,
        "clust",
    ] = caputo_estimator(X)

    width_estimatives.loc[
        dataset_name,
        "co_clust",
    ] = caputo_estimator_co_clustering(X)

width_estimatives.to_csv(
    ESTIMATIVES_PATH,
    index=True,
)

print("Task completed.")
