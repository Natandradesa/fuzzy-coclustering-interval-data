from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from algorithms.clust.utils import caputo_estimator
from algorithms.co_clust.utils_co import caputo_estimator_co_clustering
from data_generation import synthetic_dataset1, synthetic_dataset2

# ============================================================
# Settings
# ============================================================

n_samples = 60
n_features = 60

OUTPUT_FILE = ROOT / "caputo_estimatives_synthetic.csv"


# ============================================================
# Compute sigma² values
# ============================================================


DATASET_MAPPING = {"synthetic_1": synthetic_dataset1, "synthetic_2": synthetic_dataset2}

results = pd.DataFrame(
    index=DATASET_MAPPING.keys(),
    columns=[
        "sigma2_row",
        "sigma2_col",
        "sigma2_co_clust",
    ],
    dtype=float,
)

for dataset_name, dataset in DATASET_MAPPING.items():

    print(f"Dataset: {dataset_name}")

    X, _, _ = dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42,
    )

    results.loc[
        dataset_name,
        "sigma2_row",
    ] = caputo_estimator(X)

    results.loc[
        dataset_name,
        "sigma2_col",
    ] = caputo_estimator(X.T)

    results.loc[
        dataset_name,
        "sigma2_co_clust",
    ] = caputo_estimator_co_clustering(X)

results.to_csv(
    OUTPUT_FILE,
    index=True,
)

print("\nTask completed.")
print(f"Results saved to:\n{OUTPUT_FILE}")
