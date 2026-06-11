from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from algorithms import IFDK, IKFDK_O, IKFDK_K

from data_generation import synthetic_dataset1, synthetic_dataset2
from synthetic_utils import run_coclustering, save_results

# Settings

M = 1.1
N = 1.1
N_REP = 100
N_INIT = 50
N_SAMPLE_CLUSTERS = 3
N_FEATURE_CLUSTERS = 3
SHAPE = (60, 60)
N_JOBS = 2


RESULTS_DIR = ROOT / "results"
SIGMA2_FILE = ROOT / "caputo_estimatives_synthetic.csv"
caputo_estimatives = pd.read_csv(SIGMA2_FILE, index_col=0)


DATASET_MAPPING = {"synthetic_1": synthetic_dataset1, "synthetic_2": synthetic_dataset2}


for dataset_name, dataset in DATASET_MAPPING.items():

    print(f"\nDataset: {dataset_name}")

    sigma2_coclustering = caputo_estimatives.loc[dataset_name, "sigma2_co_clust"]

    models = {
        "IFDK": IFDK(
            n_sample_clusters=N_SAMPLE_CLUSTERS,
            n_feature_clusters=N_FEATURE_CLUSTERS,
            m=M,
            n=N,
            n_init=N_INIT,
        ),
        "IKFDK_O": IKFDK_O(
            n_sample_clusters=N_SAMPLE_CLUSTERS,
            n_feature_clusters=N_FEATURE_CLUSTERS,
            m=M,
            n=N,
            sigma2=sigma2_coclustering,
            n_init=N_INIT,
        ),
        "IKFDK_K": IKFDK_K(
            n_sample_clusters=N_SAMPLE_CLUSTERS,
            n_feature_clusters=N_FEATURE_CLUSTERS,
            m=M,
            n=N,
            sigma2=sigma2_coclustering,
            n_init=N_INIT,
        ),
    }

    for model_name, model in models.items():

        print(f"  -> {model_name}")

        results = run_coclustering(
            model=model,
            synthetic_dataset=dataset,
            n_rep=N_REP,
            shape=SHAPE,
            n_jobs=N_JOBS,
            random_state=10,
        )

        output_file = RESULTS_DIR / dataset_name / f"{model_name}.txt"
        save_results(results=results, output_file=output_file)

print("\nTask completed.")
