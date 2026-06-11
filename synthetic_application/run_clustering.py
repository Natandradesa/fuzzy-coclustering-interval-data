from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from algorithms import IFCM, KFCM_IV

from data_generation import synthetic_dataset1, synthetic_dataset2
from synthetic_utils import run_clustering, save_results

# Settings


M = 1.1
N_REP = 100
N_INIT = 50
SHAPE = (60, 60)
N_JOBS = 2
N_CLUSTERS = 3

RESULTS_DIR = ROOT / "results"
SIGMA2_FILE = ROOT / "caputo_estimatives_synthetic.csv"
caputo_estimatives = pd.read_csv(SIGMA2_FILE, index_col=0)


DATASET_MAPPING = {"synthetic_1": synthetic_dataset1, "synthetic_2": synthetic_dataset2}


for dataset_name, dataset in DATASET_MAPPING.items():

    print(f"\nDataset: {dataset_name}")

    sigma2_row = caputo_estimatives.loc[dataset_name, "sigma2_row"]
    sigma2_col = caputo_estimatives.loc[dataset_name, "sigma2_col"]

    models = {
        "IFCM": (
            IFCM(n_clusters=N_CLUSTERS, m=M, n_init=N_INIT),
            IFCM(n_clusters=N_CLUSTERS, m=M, n_init=N_INIT),
        ),
        "KFCM_IV": (
            KFCM_IV(n_clusters=N_CLUSTERS, m=M, sigma2=sigma2_row, n_init=N_INIT),
            KFCM_IV(n_clusters=N_CLUSTERS, m=M, sigma2=sigma2_col, n_init=N_INIT),
        ),
    }

    for model_name, (model_row, model_col) in models.items():

        print(f"  -> {model_name}")

        results = run_clustering(
            model_row=model_row,
            model_col=model_col,
            synthetic_dataset=dataset,
            n_rep=N_REP,
            shape=SHAPE,
            n_jobs=N_JOBS,
            random_state=10,
        )

        output_file = RESULTS_DIR / dataset_name / f"{model_name}.txt"
        save_results(results=results, output_file=output_file)

print("\nTask completed.")
