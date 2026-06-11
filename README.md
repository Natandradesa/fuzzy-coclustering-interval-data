
# Fuzzy Co-Clustering and Kernel Fuzzy Co-Clustering for Interval-Valued Data

This repository contains the source code, datasets, and experimental results associated with the paper:

> **Fuzzy Co-Clustering and Kernel Fuzzy Co-Clustering for Interval-Valued Data**
>
> José Nataniel Andrade de Sá, Marcelo Rodrigo Portela Ferreira, Francisco de Assis Tenório de Carvalho
>
> Accepted at the IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2026), held as part of the IEEE World Congress on Computational Intelligence (WCCI 2026).

> **Note**
>
> This repository accompanies an accepted conference paper. The final bibliographic information will be updated once the proceedings become available.

## Overview

This repository contains implementations of fuzzy clustering and fuzzy co-clustering methods for interval-valued data, including the proposed kernel fuzzy co-clustering approaches introduced in the paper.

### Clustering Baselines

- IFCM — Interval Fuzzy C-Means
- KFCM-IV — Kernel Interval Fuzzy C-Means

### Co-Clustering Methods

- IFDK — Interval Fuzzy Double K-Means
- IKFDK-O — Interval Kernel Fuzzy Double K-Means in the Original Space
- IKFDK-K — Interval Kernel Fuzzy Double K-Means in the Kernel Space

The repository also includes all scripts required to reproduce the experiments reported in the paper.

---

## Repository Organization

- `algorithms/`: implementations of the clustering and co-clustering methods.
- `benchmark_application/`: scripts for the real-data experiments.
- `interval_datasets/`: interval-valued benchmark datasets used in the experiments.
- `metrics/`: code for the evaluation measures.
- `notebooks/`: Jupyter Notebook code for analyzing the results.
- `synthetic_application/`: scripts for the synthetic-data experiments.
- `utils/`: auxiliary functions shared across the project.

The repository includes all code required to reproduce the experimental results reported in the paper.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Natandradesa/fuzzy-coclustering-interval-data.git
cd fuzzy-coclustering-interval-data
````

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Synthetic Interval-Valued Data Experiments

### Step 1: Compute Kernel Width Estimates

```bash
python synthetic_application/compute_caputo_estimatives.py
```

### Step 2: Run Clustering Experiments

```bash
python synthetic_application/run_clustering.py
```

### Step 3: Run Co-Clustering Experiments

```bash
python synthetic_application/run_coclustering.py
```

Results are stored in:

```text
synthetic_application/results/
```

---

## Real Interval-Valued Data Experiments

The real-data benchmark uses the datasets available in:

```text
interval_datasets/
```

### Step 1: Compute Kernel Width Estimates

```bash
python benchmark_application/compute/caputo_estimatives.py
```

### Step 2: Run the Benchmark

```bash
python benchmark_application/run_real_benchmark.py
```

Results are stored in:

```text
benchmark_application/results/
```

---

## Datasets

The repository includes several interval-valued benchmark datasets:

* Abalone
* Car Models
* Fish
* Fungi
* Horses
* Temperature

All datasets are located in:

```text
interval_datasets/
```

---

## Reproducibility

All experiments reported in the paper can be reproduced using the scripts provided in:

* `benchmark_application/`
* `synthetic_application/`

The implementations include:

* Dataset generation procedures
* Kernel width estimation
* Evaluation metrics
* Benchmark execution scripts

All random seeds and experimental settings used in the paper are available in the repository.

---

## Dependencies

Main dependencies:

* NumPy
* Pandas
* SciPy
* Scikit-learn
* Joblib

More details about the dependencies are available in:

```text
requirements.txt
```

---

## References

#### Interval Fuzzy C-Means (IFCM)

De Carvalho, F. A. T. (2007). *Fuzzy c-means clustering methods for symbolic interval data*. Pattern Recognition Letters, 28(4), 423–437.

#### Kernel Interval Fuzzy C-Means (KFCM-IV)

Pimentel, B. A., Costa, A. F. B. F., & Souza, R. M. C. R. (2011). *Kernel-based fuzzy clustering of interval data*. In **2011 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2011)** (pp. 497–501). IEEE.

#### Kernel Width Estimation

Caputo, B., Sim, K., Furesjö, F., & Smola, A. (2002). *Appearance-based object recognition using SVMs: Which kernel should I use?* NIPS Workshop on Statistical Methods for Computational Experiments in Visual Processing and Computer Vision.

---

## Citation

The complete citation information will be added after publication of the conference proceedings. If you use this repository in your research, please consider citing the paper after its publication.

```
```
