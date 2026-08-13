"""Data loading and perturbation utilities for the UCI grid stability dataset.

The dataset (Electrical Grid Stability Simulated Data, UCI ML Repository id=471)
has 10,000 rows, 12 predictive features (tau1-4, p1-4, g1-4), a continuous
target `stab` and a binary label `stabf` (stable/unstable). We use `stabf`.

Compute nodes on Perlmutter may lack internet access, so this module
only reads a local CSV. Use scripts/download_data.py on a login node first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [f"tau{i}" for i in range(1, 5)] + \
           [f"p{i}" for i in range(1, 5)] + \
           [f"g{i}" for i in range(1, 5)]
LABEL = "stabf"


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler | None


def load_dataset(csv_path: str,
                 n_samples: int | None,
                 test_size: float,
                 seed: int,
                 stratify: bool = True,
                 standardize: bool = True) -> DataSplit:
    df = pd.read_csv(csv_path)
    missing = set(FEATURES + [LABEL]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = (df[LABEL].astype(str).str.strip() == "unstable").astype(np.int64).to_numpy()

    # Stratified subsampling to a fixed-size subset (e.g. the 2,000-sample
    # DCAS setting), preserving class balance. Subset selection uses a fixed
    # seed (0) independent of the run seed so that every seed sees the same
    # data subset and only weight init / shuffling / noise draws vary.
    if n_samples is not None and n_samples < len(X):
        X, _, y, _ = train_test_split(
            X, y,
            train_size=n_samples,
            stratify=y if stratify else None,
            random_state=0,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=seed,
    )

    scaler = None
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return DataSplit(X_train, X_test, y_train, y_test, scaler)


# ---------------------------------------------------------------------------
# Input-level noise models (applied to standardized test features)
# ---------------------------------------------------------------------------

def apply_gaussian_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return X + rng.normal(0.0, sigma, size=X.shape)


def apply_uniform_noise(X: np.ndarray, width: float, rng: np.random.Generator) -> np.ndarray:
    return X + rng.uniform(-width, width, size=X.shape)


def apply_dropout_noise(X: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(X.shape) >= p
    return X * mask


NOISE_FNS = {
    "gaussian": apply_gaussian_noise,
    "uniform": apply_uniform_noise,
    "dropout": apply_dropout_noise,
}
