import io
import urllib.request
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# dataset name → (uci_id, target_column, columns_to_drop)
DATASETS = {
    "yacht":             (243, None,          []),
    "concrete":          (165, None,          []),
    "energy_efficiency": (242, "Y1",          []),
    "ccpp":              (294, None,          []),
    "airfoil":           (291, None,          []),
    "gas_turbine":       (551, "NOX",         ["CO"]),
    "grid_stability":    (471, "stab",        ["stabf"]),
    "parkinsons":        (188, "motor_UPDRS", ["total_UPDRS"]),
}

# Datasets that ucimlrepo cannot fetch — loaded via direct URL instead
_DIRECT = {
    "yacht": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        "cols": ["lcb", "cp", "l_d", "b_t", "l_b", "fn", "residuary_resistance"],
        "target": "residuary_resistance",
    },
}


def _load_yacht():
    cfg = _DIRECT["yacht"]
    with urllib.request.urlopen(cfg["url"]) as resp:
        raw = resp.read().decode()
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=cfg["cols"])
    df = df.dropna()
    feature_names = cfg["cols"][:-1]
    X = df[feature_names].values.astype(float)
    y = df[cfg["target"]].values.astype(float)
    return X, y, feature_names


def load_dataset(name):
    """Fetch a UCI dataset by name. Returns (X, y, feature_names)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS)}")

    if name in _DIRECT:
        return _load_yacht()

    uci_id, target_col, drop_cols = DATASETS[name]
    repo = fetch_ucirepo(id=uci_id)

    X_df = repo.data.features
    y_df = repo.data.targets

    # some UCI datasets return targets=None and include target col in features
    if y_df is None or (target_col is not None and target_col not in (y_df.columns if y_df is not None else [])):
        if target_col is not None and target_col in X_df.columns:
            y_df = X_df[[target_col]]
            X_df = X_df.drop(columns=[target_col])

    # drop unwanted columns from features
    X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors="ignore")

    # pick target column when there are multiple
    if target_col is not None and y_df is not None:
        if target_col in y_df.columns:
            y_df = y_df[[target_col]]

    df = pd.concat([X_df, y_df], axis=1).dropna()
    X_df = df.iloc[:, :X_df.shape[1]]
    y_col = df.iloc[:, X_df.shape[1]:]

    X = X_df.select_dtypes(include=[np.number]).values.astype(float)
    y = y_col.values.flatten().astype(float)
    feature_names = X_df.select_dtypes(include=[np.number]).columns.tolist()

    return X, y, feature_names
