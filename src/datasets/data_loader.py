import numpy as np
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


def load_dataset(name):
    """Fetch a UCI dataset by name. Returns (X, y, feature_names)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS)}")

    uci_id, target_col, drop_cols = DATASETS[name]
    repo = fetch_ucirepo(id=uci_id)

    X_df = repo.data.features
    y_df = repo.data.targets

    # drop unwanted columns from features if any
    X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors="ignore")

    # pick target column when there are multiple
    if target_col is not None:
        y_df = y_df[[target_col]]

    # drop rows with missing values
    import pandas as pd
    df = pd.concat([X_df, y_df], axis=1).dropna()
    X_df = df.iloc[:, :X_df.shape[1]]
    y_col = df.iloc[:, X_df.shape[1]:]

    X = X_df.select_dtypes(include=[np.number]).values.astype(float)
    y = y_col.values.flatten().astype(float)
    feature_names = X_df.select_dtypes(include=[np.number]).columns.tolist()

    return X, y, feature_names
