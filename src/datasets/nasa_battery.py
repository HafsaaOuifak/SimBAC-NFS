"""
NASA Li-Ion Battery Aging Dataset preprocessor.

Extracts per-discharge-cycle features and computes RUL (Remaining Useful Life)
from degradation trajectories. EOL is defined as capacity dropping below 80% of
the initial nominal capacity, a common battery end-of-life convention.

Batteries: B0005, B0006, B0007, B0018
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import json
from datetime import datetime


# Path constants relative to this file's project root
_RAW_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Datasets", "NASA", "raw"
)
_PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Datasets", "NASA", "processed"
)
_BATTERY_FILES = {
    "B0005": "B0005.mat",
    "B0006": "B0006.mat",
    "B0007": "B0007.mat",
    "B0018": "B0018.mat",
}
EOL_CAPACITY_RATIO = 0.80  # 80% of initial capacity (IEEE/IEC standard battery EOL)

_EXPECTED_FEATURE_COLS = [
    "voltage_mean", "voltage_min", "voltage_drop",
    "current_mean", "temperature_mean", "temperature_max", "discharge_time",
]

# Cache schema version — bump this whenever preprocessing logic changes so that
# old parquet files are automatically invalidated.
_CACHE_SCHEMA_VERSION = 2


def _cache_is_valid(cache_path: str, manifest_path: str) -> bool:
    """
    Return True only when the cached parquet and its manifest match the current
    preprocessing configuration: EOL ratio, battery IDs, feature schema, and
    schema version.  Any mismatch triggers a full reprocess from raw .mat files.
    """
    if not os.path.exists(cache_path) or not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    # Version check
    if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
        print("  Cache invalid: schema_version mismatch — reprocessing.")
        return False

    # EOL ratio check
    stored_ratio = manifest.get("eol_capacity_ratio")
    if stored_ratio != EOL_CAPACITY_RATIO:
        print(f"  Cache invalid: EOL ratio changed "
              f"({stored_ratio} → {EOL_CAPACITY_RATIO}) — reprocessing.")
        return False

    # Battery ID check
    stored_ids = set(b["battery_id"] for b in manifest.get("batteries", []))
    expected_ids = set(_BATTERY_FILES.keys())
    if stored_ids != expected_ids:
        print(f"  Cache invalid: battery IDs changed "
              f"({stored_ids} → {expected_ids}) — reprocessing.")
        return False

    # Feature schema check (read parquet metadata without loading full file)
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(cache_path)
        cached_cols = set(schema.names)
        required_cols = set(_EXPECTED_FEATURE_COLS) | {"battery_id", "rul"}
        missing = required_cols - cached_cols
        if missing:
            print(f"  Cache invalid: missing columns {missing} — reprocessing.")
            return False
    except Exception:
        # pyarrow not available or unreadable; fall through to reprocess
        return False

    return True


def _parse_discharge_cycles(mat_data, battery_name: str) -> pd.DataFrame:
    """
    Extract features from each discharge cycle of a battery .mat file.

    For each discharge cycle we compute:
      - cycle_index        : sequential discharge cycle number (0-based)
      - capacity           : measured discharge capacity (Ah)
      - voltage_mean       : mean terminal voltage during discharge
      - voltage_min        : minimum terminal voltage
      - current_mean       : mean discharge current (absolute value)
      - temperature_mean   : mean cell temperature (°C)
      - temperature_max    : peak temperature
      - discharge_time     : total discharge duration (s)
      - voltage_drop       : initial minus final voltage
      - ambient_temperature: ambient temperature setting
    """
    key = battery_name
    data = mat_data[key][0, 0]
    cycles = data["cycle"][0]

    records = []
    discharge_idx = 0

    for cycle in cycles:
        ctype = str(cycle["type"][0])
        if ctype != "discharge":
            continue

        d = cycle["data"]
        V = d["Voltage_measured"][0, 0].flatten().astype(float)
        I = d["Current_measured"][0, 0].flatten().astype(float)
        T = d["Temperature_measured"][0, 0].flatten().astype(float)
        t = d["Time"][0, 0].flatten().astype(float)
        cap = float(d["Capacity"][0, 0].flatten()[0])
        amb = float(cycle["ambient_temperature"])

        records.append({
            "cycle_index": discharge_idx,
            "capacity": cap,
            "voltage_mean": float(np.mean(V)),
            "voltage_min": float(np.min(V)),
            "voltage_drop": float(V[0] - V[-1]) if len(V) > 1 else 0.0,
            "current_mean": float(np.mean(np.abs(I))),
            "temperature_mean": float(np.mean(T)),
            "temperature_max": float(np.max(T)),
            "discharge_time": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            "ambient_temperature": amb,
        })
        discharge_idx += 1

    return pd.DataFrame(records)


def _compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RUL for each discharge cycle.

    EOL is defined as the first cycle at which measured capacity drops strictly
    below EOL_CAPACITY_RATIO × C_initial (default 80%, the IEEE/IEC standard).
    RUL_n = N_EOL - n for cycle n < N_EOL, clipped to 0 for post-EOL cycles.

    If the battery is right-censored (capacity never crosses the threshold within
    the available data), `is_censored` is set to True and EOL is assigned to the
    last observed cycle.  With the 80% threshold all four NASA cells reach true
    EOL, so no censoring occurs in practice.
    """
    initial_cap = df["capacity"].iloc[0]
    eol_threshold = EOL_CAPACITY_RATIO * initial_cap

    eol_candidates = df.index[df["capacity"] < eol_threshold].tolist()
    if eol_candidates:
        eol_idx = int(eol_candidates[0])
        censored = False
    else:
        eol_idx = len(df) - 1  # right-censored: no true EOL in observed data
        censored = True

    df = df.copy()
    df["rul"] = eol_idx - df.index
    df["rul"] = df["rul"].clip(lower=0)
    df["eol_cycle"] = eol_idx
    df["is_censored"] = censored
    df["initial_capacity"] = initial_cap
    df["eol_threshold"] = eol_threshold
    df["capacity_fade"] = initial_cap - df["capacity"]
    df["soh"] = df["capacity"] / initial_cap

    return df


def load_nasa_battery(
    raw_dir: str = None,
    processed_dir: str = None,
    force_reprocess: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess the NASA Li-Ion Battery dataset.

    Returns a combined DataFrame from all four batteries (B0005, B0006, B0007, B0018)
    with per-cycle features and RUL target. Data is cached in the processed directory.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing .mat files. Defaults to Datasets/NASA/raw/.
    processed_dir : str, optional
        Directory for cached processed data. Defaults to Datasets/NASA/processed/.
    force_reprocess : bool
        If True, re-parse from .mat even if cache exists.
    verbose : bool
        Print loading summary.

    Returns
    -------
    pd.DataFrame
        Columns: battery_id, cycle_index, capacity, voltage_mean, voltage_min,
        voltage_drop, current_mean, temperature_mean, temperature_max,
        discharge_time, ambient_temperature, rul, eol_cycle, soh,
        initial_capacity, eol_threshold, capacity_fade.
    """
    raw_dir = raw_dir or os.path.abspath(_RAW_DIR)
    processed_dir = processed_dir or os.path.abspath(_PROCESSED_DIR)
    os.makedirs(processed_dir, exist_ok=True)

    cache_path = os.path.join(processed_dir, "nasa_battery_all.parquet")
    manifest_path = os.path.join(processed_dir, "nasa_manifest.json")

    if not force_reprocess and _cache_is_valid(cache_path, manifest_path):
        if verbose:
            print(f"Loading cached NASA battery data from {cache_path}")
        df = pd.read_parquet(cache_path)
        if verbose:
            _print_summary(df)
        return df

    all_dfs = []
    manifest = []

    for battery_id, filename in _BATTERY_FILES.items():
        mat_path = os.path.join(raw_dir, filename)
        if not os.path.exists(mat_path):
            print(f"  WARNING: {mat_path} not found — skipping {battery_id}")
            continue

        mat = scipy.io.loadmat(mat_path)
        df_cycles = _parse_discharge_cycles(mat, battery_id)
        df_cycles = _compute_rul(df_cycles)
        df_cycles.insert(0, "battery_id", battery_id)

        # Save per-battery processed file
        per_bat_path = os.path.join(processed_dir, f"{battery_id}_processed.parquet")
        df_cycles.to_parquet(per_bat_path, index=True)

        all_dfs.append(df_cycles)

        manifest.append({
            "battery_id": battery_id,
            "discharge_cycles": len(df_cycles),
            "initial_capacity_ah": round(float(df_cycles["initial_capacity"].iloc[0]), 4),
            "eol_threshold_ratio": EOL_CAPACITY_RATIO,
            "eol_threshold_ah": round(float(df_cycles["eol_threshold"].iloc[0]), 4),
            "eol_cycle": int(df_cycles["eol_cycle"].iloc[0]),
            "is_censored": bool(df_cycles["is_censored"].iloc[0]),
            "final_soh": round(float(df_cycles["soh"].iloc[-1]), 4),
            "rul_range": [int(df_cycles["rul"].min()), int(df_cycles["rul"].max())],
            "processed_at": datetime.now().isoformat(),
        })

        if verbose:
            print(f"  {battery_id}: {len(df_cycles)} discharge cycles, "
                  f"EOL@cycle {df_cycles['eol_cycle'].iloc[0]}, "
                  f"RUL range [{df_cycles['rul'].min()}, {df_cycles['rul'].max()}]")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_parquet(cache_path, index=False)

    # Save manifest with enough metadata for cache validation on future loads
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset": "NASA Li-Ion Battery",
            "schema_version": _CACHE_SCHEMA_VERSION,
            "eol_capacity_ratio": EOL_CAPACITY_RATIO,
            "feature_cols": _EXPECTED_FEATURE_COLS,
            "batteries": manifest,
        }, f, indent=2)

    if verbose:
        _print_summary(df_all)

    return df_all


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\nNASA Battery Dataset Summary")
    print(f"  Total records : {len(df)}")
    print(f"  Batteries     : {df['battery_id'].unique().tolist()}")
    print(f"  Features      : {[c for c in df.columns if c not in ['battery_id','rul','eol_cycle','initial_capacity','eol_threshold']]}")
    print(f"  RUL range     : [{df['rul'].min()}, {df['rul'].max()}]")
    print(f"  Missing values: {df.isnull().sum().sum()}")


def get_nasa_lobo_splits(df: pd.DataFrame = None) -> list:
    """
    Return Leave-One-Battery-Out (LOBO) cross-validation splits.

    Each fold holds out one of the four batteries as the test set and trains
    on the remaining three.  This is the appropriate CV strategy for the NASA
    dataset because the batteries are independent experimental units; mixing
    cycles from the same cell across train and test would constitute leakage.

    Returns
    -------
    list of (train_idx, test_idx, battery_id) tuples — one per battery.
    """
    if df is None:
        df = load_nasa_battery(verbose=False)

    battery_ids = df["battery_id"].values
    indices = np.arange(len(df))
    splits = []
    for bid in sorted(df["battery_id"].unique()):
        test_mask = battery_ids == bid
        splits.append((indices[~test_mask], indices[test_mask], bid))
    return splits


def get_nasa_feature_matrix(df: pd.DataFrame = None, target: str = "rul") -> tuple:
    """
    Return (X, y) arrays for regression, using 7 engineered cycle features.

    Features selected for interpretability (≤ 7):
      voltage_mean, voltage_min, voltage_drop, current_mean,
      temperature_mean, temperature_max, discharge_time
    (ambient_temperature excluded — constant at 24°C for these batteries)
    """
    if df is None:
        df = load_nasa_battery(verbose=False)

    feature_cols = [
        "voltage_mean",
        "voltage_min",
        "voltage_drop",
        "current_mean",
        "temperature_mean",
        "temperature_max",
        "discharge_time",
    ]
    X = df[feature_cols].values
    y = df[target].values
    return X, y, feature_cols
