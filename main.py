#!/usr/bin/env python3
"""
SimBAC-NFS: main entry point.

Trains and evaluates SimBAC-NFS on one or all nine benchmark datasets using
the nested cross-validation protocol from the paper.

Usage
-----
  # Run on a single dataset (fast demo)
  python main.py --dataset yacht

  # Run on all nine datasets (full experiment, ~2-4 hours)
  python main.py --dataset all

  # Custom configuration
  python main.py --dataset concrete --nc 20 --tau 0.95 --T 5 --M 3

Datasets available
------------------
  nasa, concrete, energy_efficiency, ccpp, airfoil, yacht,
  gas_turbine, grid_stability, parkinsons

Output
------
  Prints RMSE, MAE, R² per outer fold and the mean across folds.
  Saves results to results/csv/<dataset>_main_results.csv.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.models.fcm_tsk import FCMTSKModel
from src.models.bagging_ensemble import BaggingFCMTSK, select_features_mi
from src.models.compression import GradNFSCompressor
from src.datasets.uci_loader import load_dataset
from src.datasets.nasa_battery import (
    load_nasa_battery, get_nasa_feature_matrix, get_nasa_lobo_splits,
)

# ── Default per-dataset configurations from the paper ─────────────────────────
DATASET_CONFIGS = {
    "nasa":              {"T": 5, "M": 3, "tau": 0.73, "bootstrap_ratio": 1.00},
    "concrete":          {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "energy_efficiency": {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "ccpp":              {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "airfoil":           {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "yacht":             {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "gas_turbine":       {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "grid_stability":    {"T": 7, "M": 5, "tau": 0.99, "bootstrap_ratio": 0.03},
    "parkinsons":        {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
}

NRULES_GRID = [15, 20, 25]   # inner-CV grid for n_c selection
N_INNER     = 3               # inner CV folds
MIN_RULES   = 3
LR          = 0.3             # boosting shrinkage
CI          = 0.99            # cumulative importance
ALPHA_GRID  = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
MAX_FEAT    = 7               # MI feature selection budget
SEED        = 42

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae  = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
r2   = lambda a, b: float(1 - np.sum((np.asarray(a) - np.asarray(b)) ** 2)
                              / (np.sum((np.asarray(a) - np.mean(a)) ** 2) + 1e-12))


def load_ds(ds_name):
    """Return (X, y, feature_names, outer_folds[, df])."""
    if ds_name == "nasa":
        df = load_nasa_battery(verbose=False)
        X, y, feat = get_nasa_feature_matrix(df)
        folds = [(tr, te) for tr, te, _ in get_nasa_lobo_splits(df)]
        return X, y, feat, folds, df
    X, y, feat, _ = load_dataset(ds_name, verbose=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    return X, y, feat, list(kf.split(X)), None


def fit_pool(X_tr, y_tr, nc, T, M, boot_ratio):
    """Build gradient-boosted pool of T*M FCM-TSK bags."""
    y_res = y_tr.copy().astype(np.float64)
    estimators = []
    for t in range(T):
        bag = BaggingFCMTSK(
            n_estimators=M, n_rules=nc, min_rules=MIN_RULES,
            bootstrap_ratio=boot_ratio,
            base_params={"mf_type": "gaussian"},
            random_state=SEED + t,
        )
        bag.fit(X_tr, y_res)
        round_preds = np.mean([e.predict(X_tr) for e in bag.estimators_], axis=0)
        y_res = y_res - LR * round_preds
        estimators.extend(bag.estimators_)
    return estimators


def inner_cv_select(X_tr, y_tr, tau, T, M, boot_ratio, inner_splits):
    """Pick best n_c by inner CV (no test leakage)."""
    best_nc, best_score = NRULES_GRID[0], np.inf
    for nc in NRULES_GRID:
        scores = []
        for i_tr, i_te in inner_splits:
            try:
                if len(y_tr[i_tr]) < nc * T or len(y_tr[i_te]) < 2:
                    scores.append(np.inf)
                    continue
                pool = fit_pool(X_tr[i_tr], y_tr[i_tr], nc, T, M, boot_ratio)
                comp = GradNFSCompressor(
                    tau=tau, similarity_method="combined",
                    weight_by_performance=True, cumulative_importance=CI,
                    min_cluster_importance=0.0, min_rules=MIN_RULES,
                    refit_consequents=True, tune_refit_alpha=True,
                    refit_alpha_grid=ALPHA_GRID,
                )
                cm = comp.compress(pool, X_tr[i_tr], y_tr[i_tr])
                scores.append(rmse(y_tr[i_te], cm.predict(X_tr[i_te])))
            except Exception:
                scores.append(np.inf)
        mean_score = float(np.nanmean(scores)) if scores else np.inf
        if mean_score < best_score:
            best_score, best_nc = mean_score, nc
    return best_nc


def inner_splits_for_nasa(outer_tr_idx, df):
    battery_ids = df["battery_id"].values[outer_tr_idx]
    unique = pd.unique(battery_ids)
    splits = []
    for bid in unique:
        te_local = np.where(battery_ids == bid)[0]
        tr_local = np.where(battery_ids != bid)[0]
        splits.append((outer_tr_idx[tr_local], outer_tr_idx[te_local]))
    return splits


def run_dataset(ds_name, nc_override=None, tau_override=None,
                T_override=None, M_override=None):
    cfg = DATASET_CONFIGS[ds_name]
    tau = tau_override if tau_override is not None else cfg["tau"]
    T   = T_override   if T_override   is not None else cfg["T"]
    M   = M_override   if M_override   is not None else cfg["M"]
    boot_ratio = cfg["bootstrap_ratio"]

    result = load_ds(ds_name)
    X, y, feat, folds = result[0], result[1], result[2], result[3]
    df_nasa = result[4] if len(result) > 4 else None

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # MI feature selection on training fold
        X_tr_s, sel_names, sel_idx = select_features_mi(
            X_tr, y_tr, feat, max_features=MAX_FEAT, random_state=SEED
        )
        X_te_s = X_te[:, sel_idx]

        # Inner CV splits
        if df_nasa is not None:
            inner_splits = inner_splits_for_nasa(tr_idx, df_nasa)
            # Re-index to local training fold
            inner_splits_local = []
            for i_tr_g, i_te_g in inner_splits:
                tr_local = np.array([np.where(tr_idx == i)[0][0]
                                     for i in i_tr_g if i in tr_idx])
                te_local = np.array([np.where(tr_idx == i)[0][0]
                                     for i in i_te_g if i in tr_idx])
                if len(tr_local) > 0 and len(te_local) > 0:
                    inner_splits_local.append((tr_local, te_local))
        else:
            kf = KFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
            inner_splits_local = list(kf.split(X_tr_s))

        # Select n_c
        if nc_override is not None:
            best_nc = nc_override
        else:
            best_nc = inner_cv_select(
                X_tr_s, y_tr, tau, T, M, boot_ratio, inner_splits_local
            )

        # Train final pool and compress
        pool = fit_pool(X_tr_s, y_tr, best_nc, T, M, boot_ratio)
        comp = GradNFSCompressor(
            tau=tau, similarity_method="combined",
            weight_by_performance=True, cumulative_importance=CI,
            min_cluster_importance=0.0, min_rules=MIN_RULES,
            refit_consequents=True, tune_refit_alpha=True,
            refit_alpha_grid=ALPHA_GRID,
        )
        compressed = comp.compress(pool, X_tr_s, y_tr)
        y_pred = compressed.predict(X_te_s)
        n_rules = len(compressed.centers_)

        row = {
            "fold": fold_i + 1,
            "nc": best_nc,
            "n_rules": n_rules,
            "rmse": rmse(y_te, y_pred),
            "mae":  mae(y_te, y_pred),
            "r2":   r2(y_te, y_pred),
        }
        rows.append(row)
        print(f"  Fold {fold_i + 1}: RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}"
              f"  R²={row['r2']:.4f}  n_rules={n_rules}  nc={best_nc}")

    df_res = pd.DataFrame(rows)
    means = df_res[["rmse", "mae", "r2"]].mean()
    print(f"\n  Mean: RMSE={means['rmse']:.4f}  MAE={means['mae']:.4f}"
          f"  R²={means['r2']:.4f}")
    return df_res


def main():
    parser = argparse.ArgumentParser(description="SimBAC-NFS experiment runner")
    parser.add_argument("--dataset", default="yacht",
                        help="Dataset name or 'all' (default: yacht)")
    parser.add_argument("--nc",  type=int, default=None,
                        help="Override n_c (skip inner CV selection)")
    parser.add_argument("--tau", type=float, default=None,
                        help="Override similarity threshold tau")
    parser.add_argument("--T",   type=int, default=None,
                        help="Override number of boosting rounds T")
    parser.add_argument("--M",   type=int, default=None,
                        help="Override number of bags per round M")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    if args.dataset != "all" and args.dataset not in DATASET_CONFIGS:
        parser.error(f"Unknown dataset '{args.dataset}'. "
                     f"Choose from: {', '.join(DATASET_CONFIGS)}")

    os.makedirs("results/csv", exist_ok=True)

    all_results = {}
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  "
              f"(tau={args.tau or DATASET_CONFIGS[ds]['tau']}, "
              f"T={args.T or DATASET_CONFIGS[ds]['T']}, "
              f"M={args.M or DATASET_CONFIGS[ds]['M']})")
        print("="*60)
        df_res = run_dataset(ds, nc_override=args.nc, tau_override=args.tau,
                             T_override=args.T, M_override=args.M)
        out_path = f"results/csv/{ds}_main_results.csv"
        df_res.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        all_results[ds] = df_res

    if len(datasets) > 1:
        print(f"\n{'='*60}")
        print("Summary across all datasets:")
        for ds, df_res in all_results.items():
            m = df_res[["rmse", "mae", "r2"]].mean()
            print(f"  {ds:<22}  RMSE={m['rmse']:.4f}  "
                  f"MAE={m['mae']:.4f}  R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()
