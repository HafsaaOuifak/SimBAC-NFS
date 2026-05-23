#!/usr/bin/env python3
"""
SIMBAC-NFS — run on any of the 9 benchmark datasets.

Usage:
  python main.py --dataset yacht
  python main.py --dataset all
  python main.py --dataset concrete --nc 20 --tau 0.95 --T 5 --M 3
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.models.fcm_tsk import FCMTSKModel
from src.models.bagging_ensemble import BaggingFCMTSK, select_features_mi
from src.models.compression import GradNFSCompressor
from src.datasets.uci_loader import load_dataset
from src.datasets.nasa_battery import load_nasa_battery, get_nasa_feature_matrix, get_nasa_lobo_splits

# paper defaults per dataset
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

NRULES_GRID = [15, 20, 25]
N_INNER     = 3
MIN_RULES   = 3
LR          = 0.3
CI          = 0.99
ALPHA_GRID  = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
MAX_FEAT    = 7
SEED        = 42

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae  = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
r2   = lambda a, b: float(1 - np.sum((np.asarray(a) - np.asarray(b)) ** 2)
                          / (np.sum((np.asarray(a) - np.mean(a)) ** 2) + 1e-12))


def load_ds(name):
    if name == "nasa":
        df = load_nasa_battery(verbose=False)
        X, y, feat = get_nasa_feature_matrix(df)
        folds = [(tr, te) for tr, te, _ in get_nasa_lobo_splits(df)]
        return X, y, feat, folds, df
    X, y, feat, _ = load_dataset(name, verbose=False)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    return X, y, feat, folds, None


def fit_pool(X_tr, y_tr, nc, T, M, boot_ratio):
    y_res = y_tr.copy().astype(np.float64)
    pool = []
    for t in range(T):
        bag = BaggingFCMTSK(n_estimators=M, n_rules=nc, min_rules=MIN_RULES,
                             bootstrap_ratio=boot_ratio,
                             base_params={"mf_type": "gaussian"},
                             random_state=SEED + t)
        bag.fit(X_tr, y_res)
        round_preds = np.mean([e.predict(X_tr) for e in bag.estimators_], axis=0)
        y_res = y_res - LR * round_preds
        pool.extend(bag.estimators_)
    return pool


def inner_cv_select(X_tr, y_tr, tau, T, M, boot_ratio, inner_splits):
    best_nc, best_score = NRULES_GRID[0], np.inf
    for nc in NRULES_GRID:
        scores = []
        for i_tr, i_te in inner_splits:
            try:
                if len(y_tr[i_tr]) < nc * T or len(y_tr[i_te]) < 2:
                    scores.append(np.inf); continue
                pool = fit_pool(X_tr[i_tr], y_tr[i_tr], nc, T, M, boot_ratio)
                comp = GradNFSCompressor(tau=tau, similarity_method="combined",
                                         weight_by_performance=True,
                                         cumulative_importance=CI, min_rules=MIN_RULES,
                                         refit_consequents=True, tune_refit_alpha=True,
                                         refit_alpha_grid=ALPHA_GRID)
                cm = comp.compress(pool, X_tr[i_tr], y_tr[i_tr])
                scores.append(rmse(y_tr[i_te], cm.predict(X_tr[i_te])))
            except Exception:
                scores.append(np.inf)
        s = float(np.nanmean(scores)) if scores else np.inf
        if s < best_score:
            best_score, best_nc = s, nc
    return best_nc


def run_dataset(name, nc_override=None, tau_override=None,
                T_override=None, M_override=None):
    cfg = DATASET_CONFIGS[name]
    tau = tau_override or cfg["tau"]
    T   = T_override   or cfg["T"]
    M   = M_override   or cfg["M"]
    boot_ratio = cfg["bootstrap_ratio"]

    X, y, feat, folds, df_nasa = load_ds(name)

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr_s, sel_names, sel_idx = select_features_mi(
            X_tr, y_tr, feat, max_features=MAX_FEAT, random_state=SEED)
        X_te_s = X_te[:, sel_idx]

        if df_nasa is not None:
            # nasa: leave-one-battery-out inner splits
            bids = df_nasa["battery_id"].values[tr_idx]
            inner_splits = []
            for bid in pd.unique(bids):
                te_l = np.where(bids == bid)[0]
                tr_l = np.where(bids != bid)[0]
                if len(tr_l) > 0 and len(te_l) > 0:
                    inner_splits.append((tr_l, te_l))
        else:
            inner_splits = list(KFold(n_splits=N_INNER, shuffle=True,
                                      random_state=SEED).split(X_tr_s))

        best_nc = nc_override or inner_cv_select(
            X_tr_s, y_tr, tau, T, M, boot_ratio, inner_splits)

        pool = fit_pool(X_tr_s, y_tr, best_nc, T, M, boot_ratio)
        comp = GradNFSCompressor(tau=tau, similarity_method="combined",
                                  weight_by_performance=True,
                                  cumulative_importance=CI, min_rules=MIN_RULES,
                                  refit_consequents=True, tune_refit_alpha=True,
                                  refit_alpha_grid=ALPHA_GRID)
        compressed = comp.compress(pool, X_tr_s, y_tr)
        y_pred = compressed.predict(X_te_s)

        row = {"fold": fold_i + 1, "nc": best_nc,
               "n_rules": len(compressed.centers_),
               "rmse": rmse(y_te, y_pred),
               "mae":  mae(y_te, y_pred),
               "r2":   r2(y_te, y_pred)}
        rows.append(row)
        print(f"  Fold {fold_i+1}: RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}"
              f"  R²={row['r2']:.4f}  rules={row['n_rules']}  nc={best_nc}")

    df_res = pd.DataFrame(rows)
    m = df_res[["rmse", "mae", "r2"]].mean()
    print(f"\n  Mean: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
    return df_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yacht")
    parser.add_argument("--nc",  type=int,   default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--T",   type=int,   default=None)
    parser.add_argument("--M",   type=int,   default=None)
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    if args.dataset != "all" and args.dataset not in DATASET_CONFIGS:
        parser.error(f"Unknown dataset. Choose from: {', '.join(DATASET_CONFIGS)}")

    os.makedirs("results/csv", exist_ok=True)
    for ds in datasets:
        print(f"\n{'='*55}\nDataset: {ds}\n{'='*55}")
        df_res = run_dataset(ds, nc_override=args.nc, tau_override=args.tau,
                             T_override=args.T, M_override=args.M)
        df_res.to_csv(f"results/csv/{ds}_results.csv", index=False)


if __name__ == "__main__":
    main()
