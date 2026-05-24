#!/usr/bin/env python3
"""
SIMBAC-NFS — run on any UCI benchmark dataset.

Usage:
  python main.py --dataset yacht
  python main.py --dataset all
  python main.py --dataset concrete --tau 0.95 --T 5 --M 3 --nc 20 --lr 0.3 --ci 0.99
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.models.pool import FCMTSKPool, select_features_mi
from src.models.compression import SIMBACCompressor
from src.datasets.data_loader import load_dataset, DATASETS

# paper defaults per dataset
CONFIGS = {
    "concrete":          {"T": 5, "M": 3, "tau": 0.95},
    "energy_efficiency": {"T": 5, "M": 3, "tau": 0.95},
    "ccpp":              {"T": 5, "M": 3, "tau": 0.95},
    "airfoil":           {"T": 5, "M": 3, "tau": 0.95},
    "yacht":             {"T": 5, "M": 3, "tau": 0.95},
    "gas_turbine":       {"T": 5, "M": 3, "tau": 0.95},
    "grid_stability":    {"T": 7, "M": 5, "tau": 0.99},
    "parkinsons":        {"T": 5, "M": 3, "tau": 0.95},
}

NRULES_GRID = [15, 20, 25]
MIN_RULES   = 3
MAX_FEAT    = 7
ALPHA_GRID  = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
SEED        = 42

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae  = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
r2   = lambda a, b: float(1 - np.sum((np.asarray(a) - np.asarray(b)) ** 2)
                          / (np.sum((np.asarray(a) - np.mean(a)) ** 2) + 1e-12))


def fit_pool(X_tr, y_tr, nc, T, M, lr):
    y_res = y_tr.copy().astype(float)
    pool = []
    for t in range(T):
        bag = FCMTSKPool(n_estimators=M, n_rules=nc, min_rules=MIN_RULES,
                         base_params={"mf_type": "gaussian"},
                         random_state=SEED + t)
        bag.fit(X_tr, y_res)
        y_res -= lr * np.mean([e.predict(X_tr) for e in bag.estimators_], axis=0)
        pool.extend(bag.estimators_)
    return pool


def inner_cv_select(X_tr, y_tr, tau, T, M, inner_splits, lr, ci):
    best_nc, best_rmse = NRULES_GRID[0], np.inf
    for nc in NRULES_GRID:
        scores = []
        for i_tr, i_te in inner_splits:
            try:
                pool = fit_pool(X_tr[i_tr], y_tr[i_tr], nc, T, M, lr)
                comp = SIMBACCompressor(tau=tau, weight_by_performance=True,
                                         cumulative_importance=ci, min_rules=MIN_RULES,
                                         refit_consequents=True, tune_refit_alpha=True,
                                         refit_alpha_grid=ALPHA_GRID)
                cm = comp.compress(pool, X_tr[i_tr], y_tr[i_tr])
                scores.append(rmse(y_tr[i_te], cm.predict(X_tr[i_te])))
            except Exception:
                scores.append(np.inf)
        s = float(np.nanmean(scores)) if scores else np.inf
        if s < best_rmse:
            best_rmse, best_nc = s, nc
    return best_nc


def run_dataset(name, nc=None, tau=None, T=None, M=None, lr=0.3, ci=0.99):
    cfg = CONFIGS[name]
    tau = tau or cfg["tau"]
    T   = T   or cfg["T"]
    M   = M   or cfg["M"]

    X, y, feat = load_dataset(name)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr_s, _, sel_idx = select_features_mi(X_tr, y_tr, feat,
                                                 max_features=MAX_FEAT,
                                                 random_state=SEED)
        X_te_s = X_te[:, sel_idx]

        inner_splits = list(KFold(n_splits=3, shuffle=True,
                                   random_state=SEED).split(X_tr_s))

        best_nc = nc or inner_cv_select(X_tr_s, y_tr, tau, T, M,
                                         inner_splits, lr, ci)

        pool = fit_pool(X_tr_s, y_tr, best_nc, T, M, lr)
        comp = SIMBACCompressor(tau=tau, weight_by_performance=True,
                                  cumulative_importance=ci, min_rules=MIN_RULES,
                                  refit_consequents=True, tune_refit_alpha=True,
                                  refit_alpha_grid=ALPHA_GRID)
        model = comp.compress(pool, X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        row = {"fold": fold_i + 1, "nc": best_nc,
               "n_rules": len(model.centers_),
               "rmse": rmse(y_te, y_pred),
               "mae":  mae(y_te, y_pred),
               "r2":   r2(y_te, y_pred)}
        rows.append(row)
        print(f"  Fold {fold_i+1}: RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}"
              f"  R²={row['r2']:.4f}  rules={row['n_rules']}  nc={best_nc}")

    df = pd.DataFrame(rows)
    m = df[["rmse", "mae", "r2"]].mean()
    print(f"\n  Mean: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run SIMBAC-NFS on a benchmark dataset")
    parser.add_argument("--dataset", default="yacht",
                        help=f"dataset name or 'all' — options: {list(CONFIGS)}")
    parser.add_argument("--nc",  type=int,   default=None, help="rule budget (skips inner CV)")
    parser.add_argument("--tau", type=float, default=None, help="rule merging threshold")
    parser.add_argument("--T",   type=int,   default=None, help="boosting rounds")
    parser.add_argument("--M",   type=int,   default=None, help="bags per round")
    parser.add_argument("--lr",  type=float, default=0.3,  help="learning rate (default 0.3)")
    parser.add_argument("--ci",  type=float, default=0.99, help="cumulative importance (default 0.99)")
    args = parser.parse_args()

    datasets = list(CONFIGS) if args.dataset == "all" else [args.dataset]
    if args.dataset != "all" and args.dataset not in CONFIGS:
        parser.error(f"Unknown dataset. Options: {list(CONFIGS)}")

    os.makedirs("results/csv", exist_ok=True)
    for ds in datasets:
        print(f"\n{'='*55}\nDataset: {ds}\n{'='*55}")
        df = run_dataset(ds, nc=args.nc, tau=args.tau, T=args.T,
                         M=args.M, lr=args.lr, ci=args.ci)
        df.to_csv(f"results/csv/{ds}_results.csv", index=False)


if __name__ == "__main__":
    main()
