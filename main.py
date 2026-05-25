#!/usr/bin/env python3
"""
SIMBAC-NFS — run on any UCI benchmark dataset.

Usage:
  python main.py --dataset yacht
  python main.py --dataset all
  python main.py --dataset concrete --tau 0.95 --T 5 --M 3 --nc 20 --lr 0.3 --ci 0.99

All hyperparameters default to the values used in the paper.
Override any of them on the command line.
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

# Per-dataset architecture defaults from the paper
# bootstrap_ratio=0.03 for grid_stability prevents identical clusters on the
# 10 000-sample dataset (full bootstrap collapses to 1 rule and poor RMSE).
CONFIGS = {
    "concrete":          {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "energy_efficiency": {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "ccpp":              {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "airfoil":           {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "yacht":             {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "gas_turbine":       {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
    "grid_stability":    {"T": 7, "M": 5, "tau": 0.99, "bootstrap_ratio": 0.03},
    "parkinsons":        {"T": 5, "M": 3, "tau": 0.95, "bootstrap_ratio": 1.00},
}

# Paper-fixed defaults (exposed as CLI flags so they can be overridden)
PAPER_DEFAULTS = dict(
    lr                   = 0.3,
    ci                   = 0.99,
    min_cluster_importance = 0.0,
    min_rules            = 3,
    max_feat             = 7,
    seed                 = 42,
    nrules_grid          = [15, 20, 25],
    alpha_grid           = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4],
)

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae  = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
r2   = lambda a, b: float(1 - np.sum((np.asarray(a) - np.asarray(b)) ** 2)
                          / (np.sum((np.asarray(a) - np.mean(a)) ** 2) + 1e-12))


def fit_pool(X_tr, y_tr, nc, T, M, lr,
             bootstrap_ratio=1.0, min_rules=3, seed=42):
    y_res = y_tr.copy().astype(float)
    pool  = []
    for t in range(T):
        bag = FCMTSKPool(
            n_estimators=M, n_rules=nc, min_rules=min_rules,
            bootstrap_ratio=bootstrap_ratio,
            base_params={"mf_type": "gaussian"},
            random_state=seed + t,
        )
        bag.fit(X_tr, y_res)
        y_res -= lr * np.mean([e.predict(X_tr) for e in bag.estimators_], axis=0)
        pool.extend(bag.estimators_)
    return pool


def inner_cv_select(X_tr, y_tr, tau, T, M, inner_splits, lr, ci,
                    bootstrap_ratio=1.0, min_cluster_importance=0.0,
                    nrules_grid=None, min_rules=3, alpha_grid=None, seed=42):
    if nrules_grid is None:
        nrules_grid = PAPER_DEFAULTS["nrules_grid"]
    if alpha_grid is None:
        alpha_grid = PAPER_DEFAULTS["alpha_grid"]

    best_nc, best_score = nrules_grid[0], np.inf
    for nc in nrules_grid:
        scores = []
        for i_tr, i_te in inner_splits:
            try:
                pool = fit_pool(X_tr[i_tr], y_tr[i_tr], nc, T, M, lr,
                                bootstrap_ratio=bootstrap_ratio,
                                min_rules=min_rules, seed=seed)
                comp = SIMBACCompressor(
                    tau=tau, weight_by_performance=True,
                    cumulative_importance=ci,
                    min_cluster_importance=min_cluster_importance,
                    min_rules=min_rules,
                    refit_consequents=True, tune_refit_alpha=True,
                    refit_alpha_grid=alpha_grid,
                )
                cm = comp.compress(pool, X_tr[i_tr], y_tr[i_tr])
                scores.append(rmse(y_tr[i_te], cm.predict(X_tr[i_te])))
            except Exception:
                scores.append(np.inf)
        s = float(np.nanmean(scores)) if scores else np.inf
        if s < best_score:
            best_score, best_nc = s, nc
    return best_nc


def run_dataset(name, nc=None, tau=None, T=None, M=None, bootstrap_ratio=None,
                lr=0.3, ci=0.99, min_cluster_importance=0.0, min_rules=3,
                max_feat=7, seed=42, nrules_grid=None, alpha_grid=None):
    if nrules_grid is None:
        nrules_grid = PAPER_DEFAULTS["nrules_grid"]
    if alpha_grid is None:
        alpha_grid = PAPER_DEFAULTS["alpha_grid"]

    cfg = CONFIGS[name]
    tau              = tau              or cfg["tau"]
    T                = T               or cfg["T"]
    M                = M               or cfg["M"]
    bootstrap_ratio  = bootstrap_ratio or cfg["bootstrap_ratio"]

    X, y, feat = load_dataset(name)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr_s, _, sel_idx = select_features_mi(
            X_tr, y_tr, feat, max_features=max_feat, random_state=seed,
        )
        X_te_s = X_te[:, sel_idx]

        inner_splits = list(
            KFold(n_splits=3, shuffle=True, random_state=seed).split(X_tr_s)
        )

        best_nc = nc or inner_cv_select(
            X_tr_s, y_tr, tau, T, M, inner_splits, lr, ci,
            bootstrap_ratio=bootstrap_ratio,
            min_cluster_importance=min_cluster_importance,
            nrules_grid=nrules_grid, min_rules=min_rules,
            alpha_grid=alpha_grid, seed=seed,
        )

        pool = fit_pool(X_tr_s, y_tr, best_nc, T, M, lr,
                        bootstrap_ratio=bootstrap_ratio,
                        min_rules=min_rules, seed=seed)
        comp = SIMBACCompressor(
            tau=tau, weight_by_performance=True,
            cumulative_importance=ci,
            min_cluster_importance=min_cluster_importance,
            min_rules=min_rules,
            refit_consequents=True, tune_refit_alpha=True,
            refit_alpha_grid=alpha_grid,
        )
        model = comp.compress(pool, X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        row = {
            "fold":    fold_i + 1,
            "nc":      best_nc,
            "n_rules": len(model.centers_),
            "rmse":    rmse(y_te, y_pred),
            "mae":     mae(y_te, y_pred),
            "r2":      r2(y_te, y_pred),
        }
        rows.append(row)
        print(f"  Fold {fold_i+1}: RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}"
              f"  R²={row['r2']:.4f}  rules={row['n_rules']}  nc={best_nc}")

    df = pd.DataFrame(rows)
    m = df[["rmse", "mae", "r2"]].mean()
    print(f"\n  Mean: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
    return df


def main():
    p = argparse.ArgumentParser(
        description="Run SIMBAC-NFS on a benchmark dataset. "
                    "All flags default to the values used in the paper."
    )
    p.add_argument("--dataset", default="yacht",
                   help=f"dataset name or 'all' — options: {list(CONFIGS)}")
    # architecture (per-dataset defaults in CONFIGS)
    p.add_argument("--T",   type=int,   default=None,
                   help="boosting rounds (paper: 5, except grid_stability=7)")
    p.add_argument("--M",   type=int,   default=None,
                   help="FCM-TSK bags per round (paper: 3, except grid_stability=5)")
    p.add_argument("--tau", type=float, default=None,
                   help="rule-merging threshold (paper: 0.95, except grid_stability=0.99)")
    p.add_argument("--nc",  type=int,   default=None,
                   help="FCM cluster budget — skips inner CV when set")
    p.add_argument("--bootstrap_ratio", type=float, default=None,
                   help="bootstrap sample fraction (paper: 1.0, except grid_stability=0.03)")
    # training
    p.add_argument("--lr",                   type=float, default=0.3,
                   help="boosting learning rate (paper: 0.3)")
    p.add_argument("--ci",                   type=float, default=0.99,
                   help="cumulative-importance cutoff (paper: 0.99)")
    p.add_argument("--min_cluster_importance", type=float, default=0.0,
                   help="per-cluster importance floor before CI cutoff (paper: 0.0)")
    p.add_argument("--min_rules",            type=int,   default=3,
                   help="minimum rules after compression (paper: 3)")
    p.add_argument("--max_feat",             type=int,   default=7,
                   help="MI feature selection cap (paper: 7)")
    p.add_argument("--seed",                 type=int,   default=42,
                   help="random seed (paper: 42)")
    p.add_argument("--nrules_grid",          type=int,   nargs="+",
                   default=[15, 20, 25],
                   help="inner-CV grid for nc (paper: 15 20 25)")
    p.add_argument("--alpha_grid",           type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4],
                   help="Ridge alpha search grid (paper: 1e-4 … 1e4)")
    args = p.parse_args()

    datasets = list(CONFIGS) if args.dataset == "all" else [args.dataset]
    if args.dataset != "all" and args.dataset not in CONFIGS:
        p.error(f"Unknown dataset. Options: {list(CONFIGS)}")

    os.makedirs("results/csv", exist_ok=True)
    for ds in datasets:
        print(f"\n{'='*55}\nDataset: {ds}\n{'='*55}")
        df = run_dataset(
            ds,
            nc=args.nc, tau=args.tau, T=args.T, M=args.M,
            bootstrap_ratio=args.bootstrap_ratio,
            lr=args.lr, ci=args.ci,
            min_cluster_importance=args.min_cluster_importance,
            min_rules=args.min_rules, max_feat=args.max_feat,
            seed=args.seed, nrules_grid=args.nrules_grid,
            alpha_grid=args.alpha_grid,
        )
        df.to_csv(f"results/csv/{ds}_results.csv", index=False)


if __name__ == "__main__":
    main()
