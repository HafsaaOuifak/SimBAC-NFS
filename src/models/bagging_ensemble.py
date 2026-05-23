"""
Bagging ensemble of FCM-TSK models.

Each base learner initializes FCM with the shared MAX_RULES budget and then
self-prunes to its data-adapted rule count via support-based pruning inside
its own bootstrap training fold. The ensemble therefore contains base learners
with potentially different (pruned) rule counts, which is natural under the
max-rule-then-prune strategy.
"""

import numpy as np
from typing import List, Optional, Tuple

from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import resample

from .fcm_tsk import FCMTSKModel


def select_features_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    max_features: int = 7,
    random_state: int = 42,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Mutual Information feature selection — must be called on training fold only.

    Returns
    -------
    X_selected : (n_train, k)
    selected_names : list of str
    selected_indices : np.ndarray of int
    """
    k = min(max_features, X.shape[1])
    mi_scores = mutual_info_regression(X, y, random_state=random_state)
    selected_indices = np.argsort(mi_scores)[::-1][:k]
    selected_indices = np.sort(selected_indices)
    selected_names = [feature_names[i] for i in selected_indices]
    return X[:, selected_indices], selected_names, selected_indices


def _fit_one_estimator(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    n_rules: int,
    min_rules: int,
    min_rule_support,
    base_params: dict,
    seed_m: int,
) -> Tuple["FCMTSKModel", np.ndarray]:
    """Fit a single bootstrap FCM-TSK model (picklable for joblib)."""
    n_samples = len(X)
    X_boot, y_boot = resample(X, y, n_samples=n_boot, replace=True, random_state=seed_m)
    model = FCMTSKModel(
        n_rules=n_rules,
        min_rules=min_rules,
        min_rule_support=min_rule_support,
        random_state=seed_m,
        **base_params,
    )
    model.fit(X_boot, y_boot)
    boot_idx = resample(
        np.arange(n_samples), n_samples=n_boot, replace=True, random_state=seed_m
    )
    return model, boot_idx


class BaggingFCMTSK:
    """
    Bootstrap aggregation of FCM-TSK models with support-based rule pruning.

    Each base learner is initialized with `n_rules` (= MAX_RULES) clusters and
    self-prunes via mean normalized firing strength on its bootstrap sample.
    The pruning floor `min_rules` prevents degenerate models on small samples.

    Parameters
    ----------
    n_estimators : int
        Number of base learners M.
    n_rules : int
        Maximum rule budget for FCM initialization (shared across all learners).
    min_rules : int
        Minimum rules to retain after pruning (per base learner).
    min_rule_support : float or None
        Per-rule mean firing-strength threshold; passed to FCMTSKModel.
        None → each model uses 0.5 / n_rules automatically.
    bootstrap_ratio : float
        Bootstrap sample size as a fraction of training set size.
    base_params : dict
        Additional kwargs forwarded to FCMTSKModel (e.g. ridge_alpha).
    n_jobs : int
        Parallel workers for fitting base learners (-1 = all cores).
    random_state : int
    """

    def __init__(
        self,
        n_estimators: int = 10,
        n_rules: int = 200,
        min_rules: int = 7,
        min_rule_support: float = None,
        bootstrap_ratio: float = 1.0,
        base_params: Optional[dict] = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators    = n_estimators
        self.n_rules         = n_rules
        self.min_rules       = min_rules
        self.min_rule_support = min_rule_support
        self.bootstrap_ratio = bootstrap_ratio
        self.base_params     = base_params or {}
        self.n_jobs          = n_jobs
        self.random_state    = random_state

        self.estimators_: List[FCMTSKModel] = []
        self.bootstrap_indices_: List[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingFCMTSK":
        """
        Train M base FCM-TSK models on bootstrap samples in parallel.

        Each base model: fit on bootstrap sample → self-prune → Ridge refit.
        All steps occur within that model's bootstrap training data only.
        """
        n_samples = len(X)
        n_boot    = max(1, int(np.ceil(self.bootstrap_ratio * n_samples)))

        rng   = np.random.RandomState(self.random_state)
        seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(self.n_estimators)]

        results = [
            _fit_one_estimator(
                X, y, n_boot,
                self.n_rules, self.min_rules, self.min_rule_support,
                self.base_params, seed_m,
            )
            for seed_m in seeds
        ]

        self.estimators_        = [r[0] for r in results]
        self.bootstrap_indices_ = [r[1] for r in results]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction: average of all base learner outputs."""
        predictions = np.stack([m.predict(X) for m in self.estimators_], axis=1)
        return predictions.mean(axis=1)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Return individual predictions: (n_samples, M)."""
        return np.stack([m.predict(X) for m in self.estimators_], axis=1)

    def get_all_rule_params(self) -> list:
        """Return list of rule parameter dicts from all base learners."""
        return [m.get_rule_params() for m in self.estimators_]

    def total_rules(self) -> int:
        """Total number of rules across all base learners (after pruning)."""
        return sum(m.n_rules for m in self.estimators_)

    def mean_rules(self) -> float:
        """Mean rules per base learner after pruning."""
        if not self.estimators_:
            return 0.0
        return float(np.mean([m.n_rules for m in self.estimators_]))

    def __len__(self):
        return len(self.estimators_)
