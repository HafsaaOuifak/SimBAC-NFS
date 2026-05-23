import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import resample
from .fcm_tsk import FCMTSKModel


def select_features_mi(X, y, feature_names, max_features=7, random_state=42):
    k = min(max_features, X.shape[1])
    scores = mutual_info_regression(X, y, random_state=random_state)
    idx = np.sort(np.argsort(scores)[::-1][:k])
    return X[:, idx], [feature_names[i] for i in idx], idx


def _fit_one(X, y, n_boot, n_rules, min_rules, min_rule_support, base_params, seed):
    X_b, y_b = resample(X, y, n_samples=n_boot, replace=True, random_state=seed)
    model = FCMTSKModel(n_rules=n_rules, min_rules=min_rules,
                        min_rule_support=min_rule_support,
                        random_state=seed, **base_params)
    model.fit(X_b, y_b)
    return model


class BaggingFCMTSK:
    """Bootstrap ensemble of FCM-TSK models."""

    def __init__(self, n_estimators=10, n_rules=200, min_rules=7,
                 min_rule_support=None, bootstrap_ratio=1.0,
                 base_params=None, random_state=42):
        self.n_estimators = n_estimators
        self.n_rules = n_rules
        self.min_rules = min_rules
        self.min_rule_support = min_rule_support
        self.bootstrap_ratio = bootstrap_ratio
        self.base_params = base_params or {}
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        n_boot = max(1, int(np.ceil(self.bootstrap_ratio * len(X))))
        rng = np.random.RandomState(self.random_state)
        seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(self.n_estimators)]
        self.estimators_ = [
            _fit_one(X, y, n_boot, self.n_rules, self.min_rules,
                     self.min_rule_support, self.base_params, s)
            for s in seeds
        ]
        return self

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.estimators_], axis=1).mean(axis=1)

    def total_rules(self):
        return sum(m.n_rules for m in self.estimators_)
