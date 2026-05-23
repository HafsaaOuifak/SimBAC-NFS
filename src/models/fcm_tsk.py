import numpy as np
import copy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import mutual_info_regression
from pytsk.cluster import FuzzyCMeans


class FCMTSKModel(BaseEstimator, RegressorMixin):
    """FCM-TSK: fuzzy C-means antecedents + ridge consequents."""

    def __init__(self, n_rules=200, min_rules=7, min_rule_support=None,
                 order=1, fcm_iters=300, fcm_error=1e-6, ridge_alpha=1.0,
                 random_state=42, mf_type="gaussian"):
        self.n_rules = n_rules
        self.min_rules = min_rules
        self.min_rule_support = min_rule_support
        self.order = order
        self.fcm_iters = fcm_iters
        self.fcm_error = fcm_error
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.mf_type = mf_type

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # weight features by mutual info so FCM clusters where it matters
        mi = mutual_info_regression(X, y, random_state=self.random_state or 42)
        mi_norm = mi / (mi.max() + 1e-12)
        self.mi_weights_ = np.maximum(mi_norm, 0.1)
        X_scaled_w = X_scaled * self.mi_weights_

        self.n_rules_initial_ = self.n_rules
        self.fcm_ = FuzzyCMeans(n_cluster=self.n_rules, tol_iter=self.fcm_iters,
                                error=self.fcm_error, verbose=0, order=self.order)
        self.fcm_.fit(X_scaled_w)

        # unweight back so downstream code sees normal [0,1] space
        self.centers_ = self.fcm_.cluster_centers_.copy() / self.mi_weights_
        self.sigmas_ = (np.sqrt(np.maximum(self.fcm_.variance_, 1e-8))
                        * self.fcm_.scale_ / self.mi_weights_)
        self.sigmas_ = np.maximum(self.sigmas_, 1e-4)
        self.in_dim_ = X.shape[1]
        self.n_features_in_ = self.in_dim_

        self._prune_weak_rules(X_scaled, y)
        self.train_losses_ = []
        return self

    def _prune_weak_rules(self, X_scaled, y):
        n_initial = self.n_rules_initial_
        threshold = self.min_rule_support if self.min_rule_support is not None \
            else 0.5 / n_initial

        firing = self._compute_firing_strengths(X_scaled)
        support = firing.mean(axis=0)

        keep_mask = support >= threshold
        n_keep = int(keep_mask.sum())

        if n_keep < self.min_rules:
            order_desc = np.argsort(support)[::-1]
            keep_idx = order_desc[:self.min_rules]
            keep_mask = np.zeros(n_initial, dtype=bool)
            keep_mask[keep_idx] = True
            n_keep = self.min_rules

        retained_idx = np.where(keep_mask)[0]
        new_centers = self.centers_[retained_idx]
        new_sigmas = self.sigmas_[retained_idx]

        variance = (new_sigmas / (self.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(n_cluster=n_keep, tol_iter=self.fcm_iters,
                               verbose=0, order=self.order)
        new_fcm.cluster_centers_ = new_centers.copy()
        new_fcm.variance_ = np.maximum(variance, 1e-8)
        new_fcm.scale_ = self.fcm_.scale_
        new_fcm.n_features = self.in_dim_
        new_fcm.m_ = self.fcm_.m_
        new_fcm.membership_degrees_ = np.ones((n_keep, 1)) / n_keep
        new_fcm.fitted = True
        self.fcm_ = new_fcm
        self.centers_ = new_centers.copy()
        self.sigmas_ = new_sigmas.copy()
        self.n_rules = n_keep

        P = self.fcm_.transform(X_scaled)
        self.ridge_ = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.ridge_.fit(P, y)

        coef = self.ridge_.coef_
        if self.order == 1:
            self.consequents_ = coef.reshape(n_keep, self.in_dim_ + 1)
        else:
            self.consequents_ = coef.reshape(n_keep, 1)

    def _compute_firing_strengths(self, X_scaled):
        diff = X_scaled[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]
        s = np.maximum(self.sigmas_[np.newaxis, :, :], 1e-8)

        if self.mf_type == "cauchy":
            mu = 1.0 / (1.0 + (diff / s) ** 2)
        elif self.mf_type == "laplacian":
            mu = np.exp(-np.abs(diff) / s)
        else:
            mu = np.exp(-0.5 * (diff / s) ** 2)

        W = mu.prod(axis=2)
        W = np.maximum(W, 1e-16)
        return W / (W.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X):
        check_is_fitted(self, ["scaler_", "fcm_", "ridge_"])
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self.ridge_.predict(self.fcm_.transform(X_scaled))

    def get_rule_params(self):
        check_is_fitted(self, ["centers_"])
        return {"centers": self.centers_.copy(), "sigmas": self.sigmas_.copy(),
                "consequents": self.consequents_.copy(),
                "n_rules": self.n_rules, "n_features": self.in_dim_}

    def set_rule_params(self, centers, sigmas, consequents):
        check_is_fitted(self, ["fcm_", "scaler_"])
        self.centers_ = centers.copy()
        self.sigmas_ = sigmas.copy()
        self.consequents_ = consequents.copy()
        self.n_rules = len(centers)

        variance = (sigmas / (self.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(n_cluster=self.n_rules, tol_iter=self.fcm_iters,
                               verbose=0, order=self.order)
        new_fcm.cluster_centers_ = centers.copy()
        new_fcm.variance_ = np.maximum(variance, 1e-8)
        new_fcm.scale_ = self.fcm_.scale_
        new_fcm.n_features = self.in_dim_
        new_fcm.m_ = self.fcm_.m_
        new_fcm.membership_degrees_ = np.ones((self.n_rules, 1)) / self.n_rules
        new_fcm.fitted = True
        self.fcm_ = new_fcm

        new_ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        dummy_P = np.ones((2, self.n_rules * (self.in_dim_ + 1)
                           if self.order == 1 else self.n_rules))
        new_ridge.fit(dummy_P, np.zeros(2))
        new_ridge.coef_ = consequents.flatten()
        new_ridge.intercept_ = 0.0
        self.ridge_ = new_ridge

    def get_firing_strengths(self, X):
        check_is_fitted(self, ["scaler_", "centers_"])
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self._compute_firing_strengths(X_scaled)

    def get_linguistic_labels(self, feature_names):
        """Return IF-THEN style linguistic label per feature per rule."""
        check_is_fitted(self, ["centers_"])
        labels = ["LOW", "MEDIUM-LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"]
        rules = []
        for r in range(self.n_rules):
            rule = {}
            for j, feat in enumerate(feature_names):
                c = float(np.clip(self.centers_[r, j], 0.0, 1.0))
                rule[feat] = labels[min(int(c * 5), 4)]
            rules.append(rule)
        return rules
