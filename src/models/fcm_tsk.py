"""
FCM-based TSK (Takagi-Sugeno-Kang) model with max-rule initialization
and support-based pruning.

Training follows a three-phase approach:
  Phase 1: Fuzzy C-Means clustering on input data with up to MAX_RULES clusters
            → antecedent parameters (Gaussian MF centers and widths).
  Phase 2: Prune weak rules by mean normalized firing strength on the training
            fold. Rules below the support threshold are discarded; at least
            MIN_RULES are always retained. All threshold decisions use only
            training-fold data.
  Phase 3: Ridge regression on the pruned antecedent structure → TSK consequent
            coefficients. Refitting after pruning recalibrates the THEN-parts to
            the reduced rule base without accessing test data.

The budget-then-prune strategy replaces rule-count grid search:
  - FCM always starts from the full budget (MAX_RULES) so every base learner
    explores the same structural richness.
  - Data-driven pruning then discards rules that the training distribution does
    not support, yielding compact, dataset-adapted rule bases automatically.
  - The final rule count varies per bootstrap sample and per dataset, reflecting
    genuine structural complexity rather than a pre-specified integer.
"""

import numpy as np
import copy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import mutual_info_regression

from pytsk.cluster import FuzzyCMeans


class FCMTSKModel(BaseEstimator, RegressorMixin):
    """
    Two-phase FCM-TSK model: FCM antecedents + support-pruning + Ridge consequents.

    Parameters
    ----------
    n_rules : int
        Maximum rule budget for FCM initialization (= MAX_RULES).
        The actual number of rules after pruning may be smaller.
    min_rules : int
        Hard floor: pruning will never reduce the rule count below this.
    min_rule_support : float or None
        Rules whose mean normalized firing strength falls below this threshold
        are pruned. If None, the threshold is set automatically to
        0.5 / n_rules (half the uniform average), which adapts to the budget.
    order : int
        TSK order: 1 = first-order (linear consequents), 0 = zero-order.
    fcm_iters : int
        Maximum FCM iteration count.
    fcm_error : float
        FCM convergence tolerance.
    ridge_alpha : float
        Ridge regularization for consequent estimation.
    random_state : int or None
        Seed for FCM initialization.
    """

    def __init__(
        self,
        n_rules: int = 200,
        min_rules: int = 7,
        min_rule_support: float = None,
        order: int = 1,
        fcm_iters: int = 300,
        fcm_error: float = 1e-6,
        ridge_alpha: float = 1.0,
        random_state: int = 42,
        mf_type: str = "gaussian",
    ):
        self.n_rules = n_rules
        self.min_rules = min_rules
        self.min_rule_support = min_rule_support
        self.order = order
        self.fcm_iters = fcm_iters
        self.fcm_error = fcm_error
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.mf_type = mf_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FCMTSKModel":
        """
        Fit: Phase 1 (FCM) → Phase 2 (prune weak rules) → Phase 3 (Ridge refit).

        All three phases operate exclusively on the provided (X, y), which must
        be a training fold. No test data is accessed at any point.
        """
        # FuzzyCMeans (pytsk) draws from numpy's global random state, so we
        # must set it globally here.  Within a single-threaded outer CV loop
        # this is deterministic: each fold calls fit() with the same seed.
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        # ── Phase 1: Scale and cluster ────────────────────────────────────────
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Feature-weighted FCM: weight each feature by its MI with y so that
        # FCM places clusters in predictive regions, not just geometrically
        # compact ones.  Weights are max-normalised with a 0.1 floor so no
        # feature is fully ignored.  Centers and sigmas are then unweighted
        # back to the standard [0,1] scaled space so the compressor and all
        # downstream code work unchanged — low-MI features end up with wide
        # sigmas, making them nearly inactive in firing strength computation.
        mi = mutual_info_regression(
            X, y,
            random_state=self.random_state if self.random_state is not None else 42,
        )
        mi_norm = mi / (mi.max() + 1e-12)
        self.mi_weights_ = np.maximum(mi_norm, 0.1)
        X_scaled_w = X_scaled * self.mi_weights_

        # Store initial budget before pruning may reduce it
        self.n_rules_initial_ = self.n_rules

        self.fcm_ = FuzzyCMeans(
            n_cluster=self.n_rules,
            tol_iter=self.fcm_iters,
            error=self.fcm_error,
            verbose=0,
            order=self.order,
        )
        self.fcm_.fit(X_scaled_w)

        # Unweight centers and sigmas back to standard [0,1] space
        self.centers_ = self.fcm_.cluster_centers_.copy() / self.mi_weights_
        self.sigmas_ = (
            np.sqrt(np.maximum(self.fcm_.variance_, 1e-8)) * self.fcm_.scale_
            / self.mi_weights_
        )
        self.sigmas_ = np.maximum(self.sigmas_, 1e-4)
        self.in_dim_ = X.shape[1]
        self.n_features_in_ = self.in_dim_

        # ── Phase 2: Prune weak rules on standard scaled X ────────────────────
        self._prune_weak_rules(X_scaled, y)

        self.train_losses_ = []
        return self

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_weak_rules(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Remove rules with low mean normalized firing strength, then refit Ridge.

        Decision logic (training-fold only, no test access):
          1. Compute mean normalized firing strength per rule → rule support.
          2. Prune rules below `min_rule_support` threshold
             (default: 0.5 / n_rules_initial, i.e. half the uniform average).
          3. Enforce at least self.min_rules and at most self.n_rules_initial_.
          4. Rebuild the FCM object with the pruned antecedent set.
          5. Refit Ridge consequents on the pruned structure.
        """
        n_initial = self.n_rules_initial_
        threshold = (
            self.min_rule_support
            if self.min_rule_support is not None
            else 0.5 / n_initial
        )

        # Step 1: compute per-rule mean support on training data
        firing = self._compute_firing_strengths(X_scaled)  # (N, R)
        support = firing.mean(axis=0)                       # (R,)

        # Step 2: determine which rules to keep
        keep_mask = support >= threshold
        n_keep = int(keep_mask.sum())

        # Step 3: enforce floor and ceiling
        if n_keep < self.min_rules:
            # restore the weakest rules until we hit the floor
            order_desc = np.argsort(support)[::-1]
            keep_idx = order_desc[: self.min_rules]
            keep_mask = np.zeros(n_initial, dtype=bool)
            keep_mask[keep_idx] = True
            n_keep = self.min_rules
        if n_keep > n_initial:
            n_keep = n_initial

        retained_idx = np.where(keep_mask)[0]

        # Step 4: rebuild FCM with retained antecedents
        new_centers = self.centers_[retained_idx]
        new_sigmas  = self.sigmas_[retained_idx]

        variance = (new_sigmas / (self.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(
            n_cluster=n_keep,
            tol_iter=self.fcm_iters,
            verbose=0,
            order=self.order,
        )
        new_fcm.cluster_centers_    = new_centers.copy()
        new_fcm.variance_           = np.maximum(variance, 1e-8)
        new_fcm.scale_              = self.fcm_.scale_
        new_fcm.n_features          = self.in_dim_
        new_fcm.m_                  = self.fcm_.m_
        new_fcm.membership_degrees_ = np.ones((n_keep, 1)) / n_keep
        new_fcm.fitted              = True
        self.fcm_     = new_fcm
        self.centers_ = new_centers.copy()
        self.sigmas_  = new_sigmas.copy()

        # Update n_rules to actual post-pruning count
        self.n_rules  = n_keep

        # Record which rules survived (for diagnostics)
        self.pruned_support_     = float(support[retained_idx].mean())
        self.n_rules_pruned_     = n_keep
        self.pruning_threshold_  = threshold

        # Step 5: refit Ridge on pruned antecedent structure
        P = self.fcm_.transform(X_scaled)   # (N, n_keep*(D+1)) or (N, n_keep)
        self.ridge_ = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.ridge_.fit(P, y)

        coef = self.ridge_.coef_
        if self.order == 1:
            self.consequents_ = coef.reshape(n_keep, self.in_dim_ + 1)
        else:
            self.consequents_ = coef.reshape(n_keep, 1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _compute_firing_strengths(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return normalized firing strength matrix: (N, R) — fully vectorized."""
        # diff: (N, R, D) via broadcasting
        diff = X_scaled[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]
        s = np.maximum(self.sigmas_[np.newaxis, :, :], 1e-8)  # (1, R, D)

        if self.mf_type == "cauchy":
            mu = 1.0 / (1.0 + (diff / s) ** 2)
        elif self.mf_type == "laplacian":
            mu = np.exp(-np.abs(diff) / s)
        else:  # default: gaussian
            mu = np.exp(-0.5 * (diff / s) ** 2)

        W = mu.prod(axis=2)  # (N, R) — product over features
        W = np.maximum(W, 1e-16)
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        return W_norm

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression targets."""
        check_is_fitted(self, ["scaler_", "fcm_", "ridge_"])
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X)
        P = self.fcm_.transform(X_scaled)
        return self.ridge_.predict(P)

    # ------------------------------------------------------------------
    # Compression framework interface
    # ------------------------------------------------------------------

    def get_rule_params(self) -> dict:
        """Return rule parameters for the compression framework."""
        check_is_fitted(self, ["centers_"])
        return {
            "centers":     self.centers_.copy(),
            "sigmas":      self.sigmas_.copy(),
            "consequents": self.consequents_.copy(),
            "n_rules":     self.n_rules,
            "n_features":  self.in_dim_,
        }

    def set_rule_params(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
        consequents: np.ndarray,
    ) -> None:
        """Overwrite rule parameters (used by compression framework)."""
        check_is_fitted(self, ["fcm_", "scaler_"])
        self.centers_     = centers.copy()
        self.sigmas_      = sigmas.copy()
        self.consequents_ = consequents.copy()
        self.n_rules      = len(centers)

        variance = (sigmas / (self.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(
            n_cluster=self.n_rules,
            tol_iter=self.fcm_iters,
            verbose=0,
            order=self.order,
        )
        new_fcm.cluster_centers_    = centers.copy()
        new_fcm.variance_           = np.maximum(variance, 1e-8)
        new_fcm.scale_              = self.fcm_.scale_
        new_fcm.n_features          = self.in_dim_
        new_fcm.m_                  = self.fcm_.m_
        new_fcm.membership_degrees_ = np.ones((self.n_rules, 1)) / self.n_rules
        new_fcm.fitted              = True
        self.fcm_ = new_fcm

        new_ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        dummy_P = np.ones((2, self.n_rules * (self.in_dim_ + 1) if self.order == 1 else self.n_rules))
        new_ridge.fit(dummy_P, np.zeros(2))
        new_ridge.coef_      = consequents.flatten()
        new_ridge.intercept_ = 0.0
        self.ridge_ = new_ridge

    def get_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Return normalized firing strengths: (N, R)."""
        check_is_fitted(self, ["scaler_", "centers_"])
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self._compute_firing_strengths(X_scaled)

    def get_linguistic_labels(self, feature_names: list) -> list:
        """Generate human-readable linguistic labels for each fuzzy rule."""
        check_is_fitted(self, ["centers_"])
        labels   = ["LOW", "MEDIUM-LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"]
        n_labels = len(labels)
        rules = []
        for r in range(self.n_rules):
            rule = {}
            for j, feat in enumerate(feature_names):
                c         = float(np.clip(self.centers_[r, j], 0.0, 1.0))
                label_idx = min(int(c * n_labels), n_labels - 1)
                rule[feat] = labels[label_idx]
            rules.append(rule)
        return rules
