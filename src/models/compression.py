"""
SimBAC-NFS: Clustered Rule Antecedents with Fitted TSK-consequents for Neuro-Fuzzy Systems.

This is the core methodological contribution of the paper. Given M trained
FCM-TSK base learners, it:

  1. Projects all rules from all bootstrap learners into one training-fold
     coordinate system.
  2. Pools the full ensemble rule population (M × R candidate rules).
  3. Merges redundant rules by complete-linkage clustering after converting
     antecedent similarity to distance as 1 - similarity.
  4. Keeps the important merged concepts by support/weight rather than forcing
     the compressed model to keep the same number of rules as a base learner.
  5. Produces ONE final compressed FCM-TSK model with a single coherent
     interpretable fuzzy rule base.

The compression threshold τ is a complete-linkage similarity threshold:
  - High τ (e.g., 0.97–0.99): stricter merging — only very similar rules merge
    → more concepts kept (less compression).
  - Low τ (e.g., 0.85–0.90): more permissive merging — less similar rules can
    merge → fewer concepts kept (more compression).

Internally the dendrogram is cut at distance 1-τ, so τ is directly the minimum
within-cluster similarity required to form a group.

The paper sweeps τ ∈ {0.85, 0.90, 0.93, 0.95, 0.97, 0.99} for the Pareto
analysis, with τ=0.99 used as the default standalone compressed setting.
"""

import numpy as np
import copy
from typing import List
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

from .fcm_tsk import FCMTSKModel
from .similarity import RuleSimilarity


class GradNFSCompressor:
    """
    Compress a bagging ensemble of FCM-TSK models into one interpretable model.

    Parameters
    ----------
    tau : float
        Complete-linkage similarity threshold in [0, 1].  A group is formed
        only when every pair within the cluster satisfies similarity >= tau
        (equivalently, the dendrogram is cut at distance 1 - tau).
        Higher tau is stricter and keeps more distinct concepts; lower tau is
        more permissive and allows more aggressive merging.
    similarity_method : str
        One of 'bhattacharyya', 'wasserstein', 'centroid', 'cosine', 'combined'.
    weight_by_performance : bool
        If True, weight each model's contribution by its training performance
        (lower MSE → higher weight). If False, uniform weighting is used.
    min_rules : int or None
        Minimum number of concepts to keep after all filtering steps.  When
        the importance-based filters (min_cluster_importance + cumulative_
        importance + max_rules) would leave fewer than min_rules concepts, the
        next most important discarded concepts are restored until the floor is
        met.  Set to None (default) to disable the floor.
    refit_consequents : bool
        If False (default), consequents are averaged from the cluster members
        using the same confidence weights applied to antecedent consolidation
        — the model is derived entirely from the base learners with no
        post-compression fitting step.  If True, Ridge regression is re-solved
        on the consolidated antecedent structure; useful as an ablation to
        quantify how much a post-compression re-fit changes results.
    tune_refit_alpha : bool
        If True and refit_consequents=True, choose the Ridge alpha by
        training-fold cross-validation before fitting the final consequent
        model on the full training fold.
    refit_alpha_grid : array-like or None
        Candidate Ridge alpha values for tuned consequent refitting.  Defaults
        to a small logarithmic grid spanning weak to strong regularization.
    uniform_consolidation : bool
        If True, ignore model quality and rule support during consolidation:
        all candidate rules receive equal weight. Useful as an ablation.
    """

    def __init__(
        self,
        tau="auto",
        similarity_method: str = "combined",
        weight_by_performance: bool = True,
        min_cluster_importance: float = 0.005,
        cumulative_importance: float = 0.995,
        max_rules: int = None,
        min_rules: int = None,
        composite_weights=None,
        refit_consequents: bool = False,
        local_refit: bool = False,
        tune_refit_alpha: bool = True,
        refit_alpha_grid=None,
        uniform_consolidation: bool = False,
    ):
        self.tau = tau
        self.similarity_method = similarity_method
        self.weight_by_performance = weight_by_performance
        self.min_cluster_importance = min_cluster_importance
        self.cumulative_importance = cumulative_importance
        self.max_rules = max_rules
        self.min_rules = min_rules
        self.composite_weights = composite_weights
        self.refit_consequents = refit_consequents
        self.tune_refit_alpha = tune_refit_alpha
        self.refit_alpha_grid = (
            list(refit_alpha_grid)
            if refit_alpha_grid is not None
            else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
        )
        self.local_refit = local_refit
        self.uniform_consolidation = uniform_consolidation

        self._sim_calc = RuleSimilarity(
            method=similarity_method, composite_weights=composite_weights
        )

    def compress(
        self,
        models: List[FCMTSKModel],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> FCMTSKModel:
        """
        Compress M base models into a single interpretable FCM-TSK model.

        Parameters
        ----------
        models : list of FCMTSKModel
            The M trained base learners.
        X_train : np.ndarray
            Training data (used for performance weighting and firing strength
            analysis if weight_by_performance=True).
        y_train : np.ndarray

        Returns
        -------
        compressed_model : FCMTSKModel
            A new FCMTSKModel instance with compressed rule parameters.
        """
        M = len(models)
        if M == 0:
            raise ValueError("Need at least one model to compress.")
        if M == 1:
            return copy.deepcopy(models[0])

        # Express all rule antecedents in one fold-level normalized space before
        # comparing/averaging. Each bootstrap learner owns a different scaler.
        common_scaler = MinMaxScaler().fit(np.asarray(X_train, dtype=np.float64))
        pooled = self._pool_rules(models, X_train, common_scaler)
        n_rules_per_model = models[0].n_rules
        n_features = pooled["centers"].shape[1]
        n_candidates = pooled["centers"].shape[0]

        # Step 1: compute model performance weights
        model_weights = self._compute_model_weights(models, X_train, y_train)

        # Step 2: score candidate-rule importance by model quality and support.
        # uniform_consolidation=True replaces these with equal weights (ablation).
        if self.uniform_consolidation:
            candidate_weights = np.ones(n_candidates) / n_candidates
        else:
            candidate_weights = model_weights[pooled["model_idx"]] * pooled["support"]
            candidate_weights = candidate_weights / (candidate_weights.sum() + 1e-12)

        # Step 3: merge redundant rules. τ is either fixed or auto-selected
        # from the largest gap in the dendrogram (label-free, no leakage).
        sim_matrix = self._candidate_similarity_matrix(
            pooled["centers"], pooled["sigmas"]
        )
        tau = self._resolve_tau(sim_matrix)
        clusters = self._cluster_by_similarity(sim_matrix, tau)

        # Step 4: consolidate each merged concept and prune only unsupported
        # concepts, not arbitrary rule positions.
        concepts = []
        for members in clusters:
            members = np.asarray(members, dtype=int)
            importance = float(candidate_weights[members].sum())
            local_w = candidate_weights[members]
            local_w = local_w / (local_w.sum() + 1e-12)
            centers = np.sum(pooled["centers"][members] * local_w[:, None], axis=0)
            sigmas = np.sum(pooled["sigmas"][members] * local_w[:, None], axis=0)

            if len(members) > 1:
                pair_sims = sim_matrix[np.ix_(members, members)]
                tri = pair_sims[np.triu_indices(len(members), k=1)]
                stability = float(np.mean(tri)) if len(tri) else 1.0
            else:
                stability = 1.0

            concepts.append({
                "members": members.tolist(),
                "importance": importance,
                "center": centers,
                "sigma": sigmas,
                "stability": stability,
            })

        retained = self._select_important_concepts(concepts)
        retained = sorted(retained, key=lambda c: c["importance"], reverse=True)
        n_retained = len(retained)

        consolidated_centers = np.vstack([c["center"] for c in retained])
        consolidated_sigmas = np.vstack([c["sigma"] for c in retained])

        # Ensure sigmas are positive
        consolidated_sigmas = np.maximum(consolidated_sigmas, 1e-4)

        # Step 6: Build compressed model — set antecedents, then average
        # consequents directly from base-learner rules (default) or optionally
        # re-fit Ridge regression on the consolidated structure (ablation).
        if self.local_refit:
            compressed_model = self._rebuild_model_local_refit(
                reference_model=models[0],
                centers=consolidated_centers,
                sigmas=consolidated_sigmas,
                scaler=common_scaler,
                X_train=X_train,
                y_train=y_train,
                n_retained=n_retained,
            )
        elif self.refit_consequents:
            compressed_model = self._rebuild_model_refit(
                reference_model=models[0],
                centers=consolidated_centers,
                sigmas=consolidated_sigmas,
                scaler=common_scaler,
                X_train=X_train,
                y_train=y_train,
                n_retained=n_retained,
            )
        else:
            compressed_model = self._rebuild_model_avg_consequents(
                models=models,
                retained=retained,
                pooled=pooled,
                candidate_weights=candidate_weights,
                centers=consolidated_centers,
                sigmas=consolidated_sigmas,
                scaler=common_scaler,
                n_retained=n_retained,
            )

        # Store compression metadata
        total_ensemble_rules = n_candidates
        compressed_model.compression_meta_ = {
            "tau": tau,
            "similarity_method": self.similarity_method,
            "composite_weights": list(self.composite_weights) if self.composite_weights is not None else [0.5, 0.3, 0.2],
            "M_base_learners": M,
            "n_rules_per_model": n_rules_per_model,
            "n_rules_ensemble_total": total_ensemble_rules,
            "n_rules_after": n_retained,
            # Compression: from ensemble total to single compressed model
            "compression_ratio": round(1.0 - n_retained / total_ensemble_rules, 4),
            "relative_to_base_rule_ratio": round(n_retained / max(n_rules_per_model, 1), 4),
            "n_candidate_rules": int(n_candidates),
            "n_rule_clusters_before_importance": int(len(concepts)),
            "min_cluster_importance": self.min_cluster_importance,
            "cumulative_importance": self.cumulative_importance,
            "min_rules": self.min_rules,
            "max_rules": self.max_rules,
            "rule_stability_scores": [float(c["stability"]) for c in retained],
            "retained_cluster_importance": [float(c["importance"]) for c in retained],
            "retained_cluster_sizes": [int(len(c["members"])) for c in retained],
            "model_weights": model_weights.tolist(),
            "refit_consequents": self.refit_consequents,
            "tune_refit_alpha": self.tune_refit_alpha,
            "refit_alpha": float(getattr(compressed_model, "refit_alpha_", np.nan)),
            "uniform_consolidation": self.uniform_consolidation,
        }

        return compressed_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_tau(self, sim_matrix: np.ndarray) -> float:
        """Return the effective τ — auto-selected or fixed."""
        if self.tau != "auto":
            return float(self.tau)
        return self._auto_tau_from_dendrogram(sim_matrix)

    def _auto_tau_from_dendrogram(self, sim_matrix: np.ndarray) -> float:
        """Label-free τ: largest dendrogram gap that still leaves ≥ min_rules clusters.

        Converts the similarity matrix to distances (1 − sim), runs complete-
        linkage hierarchical clustering, then finds the largest gap between
        consecutive merge heights — but only among gaps where at least
        min_rules clusters would remain after the cut.  This prevents the gap
        criterion from collapsing the pool into a single mega-cluster.
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        n = sim_matrix.shape[0]
        min_r = max(self.min_rules or 3, 3)
        if n <= min_r:
            return 0.5

        dist = 1.0 - np.clip(sim_matrix, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="complete")
        heights = Z[:, 2]

        if len(heights) < 2:
            return float(heights[0]) if len(heights) == 1 else 0.5

        # Only consider gaps that leave at least min_r clusters:
        # cutting after merge k (0-indexed) leaves n - (k+1) clusters.
        # We need n - (k+1) >= min_r  →  k <= n - 1 - min_r
        max_k = n - 1 - min_r          # last valid gap index
        if max_k < 1:
            return float(heights[0])

        gaps = np.diff(heights[:max_k + 2])   # gaps up to and including max_k
        gap_idx = int(np.argmax(gaps[:max_k + 1]))

        # τ = midpoint between the two heights surrounding the gap
        tau = float((heights[gap_idx] + heights[gap_idx + 1]) / 2.0)
        return max(tau, 1e-4)

    def _rebuild_model_local_refit(
        self,
        reference_model: "FCMTSKModel",
        centers: np.ndarray,
        sigmas: np.ndarray,
        scaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_retained: int,
    ) -> "FCMTSKModel":
        """Local consequent refitting: each rule gets its own WLS fit.

        For each retained rule r, fit Ridge regression with sample weights
        equal to the rule's normalized firing strength W[:, r].  This gives
        each rule a locally optimal THEN-part — analogous to how tree
        ensembles compute leaf values — while the IF-parts remain the
        compressed fuzzy antecedents.  Rules with very low total support
        fall back to a zero consequent.  Fitting is parallelized over rules.
        """
        from pytsk.cluster import FuzzyCMeans
        from sklearn.linear_model import Ridge

        in_dim = reference_model.in_dim_

        new_model = FCMTSKModel(
            n_rules=n_retained,
            order=reference_model.order,
            fcm_iters=reference_model.fcm_iters,
            ridge_alpha=reference_model.ridge_alpha,
            random_state=reference_model.random_state,
            mf_type=getattr(reference_model, "mf_type", "gaussian"),
        )
        new_model.scaler_ = copy.deepcopy(scaler)
        new_model.in_dim_ = in_dim
        new_model.n_features_in_ = in_dim
        new_model.train_losses_ = []
        new_model.centers_ = centers.copy()
        new_model.sigmas_ = sigmas.copy()

        variance = (sigmas / (reference_model.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(
            n_cluster=n_retained,
            tol_iter=reference_model.fcm_iters,
            verbose=0,
            order=reference_model.order,
        )
        new_fcm.cluster_centers_ = centers.copy()
        new_fcm.variance_ = np.maximum(variance, 1e-8)
        new_fcm.scale_ = reference_model.fcm_.scale_
        new_fcm.n_features = in_dim
        new_fcm.m_ = reference_model.fcm_.m_
        new_fcm.membership_degrees_ = np.ones((n_retained, 1)) / n_retained
        new_fcm.fitted = True
        new_model.fcm_ = new_fcm

        X_scaled = scaler.transform(np.asarray(X_train, dtype=np.float64))
        y_arr = np.asarray(y_train, dtype=np.float64).flatten()
        N = len(y_arr)

        # Firing strengths: (N, R) — vectorized via FCMTSKModel
        W = new_model._compute_firing_strengths(X_scaled)  # (N, R)

        # Select a single alpha on the full design matrix (shared across rules)
        P_global = new_fcm.transform(X_scaled)
        alpha = self._select_refit_alpha(
            P=P_global, y=y_arr,
            default_alpha=reference_model.ridge_alpha,
            random_state=reference_model.random_state,
        )

        # Design matrix with explicit bias column (fit_intercept=False per rule)
        X_aug = np.column_stack([np.ones(N), X_scaled])  # (N, D+1)

        def _fit_rule(r):
            w = W[:, r]
            if w.sum() < 1e-8:
                return np.zeros(in_dim + 1)
            w_scaled = w / w.sum() * N
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(X_aug, y_arr, sample_weight=w_scaled)
            return ridge.coef_  # (D+1,)

        coefs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_fit_rule)(r) for r in range(n_retained)
        )
        local_consequents = np.vstack(coefs)  # (R, D+1)
        new_model.consequents_ = local_consequents

        # Rebuild ridge_ so predict() works: coef_ = consequents_.flatten(),
        # intercept_ = 0 (bias is already in X_aug column).
        dummy_P = np.ones((2, n_retained * (in_dim + 1)))
        ridge_shell = Ridge(alpha=alpha, fit_intercept=True)
        ridge_shell.fit(dummy_P, np.zeros(2))
        ridge_shell.coef_ = local_consequents.flatten()
        ridge_shell.intercept_ = 0.0
        new_model.ridge_ = ridge_shell
        new_model.ridge_alpha = alpha
        new_model.refit_alpha_ = alpha

        return new_model

    def _pool_rules(
        self,
        models: List[FCMTSKModel],
        X_train: np.ndarray,
        common_scaler: MinMaxScaler,
    ) -> dict:
        """Return all ensemble rules in the common training-fold scaled space."""
        centers, sigmas, support, model_idx, rule_idx = [], [], [], [], []
        X_train = np.asarray(X_train, dtype=np.float64)

        for m_idx, model in enumerate(models):
            p = model.get_rule_params()

            raw_centers = model.scaler_.inverse_transform(p["centers"])
            common_centers = common_scaler.transform(raw_centers)

            # MinMaxScaler deltas: x_scaled = x_raw * scale_ + min_.
            raw_sigmas = p["sigmas"] / (model.scaler_.scale_ + 1e-12)
            common_sigmas = raw_sigmas * common_scaler.scale_

            firing = model.get_firing_strengths(X_train)
            rule_support = firing.mean(axis=0)

            for r in range(p["n_rules"]):
                centers.append(common_centers[r])
                sigmas.append(common_sigmas[r])
                support.append(float(rule_support[r]))
                model_idx.append(m_idx)
                rule_idx.append(r)

        return {
            "centers": np.asarray(centers, dtype=float),
            "sigmas": np.maximum(np.asarray(sigmas, dtype=float), 1e-4),
            "support": np.asarray(support, dtype=float),
            "model_idx": np.asarray(model_idx, dtype=int),
            "rule_idx": np.asarray(rule_idx, dtype=int),
        }

    def _candidate_similarity_matrix(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        return self._sim_calc.similarity_matrix_vectorized(centers, sigmas)

    def _cluster_by_similarity(self, sim_matrix: np.ndarray, tau: float) -> list:
        """Complete-linkage clusters with all merged rules mutually coherent.

        A simple connected-component graph can merge A with C through a chain
        A~B~C even when A and C are not actually similar. Complete linkage uses
        the maximum within-cluster dissimilarity, so a cluster is only formed
        when every pair within the cluster has similarity >= tau (equivalently,
        distance <= 1-tau).
        """
        n = sim_matrix.shape[0]
        if n <= 1:
            return [list(range(n))]

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        dist = 1.0 - np.clip(sim_matrix, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=1.0 - tau, criterion="distance")

        groups = {}
        for i, label in enumerate(labels):
            groups.setdefault(int(label), []).append(i)
        return list(groups.values())

    def _select_important_concepts(self, concepts: list) -> list:
        """Keep supported concepts until cumulative importance is covered.

        Filtering order:
          1. Remove concepts below min_cluster_importance (at least one kept).
          2. Truncate at cumulative_importance threshold.
          3. Truncate at max_rules cap.
          4. Restore discarded concepts (by importance rank) until min_rules
             is satisfied — trades interpretability for accuracy when forced.
        """
        ordered = sorted(concepts, key=lambda c: c["importance"], reverse=True)
        kept = [
            c for c in ordered
            if c["importance"] >= self.min_cluster_importance
        ]
        if not kept and ordered:
            kept = [ordered[0]]

        if self.cumulative_importance is not None and kept:
            selected, cum = [], 0.0
            for c in kept:
                selected.append(c)
                cum += c["importance"]
                if cum >= self.cumulative_importance:
                    break
            kept = selected

        if self.max_rules is not None and self.max_rules > 0:
            kept = kept[:self.max_rules]

        # Enforce min_rules floor: restore next-most-important concepts if needed.
        if self.min_rules is not None and len(kept) < self.min_rules:
            kept_ids = {id(c) for c in kept}
            extras = [c for c in ordered if id(c) not in kept_ids]
            kept = kept + extras[: self.min_rules - len(kept)]

        return kept

    def _compute_model_weights(
        self,
        models: List[FCMTSKModel],
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute model contribution weights based on training MSE.

        Weight = 1 / (MSE + eps), normalized across models.
        If weight_by_performance=False, returns uniform weights.
        """
        if not self.weight_by_performance:
            return np.ones(len(models)) / len(models)

        mse_vals = []
        for m in models:
            y_pred = m.predict(X)
            mse = float(np.mean((y - y_pred) ** 2))
            mse_vals.append(mse)

        mse_arr = np.array(mse_vals) + 1e-10
        weights = 1.0 / mse_arr
        return weights / weights.sum()

    def _rebuild_model_refit(
        self,
        reference_model: FCMTSKModel,
        centers: np.ndarray,
        sigmas: np.ndarray,
        scaler: MinMaxScaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_retained: int,
    ) -> FCMTSKModel:
        """
        Accuracy-calibrated variant: consolidated antecedents + Ridge consequents.

        Consequent parameters are re-solved by Ridge regression against the
        consolidated antecedent structure.  When tune_refit_alpha=True, the
        Ridge alpha is selected by cross-validation using only the training
        fold, then the final Ridge model is fitted on the full training fold.
        This keeps the compressed IF-parts fixed while calibrating the THEN
        equations to the new compact partition.
        """
        from pytsk.cluster import FuzzyCMeans
        from sklearn.linear_model import Ridge

        in_dim = reference_model.in_dim_

        # Build new model
        new_model = FCMTSKModel(
            n_rules=n_retained,
            order=reference_model.order,
            fcm_iters=reference_model.fcm_iters,
            ridge_alpha=reference_model.ridge_alpha,
            random_state=reference_model.random_state,
        )
        new_model.scaler_ = copy.deepcopy(scaler)
        new_model.in_dim_ = in_dim
        new_model.n_features_in_ = in_dim
        new_model.train_losses_ = []
        new_model.centers_ = centers.copy()
        new_model.sigmas_ = sigmas.copy()

        # Reconstruct FCM object with consolidated centers/variances
        variance = (sigmas / (reference_model.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(
            n_cluster=n_retained,
            tol_iter=reference_model.fcm_iters,
            verbose=0,
            order=reference_model.order,
        )
        new_fcm.cluster_centers_ = centers.copy()
        new_fcm.variance_ = np.maximum(variance, 1e-8)
        new_fcm.scale_ = reference_model.fcm_.scale_
        new_fcm.n_features = in_dim
        new_fcm.m_ = reference_model.fcm_.m_
        new_fcm.membership_degrees_ = (
            np.ones((n_retained, 1)) / n_retained
        )
        new_fcm.fitted = True
        new_model.fcm_ = new_fcm

        # Re-solve consequents using training data and consolidated antecedents
        X_scaled = new_model.scaler_.transform(
            np.asarray(X_train, dtype=np.float64)
        )
        P = new_fcm.transform(X_scaled)  # (N, n_retained*(D+1))

        y_arr = np.asarray(y_train, dtype=np.float64).flatten()
        alpha = self._select_refit_alpha(
            P=P,
            y=y_arr,
            default_alpha=reference_model.ridge_alpha,
            random_state=reference_model.random_state,
        )
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(P, y_arr)
        new_model.ridge_ = ridge
        new_model.ridge_alpha = alpha
        new_model.refit_alpha_ = alpha

        # Store consequents in (R, D+1) form
        coef = ridge.coef_
        if reference_model.order == 1:
            new_model.consequents_ = coef.reshape(n_retained, in_dim + 1)
        else:
            new_model.consequents_ = coef.reshape(n_retained, 1)

        return new_model

    def _select_refit_alpha(
        self,
        P: np.ndarray,
        y: np.ndarray,
        default_alpha: float,
        random_state: int = 42,
    ) -> float:
        """Select Ridge alpha for refit using training-fold CV only."""
        if not self.tune_refit_alpha:
            return float(default_alpha)

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        P = np.asarray(P, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        n_samples = len(y)
        if n_samples < 12:
            return float(default_alpha)

        n_splits = min(3, n_samples)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_alpha, best_rmse = float(default_alpha), np.inf

        for alpha in self.refit_alpha_grid:
            alpha = float(alpha)
            fold_rmse = []
            for tr_idx, va_idx in kf.split(P):
                try:
                    ridge = Ridge(alpha=alpha, fit_intercept=True)
                    ridge.fit(P[tr_idx], y[tr_idx])
                    pred = ridge.predict(P[va_idx])
                    fold_rmse.append(float(np.sqrt(np.mean((y[va_idx] - pred) ** 2))))
                except Exception:
                    fold_rmse.append(np.inf)
            rmse = float(np.mean(fold_rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        return best_alpha

    def _rebuild_model_avg_consequents(
        self,
        models: List[FCMTSKModel],
        retained: list,
        pooled: dict,
        candidate_weights: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        scaler,
        n_retained: int,
    ) -> FCMTSKModel:
        """
        Build the compressed model using averaged (not re-fitted) consequents.

        This is the default path.  For each retained concept cluster, the
        consequent coefficients of its member rules are averaged with the same
        candidate weights used for antecedent consolidation.  The Ridge global
        intercept is also averaged across member models so that the compressed
        model's prediction scale is consistent with the source ensemble.
        """
        from pytsk.cluster import FuzzyCMeans
        from sklearn.linear_model import Ridge

        reference_model = models[0]
        in_dim = reference_model.in_dim_

        new_model = FCMTSKModel(
            n_rules=n_retained,
            order=reference_model.order,
            fcm_iters=reference_model.fcm_iters,
            ridge_alpha=reference_model.ridge_alpha,
            random_state=reference_model.random_state,
        )
        new_model.scaler_ = copy.deepcopy(scaler)
        new_model.in_dim_ = in_dim
        new_model.n_features_in_ = in_dim
        new_model.train_losses_ = []
        new_model.centers_ = centers.copy()
        new_model.sigmas_ = sigmas.copy()

        variance = (sigmas / (reference_model.fcm_.scale_ + 1e-12)) ** 2
        new_fcm = FuzzyCMeans(
            n_cluster=n_retained,
            tol_iter=reference_model.fcm_iters,
            verbose=0,
            order=reference_model.order,
        )
        new_fcm.cluster_centers_ = centers.copy()
        new_fcm.variance_ = np.maximum(variance, 1e-8)
        new_fcm.scale_ = reference_model.fcm_.scale_
        new_fcm.n_features = in_dim
        new_fcm.m_ = reference_model.fcm_.m_
        new_fcm.membership_degrees_ = np.ones((n_retained, 1)) / n_retained
        new_fcm.fitted = True
        new_model.fcm_ = new_fcm

        # Average consequent coefficients from cluster members using the same
        # candidate weights as antecedent consolidation.
        n_coef = in_dim + 1 if reference_model.order == 1 else 1
        avg_consequents = np.zeros((n_retained, n_coef))
        for g_idx, concept in enumerate(retained):
            members = np.asarray(concept["members"], dtype=int)
            w = candidate_weights[members]
            w = w / (w.sum() + 1e-12)
            for local_i, cand_idx in enumerate(members):
                m_idx = pooled["model_idx"][cand_idx]
                r_idx = pooled["rule_idx"][cand_idx]
                m_params = models[m_idx].get_rule_params()
                cons = m_params["consequents"][r_idx]
                if cons.ndim == 1:
                    avg_consequents[g_idx] += w[local_i] * cons[:n_coef]
                else:
                    avg_consequents[g_idx] += w[local_i] * cons.flatten()[:n_coef]

        # Average the Ridge global intercept across all member models, weighted
        # by each member's candidate weight (summed over that model's members).
        # This preserves the prediction offset carried by the base-learner Ridge fits.
        model_total_w = np.zeros(len(models))
        for cand_idx, m_idx in enumerate(pooled["model_idx"]):
            model_total_w[m_idx] += candidate_weights[cand_idx]
        model_total_w /= model_total_w.sum() + 1e-12
        avg_intercept = float(
            sum(
                model_total_w[m_idx] * float(models[m_idx].ridge_.intercept_)
                for m_idx in range(len(models))
            )
        )

        new_model.consequents_ = avg_consequents
        ridge = Ridge(alpha=reference_model.ridge_alpha, fit_intercept=True)
        ridge.coef_ = avg_consequents.flatten()
        ridge.intercept_ = avg_intercept
        new_model.ridge_ = ridge

        return new_model

    def compress_from_antecedents(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        common_scaler=None,
        reference_stage: FCMTSKModel = None,
        order: int = 1,
        fcm_iters: int = 300,
        ridge_alpha: float = 1.0,
        random_state: int = 42,
    ) -> FCMTSKModel:
        """
        Compress a raw pool of antecedents (no model objects) and refit NFS.

        Used by GradFuzzyTSK to compress all boosting-stage antecedents into
        one interpretable model.  Centers and sigmas must already be in the
        common_scaler coordinate system.

        Parameters
        ----------
        centers : (n_candidates, D)
        sigmas  : (n_candidates, D)  — in common_scaler units, > 0
        X_train, y_train : training fold (raw feature space)
        common_scaler : fitted MinMaxScaler used to map X to the same space
            as centers/sigmas.  Fitted on X_train if None.
        reference_stage : FCMTSKModel whose FCM scale_ and m_ are reused so
            that pytsk's internal arithmetic is consistent.
        """
        from pytsk.cluster import FuzzyCMeans
        from sklearn.linear_model import Ridge

        centers = np.asarray(centers, dtype=np.float64)
        sigmas = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-4)
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).flatten()
        n_candidates, in_dim = centers.shape

        if common_scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            common_scaler = MinMaxScaler().fit(X_train)
        X_scaled = common_scaler.transform(X_train)

        # Importance weights from mean normalized firing strength
        support = self._compute_raw_support(X_scaled, centers, sigmas)
        candidate_weights = support / (support.sum() + 1e-12)

        # SimBAC-NFS clustering on pooled antecedents
        sim_matrix = self._candidate_similarity_matrix(centers, sigmas)
        clusters = self._cluster_by_similarity(sim_matrix, self.tau)

        concepts = []
        for members in clusters:
            members = np.asarray(members, dtype=int)
            importance = float(candidate_weights[members].sum())
            local_w = candidate_weights[members]
            local_w = local_w / (local_w.sum() + 1e-12)
            merged_centers = np.sum(centers[members] * local_w[:, None], axis=0)
            merged_sigmas = np.sum(sigmas[members] * local_w[:, None], axis=0)
            if len(members) > 1:
                pair_sims = sim_matrix[np.ix_(members, members)]
                tri = pair_sims[np.triu_indices(len(members), k=1)]
                stability = float(np.mean(tri)) if len(tri) else 1.0
            else:
                stability = 1.0
            concepts.append({
                "members": members.tolist(),
                "importance": importance,
                "center": merged_centers,
                "sigma": merged_sigmas,
                "stability": stability,
            })

        retained = self._select_important_concepts(concepts)
        retained = sorted(retained, key=lambda c: c["importance"], reverse=True)
        n_retained = len(retained)

        consolidated_centers = np.vstack([c["center"] for c in retained])
        consolidated_sigmas = np.maximum(
            np.vstack([c["sigma"] for c in retained]), 1e-4
        )

        # FCM scale_/m_ — reuse reference stage values for pytsk consistency
        fcm_scale = float(getattr(
            getattr(reference_stage, "fcm_", None), "scale_", 1.0
        )) if reference_stage is not None else 1.0
        fcm_m = float(getattr(
            getattr(reference_stage, "fcm_", None), "m_", 2.0
        )) if reference_stage is not None else 2.0

        variance = (consolidated_sigmas / (fcm_scale + 1e-12)) ** 2

        new_fcm = FuzzyCMeans(
            n_cluster=n_retained,
            tol_iter=fcm_iters,
            verbose=0,
            order=order,
        )
        new_fcm.cluster_centers_ = consolidated_centers.copy()
        new_fcm.variance_ = np.maximum(variance, 1e-8)
        new_fcm.scale_ = fcm_scale
        new_fcm.n_features = in_dim
        new_fcm.m_ = fcm_m
        new_fcm.membership_degrees_ = np.ones((n_retained, 1)) / n_retained
        new_fcm.fitted = True

        new_model = FCMTSKModel(
            n_rules=n_retained,
            order=order,
            fcm_iters=fcm_iters,
            ridge_alpha=ridge_alpha,
            random_state=random_state,
        )
        new_model.scaler_ = copy.deepcopy(common_scaler)
        new_model.in_dim_ = in_dim
        new_model.n_features_in_ = in_dim
        new_model.train_losses_ = []
        new_model.centers_ = consolidated_centers.copy()
        new_model.sigmas_ = consolidated_sigmas.copy()
        new_model.fcm_ = new_fcm

        # NFS refit: Ridge on compressed antecedent structure
        P = new_fcm.transform(X_scaled)
        alpha = self._select_refit_alpha(
            P=P, y=y_train,
            default_alpha=ridge_alpha,
            random_state=random_state,
        )
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(P, y_train)
        new_model.ridge_ = ridge
        new_model.ridge_alpha = alpha
        new_model.refit_alpha_ = alpha

        if order == 1:
            new_model.consequents_ = ridge.coef_.reshape(n_retained, in_dim + 1)
        else:
            new_model.consequents_ = ridge.coef_.reshape(n_retained, 1)

        new_model.compression_meta_ = {
            "tau": self.tau,
            "n_candidates": n_candidates,
            "n_rules_after": n_retained,
            "compression_ratio": round(1.0 - n_retained / max(n_candidates, 1), 4),
            "refit_alpha": float(alpha),
            "method": "compress_from_antecedents",
            "max_rules": self.max_rules,
            "min_rules": self.min_rules,
        }
        return new_model

    def _compute_raw_support(
        self,
        X_scaled: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        """Mean normalized firing strength for a raw antecedent pool (vectorized)."""
        # X_scaled: (N, D), centers: (R, D), sigmas: (R, D)
        diff = X_scaled[:, None, :] - centers[None, :, :]   # (N, R, D)
        s = np.maximum(sigmas[None, :, :], 1e-8)             # (1, R, D)
        W = np.exp(-0.5 * (diff / s) ** 2).prod(axis=2)      # (N, R)
        W = np.maximum(W, 1e-16)
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        return W_norm.mean(axis=0)

    def get_compression_summary(self, compressed_model: FCMTSKModel) -> dict:
        """Return a human-readable summary of the compression."""
        meta = getattr(compressed_model, "compression_meta_", {})
        if not meta:
            return {"error": "No compression metadata found."}
        return {
            "tau": meta["tau"],
            "similarity_method": meta["similarity_method"],
            "base_learners": meta["M_base_learners"],
            "rules_per_model": meta["n_rules_per_model"],
            "ensemble_total_rules": meta["n_rules_ensemble_total"],
            "rules_after": meta["n_rules_after"],
            "compression_pct": f"{meta['compression_ratio']*100:.1f}%",
            "relative_to_base_rule_ratio": meta["relative_to_base_rule_ratio"],
            "n_candidate_rules": meta["n_candidate_rules"],
            "n_rule_clusters_before_importance": meta["n_rule_clusters_before_importance"],
            "mean_rule_stability": round(float(np.mean(meta["rule_stability_scores"])), 4),
            "refit_alpha": meta.get("refit_alpha", None),
        }

# backwards-compatible alias
SIMBACCompressor = GradNFSCompressor
