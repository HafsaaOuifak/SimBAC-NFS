import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .fcm_tsk import FCMTSKModel
from .similarity import RuleSimilarity


class GradNFSCompressor:
    """
    Compress a pool of FCM-TSK models into one compact interpretable model.

    tau controls how aggressively rules are merged — higher means stricter
    (only very similar rules merge, so more rules survive).
    """

    def __init__(self, tau=0.95, similarity_method="combined",
                 weight_by_performance=True, min_cluster_importance=0.005,
                 cumulative_importance=0.995, max_rules=None, min_rules=None,
                 refit_consequents=False, tune_refit_alpha=True,
                 refit_alpha_grid=None):
        self.tau = tau
        self.similarity_method = similarity_method
        self.weight_by_performance = weight_by_performance
        self.min_cluster_importance = min_cluster_importance
        self.cumulative_importance = cumulative_importance
        self.max_rules = max_rules
        self.min_rules = min_rules
        self.refit_consequents = refit_consequents
        self.tune_refit_alpha = tune_refit_alpha
        self.refit_alpha_grid = list(refit_alpha_grid) if refit_alpha_grid is not None \
            else [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
        self._sim = RuleSimilarity(method=similarity_method)

    def compress(self, models, X_train, y_train):
        if len(models) == 0:
            raise ValueError("Need at least one model.")
        if len(models) == 1:
            return copy.deepcopy(models[0])

        # bring all rules into one common coordinate system
        common_scaler = MinMaxScaler().fit(np.asarray(X_train, dtype=np.float64))
        pooled = self._pool_rules(models, X_train, common_scaler)

        # weight each rule: how good was its model × how often it fires
        model_weights = self._compute_model_weights(models, X_train, y_train)
        candidate_weights = model_weights[pooled["model_idx"]] * pooled["support"]
        candidate_weights /= candidate_weights.sum() + 1e-12

        # cluster similar rules together
        sim_matrix = self._sim.similarity_matrix_vectorized(
            pooled["centers"], pooled["sigmas"])
        clusters = self._cluster(sim_matrix, float(self.tau))

        # consolidate each cluster into one representative rule
        concepts = []
        for members in clusters:
            members = np.asarray(members, dtype=int)
            importance = float(candidate_weights[members].sum())
            lw = candidate_weights[members] / (candidate_weights[members].sum() + 1e-12)
            concepts.append({
                "members": members.tolist(),
                "importance": importance,
                "center": np.sum(pooled["centers"][members] * lw[:, None], axis=0),
                "sigma": np.sum(pooled["sigmas"][members] * lw[:, None], axis=0),
            })

        retained = self._select_concepts(concepts)
        retained = sorted(retained, key=lambda c: c["importance"], reverse=True)

        centers = np.vstack([c["center"] for c in retained])
        sigmas = np.maximum(np.vstack([c["sigma"] for c in retained]), 1e-4)

        if self.refit_consequents:
            compressed = self._rebuild_refit(models[0], centers, sigmas,
                                             common_scaler, X_train, y_train)
        else:
            compressed = self._rebuild_avg(models, retained, pooled,
                                           candidate_weights, centers, sigmas,
                                           common_scaler)

        compressed.compression_meta_ = {
            "tau": float(self.tau),
            "similarity_method": self.similarity_method,
            "composite_weights": [0.5, 0.3, 0.2],
            "M_base_learners": len(models),
            "n_rules_per_model": models[0].n_rules,
            "n_rules_ensemble_total": pooled["centers"].shape[0],
            "n_rules_after": len(retained),
            "compression_ratio": round(1.0 - len(retained) / pooled["centers"].shape[0], 4),
            "compression_pct": f"{100*(1 - len(retained)/pooled['centers'].shape[0]):.1f}%",
            "rule_stability_scores": [1.0] * len(retained),
            "refit_alpha": float(getattr(compressed, "refit_alpha_", float("nan"))),
        }
        return compressed

    # ---------------------------------------------------------------

    def _pool_rules(self, models, X_train, common_scaler):
        centers, sigmas, support, model_idx = [], [], [], []
        X_train = np.asarray(X_train, dtype=np.float64)
        for m_i, model in enumerate(models):
            p = model.get_rule_params()
            raw_c = model.scaler_.inverse_transform(p["centers"])
            c_common = common_scaler.transform(raw_c)
            s_common = p["sigmas"] / (model.scaler_.scale_ + 1e-12) * common_scaler.scale_
            fs = model.get_firing_strengths(X_train).mean(axis=0)
            for r in range(p["n_rules"]):
                centers.append(c_common[r])
                sigmas.append(s_common[r])
                support.append(float(fs[r]))
                model_idx.append(m_i)
        return {
            "centers": np.asarray(centers),
            "sigmas": np.maximum(np.asarray(sigmas), 1e-4),
            "support": np.asarray(support),
            "model_idx": np.asarray(model_idx, dtype=int),
            "rule_idx": np.arange(len(centers)),  # kept for avg-consequents path
        }

    def _compute_model_weights(self, models, X, y):
        if not self.weight_by_performance:
            return np.ones(len(models)) / len(models)
        mse = np.array([np.mean((y - m.predict(X)) ** 2) for m in models]) + 1e-10
        w = 1.0 / mse
        return w / w.sum()

    def _cluster(self, sim_matrix, tau):
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        n = sim_matrix.shape[0]
        if n <= 1:
            return [list(range(n))]
        dist = np.clip(1.0 - sim_matrix, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="complete")
        labels = fcluster(Z, t=1.0 - tau, criterion="distance")
        groups = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        return list(groups.values())

    def _select_concepts(self, concepts):
        ordered = sorted(concepts, key=lambda c: c["importance"], reverse=True)
        kept = [c for c in ordered if c["importance"] >= self.min_cluster_importance]
        if not kept:
            kept = [ordered[0]]
        if self.cumulative_importance is not None:
            selected, cum = [], 0.0
            for c in kept:
                selected.append(c)
                cum += c["importance"]
                if cum >= self.cumulative_importance:
                    break
            kept = selected
        if self.max_rules:
            kept = kept[:self.max_rules]
        # restore rules if we fell below the floor
        if self.min_rules and len(kept) < self.min_rules:
            extras = [c for c in ordered if id(c) not in {id(x) for x in kept}]
            kept += extras[:self.min_rules - len(kept)]
        return kept

    def _rebuild_refit(self, ref, centers, sigmas, scaler, X_train, y_train):
        from pytsk.cluster import FuzzyCMeans
        n = len(centers)
        m = FCMTSKModel(n_rules=n, order=ref.order, fcm_iters=ref.fcm_iters,
                        ridge_alpha=ref.ridge_alpha, random_state=ref.random_state)
        m.scaler_ = copy.deepcopy(scaler)
        m.in_dim_ = ref.in_dim_
        m.n_features_in_ = ref.in_dim_
        m.train_losses_ = []
        m.centers_ = centers.copy()
        m.sigmas_ = sigmas.copy()

        fcm = FuzzyCMeans(n_cluster=n, tol_iter=ref.fcm_iters, verbose=0, order=ref.order)
        fcm.cluster_centers_ = centers.copy()
        fcm.variance_ = np.maximum((sigmas / (ref.fcm_.scale_ + 1e-12)) ** 2, 1e-8)
        fcm.scale_ = ref.fcm_.scale_
        fcm.n_features = ref.in_dim_
        fcm.m_ = ref.fcm_.m_
        fcm.membership_degrees_ = np.ones((n, 1)) / n
        fcm.fitted = True
        m.fcm_ = fcm

        X_scaled = scaler.transform(np.asarray(X_train, dtype=np.float64))
        P = fcm.transform(X_scaled)
        y_arr = np.asarray(y_train, dtype=np.float64).flatten()

        alpha = self._pick_alpha(P, y_arr, ref.ridge_alpha, ref.random_state)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(P, y_arr)
        m.ridge_ = ridge
        m.ridge_alpha = alpha
        m.refit_alpha_ = alpha
        m.consequents_ = ridge.coef_.reshape(n, ref.in_dim_ + 1 if ref.order == 1 else 1)
        return m

    def _rebuild_avg(self, models, retained, pooled, cand_w, centers, sigmas, scaler):
        from pytsk.cluster import FuzzyCMeans
        ref = models[0]
        n = len(retained)
        m = FCMTSKModel(n_rules=n, order=ref.order, fcm_iters=ref.fcm_iters,
                        ridge_alpha=ref.ridge_alpha, random_state=ref.random_state)
        m.scaler_ = copy.deepcopy(scaler)
        m.in_dim_ = ref.in_dim_
        m.n_features_in_ = ref.in_dim_
        m.train_losses_ = []
        m.centers_ = centers.copy()
        m.sigmas_ = sigmas.copy()

        fcm = FuzzyCMeans(n_cluster=n, tol_iter=ref.fcm_iters, verbose=0, order=ref.order)
        fcm.cluster_centers_ = centers.copy()
        fcm.variance_ = np.maximum((sigmas / (ref.fcm_.scale_ + 1e-12)) ** 2, 1e-8)
        fcm.scale_ = ref.fcm_.scale_
        fcm.n_features = ref.in_dim_
        fcm.m_ = ref.fcm_.m_
        fcm.membership_degrees_ = np.ones((n, 1)) / n
        fcm.fitted = True
        m.fcm_ = fcm

        n_coef = ref.in_dim_ + 1 if ref.order == 1 else 1
        avg_cons = np.zeros((n, n_coef))
        for g, concept in enumerate(retained):
            members = np.asarray(concept["members"], dtype=int)
            w = cand_w[members]
            w /= w.sum() + 1e-12
            for li, ci in enumerate(members):
                mi = pooled["model_idx"][ci]
                ri = pooled["rule_idx"][ci]
                cons = models[mi].get_rule_params()["consequents"][ri]
                avg_cons[g] += w[li] * cons.flatten()[:n_coef]

        model_w = np.zeros(len(models))
        for ci, mi in enumerate(pooled["model_idx"]):
            model_w[mi] += cand_w[ci]
        model_w /= model_w.sum() + 1e-12
        avg_intercept = sum(model_w[mi] * float(models[mi].ridge_.intercept_)
                            for mi in range(len(models)))

        m.consequents_ = avg_cons
        ridge = Ridge(alpha=ref.ridge_alpha, fit_intercept=True)
        ridge.coef_ = avg_cons.flatten()
        ridge.intercept_ = avg_intercept
        m.ridge_ = ridge
        return m

    def _pick_alpha(self, P, y, default_alpha, random_state):
        if not self.tune_refit_alpha or len(y) < 12:
            return float(default_alpha)
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        best, best_rmse = float(default_alpha), np.inf
        for alpha in self.refit_alpha_grid:
            scores = []
            for tr, va in kf.split(P):
                try:
                    r = Ridge(alpha=alpha, fit_intercept=True)
                    r.fit(P[tr], y[tr])
                    scores.append(float(np.sqrt(np.mean((y[va] - r.predict(P[va])) ** 2))))
                except Exception:
                    scores.append(np.inf)
            if np.mean(scores) < best_rmse:
                best_rmse = np.mean(scores)
                best = float(alpha)
        return best
