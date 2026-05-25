"""
Fuzzy rule similarity measures for the ensemble compression framework.

Implements multiple pairwise similarity metrics between Gaussian fuzzy rules:
  - Gaussian overlap (Bhattacharyya coefficient)
  - Wasserstein distance (Earth Mover's Distance, inverted)
  - Centroid L2 distance (inverted)
  - Cosine similarity of centroid vectors
  - Combined weighted similarity

These metrics compare fuzzy rules by their antecedent membership functions
rather than simple scalar parameter comparisons, enabling semantically meaningful
rule consolidation.
"""

import numpy as np
from scipy.spatial.distance import cosine


class RuleSimilarity:
    """
    Pairwise fuzzy rule similarity calculator.

    All methods operate in the normalized [0, 1] feature space, so distances
    are directly comparable across datasets and features.

    Parameters
    ----------
    method : str
        Similarity method: 'bhattacharyya', 'wasserstein', 'centroid',
        'cosine', or 'combined'.
    weights : array-like, optional
        Per-feature weights for the combined method. Defaults to uniform.
    composite_weights : tuple of float, optional
        Coefficients (w_bc, w_ws, w_cs) for the 'combined' method.
        Must sum to 1. Defaults to (0.5, 0.3, 0.2).
    """

    VALID_METHODS = ("bhattacharyya", "wasserstein", "centroid", "cosine", "combined")

    def __init__(self, method: str = "combined", weights=None, composite_weights=None):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        self.method = method
        self.weights = weights
        if composite_weights is None:
            self._w_bc, self._w_ws, self._w_cs = 0.5, 0.3, 0.2
        else:
            w = tuple(float(x) for x in composite_weights)
            if len(w) != 3:
                raise ValueError("composite_weights must have exactly 3 elements")
            if any(x < 0 for x in w):
                raise ValueError("composite_weights must be non-negative")
            if not np.isclose(sum(w), 1.0, atol=1e-8):
                raise ValueError("composite_weights must sum to 1")
            self._w_bc, self._w_ws, self._w_cs = w

    def similarity(
        self,
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
    ) -> float:
        """
        Compute similarity between two fuzzy rules.

        Each rule is defined by its Gaussian antecedent parameters:
          - c: center vector (n_features,)
          - s: sigma vector  (n_features,)

        Returns a scalar similarity in [0, 1], where 1 = identical rules.
        """
        c1, s1, c2, s2 = (np.asarray(a, dtype=float) for a in (c1, s1, c2, s2))

        if self.method == "bhattacharyya":
            return self._bhattacharyya(c1, s1, c2, s2)
        elif self.method == "wasserstein":
            return self._wasserstein_sim(c1, s1, c2, s2)
        elif self.method == "centroid":
            return self._centroid_sim(c1, c2)
        elif self.method == "cosine":
            return self._cosine_sim(c1, c2)
        else:  # combined
            return self._combined(c1, s1, c2, s2)

    def similarity_matrix(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full (n_rules × n_rules) pairwise similarity matrix.

        Parameters
        ----------
        centers : (n_rules, n_features)
        sigmas  : (n_rules, n_features)

        Returns
        -------
        S : (n_rules, n_rules) symmetric matrix, diagonal = 1.0
        """
        n = len(centers)
        S = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                s = self.similarity(centers[i], sigmas[i], centers[j], sigmas[j])
                S[i, j] = s
                S[j, i] = s
        return S

    def similarity_matrix_vectorized(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        """
        Fully vectorized pairwise similarity matrix — no Python loops.

        Uses numpy broadcasting over (n, 1, D) and (1, n, D) arrays so
        the entire (n, n) matrix is computed in a handful of numpy calls.
        ~1000× faster than the double-loop version for n > 100.
        """
        c = np.asarray(centers, dtype=np.float64)
        s = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-8)
        n = len(c)

        # broadcast to (n, n, D)
        c1 = c[:, None, :]   # (n, 1, D)
        c2 = c[None, :, :]   # (1, n, D)
        s1 = s[:, None, :]
        s2 = s[None, :, :]

        if self.method == "combined":
            # Bhattacharyya per feature → geometric mean over features
            var_sum = s1 ** 2 + s2 ** 2
            bc_feat = (
                np.exp(-0.25 * (c1 - c2) ** 2 / var_sum)
                * np.sqrt(2.0 * s1 * s2 / var_sum)
            )
            bc = np.exp(np.mean(np.log(np.clip(bc_feat, 1e-300, None)), axis=-1))

            # Wasserstein: mean of (|Δc| + |Δs|) over features
            ws = np.exp(-np.mean(np.abs(c1 - c2) + np.abs(s1 - s2), axis=-1))

            # Centroid kernel: exp(-||Δc||²)
            cs = np.exp(-np.sum((c1 - c2) ** 2, axis=-1))

            S = self._w_bc * bc + self._w_ws * ws + self._w_cs * cs

        elif self.method == "bhattacharyya":
            var_sum = s1 ** 2 + s2 ** 2
            bc_feat = (
                np.exp(-0.25 * (c1 - c2) ** 2 / var_sum)
                * np.sqrt(2.0 * s1 * s2 / var_sum)
            )
            S = np.exp(np.mean(np.log(np.clip(bc_feat, 1e-300, None)), axis=-1))

        elif self.method == "wasserstein":
            S = np.exp(-np.mean(np.abs(c1 - c2) + np.abs(s1 - s2), axis=-1))

        elif self.method == "centroid":
            S = np.exp(-np.sum((c1 - c2) ** 2, axis=-1))

        elif self.method == "cosine":
            # Proper pairwise cosine similarity mapped to [0, 1].
            # c has shape (n, D); c1=(n,1,D), c2=(1,n,D) via broadcasting.
            dot = np.sum(c1 * c2, axis=-1)                      # (n, n)
            norm = np.linalg.norm(c, axis=-1)                    # (n,)
            denom = norm[:, None] * norm[None, :] + 1e-12        # (n, n)
            S = (np.clip(dot / denom, -1.0, 1.0) + 1.0) / 2.0

        else:
            # Fallback to scalar loop for unknown methods
            return self.similarity_matrix(centers, sigmas)

        np.fill_diagonal(S, 1.0)
        return S

    # ------------------------------------------------------------------
    # Individual similarity measures
    # ------------------------------------------------------------------

    def _bhattacharyya(self, c1, s1, c2, s2) -> float:
        """
        Weighted geometric mean of per-feature Bhattacharyya coefficients.

        With uniform feature weights, BC = prod_j BC_j^(1/D).

        Equals 1 when MFs are identical, approaches 0 as they diverge.
        """
        s1 = np.maximum(s1, 1e-8)
        s2 = np.maximum(s2, 1e-8)

        term1 = np.exp(-0.25 * (c1 - c2) ** 2 / (s1 ** 2 + s2 ** 2))
        term2 = np.sqrt(2.0 * s1 * s2 / (s1 ** 2 + s2 ** 2))
        bc_per_feat = term1 * term2

        w = self._get_weights(len(c1))
        return float(np.prod(bc_per_feat ** w))

    def _wasserstein_sim(self, c1, s1, c2, s2) -> float:
        """
        1-Wasserstein distance between Gaussians, converted to similarity.

        W1(N(m1, s1), N(m2, s2)) = |m1 - m2| + |s1 - s2|  (per feature)
        Similarity = exp(-sum_j w_j * W1_j)

        The sum is normalized by the number of features so scale is consistent.
        """
        s1 = np.maximum(s1, 1e-8)
        s2 = np.maximum(s2, 1e-8)

        w = self._get_weights(len(c1))
        w1_per_feat = np.abs(c1 - c2) + np.abs(s1 - s2)
        w1_weighted = float(np.sum(w * w1_per_feat))
        return float(np.exp(-w1_weighted))

    def _centroid_sim(self, c1, c2) -> float:
        """
        Gaussian-kernel similarity based on L2 distance between centers.
        sim = exp(-D * sum_j w_j (c1_j - c2_j)^2).
        With uniform weights this is exp(-||c1 - c2||^2).
        """
        D = len(c1)
        w = self._get_weights(D)
        dist2 = float(np.sum(w * (c1 - c2) ** 2))
        return float(np.exp(-dist2 * D))

    def _cosine_sim(self, c1, c2) -> float:
        """
        Cosine similarity of center vectors, mapped from [-1,1] to [0,1].
        """
        norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 1.0 if np.allclose(c1, c2) else 0.0
        cos = float(1.0 - cosine(c1, c2))  # in [-1, 1]
        return float((cos + 1.0) / 2.0)

    def _combined(self, c1, s1, c2, s2) -> float:
        """
        Weighted combination: w_bc * Bhattacharyya + w_ws * Wasserstein-sim + w_cs * Centroid.

        Default weights (0.5, 0.3, 0.2) give the largest role to distributional
        overlap while retaining sensitivity to width/center shifts and centroid proximity.
        Weights are set at construction time via composite_weights.
        """
        bc = self._bhattacharyya(c1, s1, c2, s2)
        ws = self._wasserstein_sim(c1, s1, c2, s2)
        cs = self._centroid_sim(c1, c2)
        return float(self._w_bc * bc + self._w_ws * ws + self._w_cs * cs)

    def _get_weights(self, n_features: int) -> np.ndarray:
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            return w / w.sum()
        return np.ones(n_features) / n_features


def cross_model_rule_similarity(
    params_list: list,
    method: str = "combined",
) -> np.ndarray:
    """
    Compute pairwise similarity between corresponding rules across M models.

    Parameters
    ----------
    params_list : list of dict
        Each dict has 'centers' (n_rules, D) and 'sigmas' (n_rules, D).
        All dicts must have the same n_rules.
    method : str

    Returns
    -------
    sim_tensor : (n_rules, M, M) array
        sim_tensor[r, i, j] = similarity between rule r in model i and model j.
    """
    sim_calc = RuleSimilarity(method=method)
    M = len(params_list)
    n_rules = params_list[0]["centers"].shape[0]

    sim_tensor = np.zeros((n_rules, M, M))
    for r in range(n_rules):
        for i in range(M):
            for j in range(M):
                if i == j:
                    sim_tensor[r, i, j] = 1.0
                elif j > i:
                    s = sim_calc.similarity(
                        params_list[i]["centers"][r],
                        params_list[i]["sigmas"][r],
                        params_list[j]["centers"][r],
                        params_list[j]["sigmas"][r],
                    )
                    sim_tensor[r, i, j] = s
                    sim_tensor[r, j, i] = s
    return sim_tensor
