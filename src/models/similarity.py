import numpy as np


class RuleSimilarity:
    """Pairwise similarity between Gaussian fuzzy rules."""

    def __init__(self, method="combined", composite_weights=None):
        self.method = method
        if composite_weights is None:
            self._w_bc, self._w_ws, self._w_cs = 0.5, 0.3, 0.2
        else:
            self._w_bc, self._w_ws, self._w_cs = composite_weights

    def similarity_matrix_vectorized(self, centers, sigmas):
        c = np.asarray(centers, dtype=np.float64)
        s = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-8)

        c1, c2 = c[:, None, :], c[None, :, :]
        s1, s2 = s[:, None, :], s[None, :, :]

        if self.method == "combined":
            var_sum = s1 ** 2 + s2 ** 2
            bc = np.exp(np.mean(np.log(np.clip(
                np.exp(-0.25 * (c1 - c2) ** 2 / var_sum) * np.sqrt(2 * s1 * s2 / var_sum),
                1e-300, None)), axis=-1))
            ws = np.exp(-np.mean(np.abs(c1 - c2) + np.abs(s1 - s2), axis=-1))
            cs = np.exp(-np.sum((c1 - c2) ** 2, axis=-1))
            S = self._w_bc * bc + self._w_ws * ws + self._w_cs * cs
        elif self.method == "bhattacharyya":
            var_sum = s1 ** 2 + s2 ** 2
            S = np.exp(np.mean(np.log(np.clip(
                np.exp(-0.25 * (c1 - c2) ** 2 / var_sum) * np.sqrt(2 * s1 * s2 / var_sum),
                1e-300, None)), axis=-1))
        elif self.method == "wasserstein":
            S = np.exp(-np.mean(np.abs(c1 - c2) + np.abs(s1 - s2), axis=-1))
        else:  # centroid
            S = np.exp(-np.sum((c1 - c2) ** 2, axis=-1))

        np.fill_diagonal(S, 1.0)
        return S
