"""CEC covariance kernels, transported through the existing GPU owner service.

Only the numerical methods from the authoritative compute_backend.py are
ported. AWAD and all optimizer RNG/control remain on CPU, as in DE_MC_CF.
"""
import numpy as np


class CECCovarianceKernels:
    def __init__(self, backend):
        self.backend = backend
        self.xp = backend.xp

    def asarray(self, value):
        return self.xp.asarray(value, dtype=self.xp.float64)

    def to_cpu(self, value):
        return self.backend.to_cpu(value)

    def covariance(self, population, n_dims: int):
        xp = self.xp
        pop = self.asarray(population)
        if pop.ndim == 2:
            sigma = xp.cov(pop, rowvar=False)
        elif pop.ndim == 3:
            centered = pop - xp.mean(pop, axis=1, keepdims=True)
            denominator = max(pop.shape[1] - 1, 1)
            sigma = xp.matmul(centered.swapaxes(-1, -2), centered) / denominator
        else:
            raise ValueError("Population must have shape (population, dims) or (batch, population, dims).")
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1)
        expected_tail = (n_dims, n_dims)
        if sigma.shape[-2:] != expected_tail:
            leading = pop.shape[:-2]
            sigma = xp.broadcast_to(
                xp.eye(n_dims, dtype=xp.float64) * 1e-6,
                (*leading, n_dims, n_dims),
            ).copy()
        identity = xp.eye(n_dims, dtype=xp.float64)
        return (sigma + sigma.swapaxes(-1, -2)) / 2.0 + 1e-6 * identity

    def covariance_inverse(self, sigma, method: str):
        xp = self.xp
        n_dims = sigma.shape[-1]
        try:
            if method == "direct":
                return xp.linalg.inv(sigma)
            if method in {"cholesky", "cholesky_solve"}:
                chol = xp.linalg.cholesky(sigma)
                identity = xp.broadcast_to(
                    xp.eye(n_dims, dtype=xp.float64), sigma.shape
                )
                return xp.linalg.solve(
                    chol.swapaxes(-1, -2), xp.linalg.solve(chol, identity)
                )
            raise ValueError(f"Unknown covariance inverse method: {method}")
        except xp.linalg.LinAlgError:
            return xp.linalg.pinv(sigma)

    def covariance_factor(self, sigma, method: str):
        """Return one reusable factor and how distances must consume it."""
        xp = self.xp
        try:
            if method == "direct":
                return xp.linalg.inv(sigma), "inverse"
            if method == "cholesky_solve":
                return xp.linalg.cholesky(sigma), "cholesky"
            if method == "cholesky":
                return self.covariance_inverse(sigma, "cholesky"), "inverse"
            raise ValueError(f"Unknown covariance inverse method: {method}")
        except xp.linalg.LinAlgError:
            return xp.linalg.pinv(sigma), "inverse"

    def distances_from_factor(self, population, factor, factor_kind: str):
        """Compute squared Mahalanobis distances from an existing factorization."""
        xp = self.xp
        pop = self.asarray(population)
        diff = pop - xp.mean(pop, axis=-2, keepdims=True)
        if factor_kind == "cholesky":
            whitened = xp.linalg.solve(factor, diff.swapaxes(-1, -2))
            return xp.sum(whitened * whitened, axis=-2)
        if factor_kind == "inverse":
            return xp.sum(xp.matmul(diff, factor) * diff, axis=-1)
        raise ValueError(f"Unknown covariance factor kind: {factor_kind}")

    def mahalanobis(self, population, n_dims: int, method: str = "cholesky"):
        """Return covariance, Cholesky (if valid), distances, all on this backend."""
        xp = self.xp
        pop = self.asarray(population)
        sigma = self.covariance(pop, n_dims)
        if method == "cholesky":
            try:
                chol = xp.linalg.cholesky(sigma)
            except xp.linalg.LinAlgError:
                chol = None
            sigma_inv = self.covariance_inverse(sigma, method)
            dist2 = self.distances_from_factor(pop, sigma_inv, "inverse")
            return sigma, chol, dist2
        factor, factor_kind = self.covariance_factor(sigma, method)
        chol = factor if factor_kind == "cholesky" else None
        dist2 = self.distances_from_factor(pop, factor, factor_kind)
        return sigma, chol, dist2

    def mahalanobis_distances(self, population, n_dims: int, method: str = "cholesky"):
        """Distance-only path that avoids the unused/duplicate Cholesky factor."""
        xp = self.xp
        pop = self.asarray(population)
        sigma = self.covariance(pop, n_dims)
        factor, factor_kind = self.covariance_factor(sigma, method)
        return self.distances_from_factor(pop, factor, factor_kind)

    def mahalanobis_cpu(self, population, n_dims: int, method: str = "cholesky"):
        sigma, chol, dist2 = self.mahalanobis(population, n_dims, method)
        return (
            self.to_cpu(sigma),
            None if chol is None else self.to_cpu(chol),
            self.to_cpu(dist2),
        )

    def close_far_indices(
        self,
        population,
        n_dims: int,
        threshold: float,
        method: str,
        include_distances: bool = True,
    ):
        """Transfer only final small index arrays back to the CPU boundary."""
        dist2 = self.mahalanobis_distances(population, n_dims, method)
        if include_distances:
            dist2_cpu = self.to_cpu(dist2)
            close_mask = dist2_cpu <= threshold
            return np.flatnonzero(close_mask), np.flatnonzero(~close_mask), dist2_cpu
        close_mask = self.to_cpu(dist2 <= threshold).astype(bool, copy=False)
        return np.flatnonzero(close_mask), np.flatnonzero(~close_mask)

    def close_far_masks(
        self,
        population,
        n_dims: int,
        threshold: float,
        method: str,
    ):
        """Return backend-resident masks without flattening a run batch."""
        dist2 = self.mahalanobis_distances(population, n_dims, method)
        return dist2 <= threshold, dist2 > threshold, dist2



class MaCRODETBackend:
    def __init__(self, compute_device="cpu", gpu_device_id=0, gpu_memory_fraction=0.85):
        from .transport import DiversityMathBatcher
        self.batcher = DiversityMathBatcher(compute_device, gpu_device_id, gpu_memory_fraction)

    def mahalanobis_cpu(self, population, n_dims, method):
        return self.batcher.macro_de_t_covariance("mahalanobis_cpu", population, n_dims, method)

    def close_far_indices(self, population, n_dims, threshold, method, include_distances=True):
        return self.batcher.macro_de_t_covariance(
            "close_far_indices", population, n_dims, threshold, method, include_distances,
        )
