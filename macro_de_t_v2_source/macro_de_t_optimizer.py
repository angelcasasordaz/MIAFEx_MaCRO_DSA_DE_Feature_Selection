"""CEC DE-MC-CF port; scientific methods retain the supplied source arithmetic.

Source chain: de_mc_cf_optimizer.DE_MC_CF -> de_mc_optimizer.DE_MC ->
de_ablation_base.MahalanobisDEBase. No dependency on FS MaCRO-DE or DSA-DE.
Only backend transport and public class identity are adapted.
"""
import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy.stats import chi2

from .macro_de_t_backend import MaCRODETBackend


class _CECMahalanobisDEBase(Optimizer):
    """Shared DE/rand/1/bin mechanics for MaCRO-DE ablation variants."""

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        wf=0.5,
        cr=0.9,
        mahalanobis_q=0.68,
        compute_device="cpu",
        gpu_device_id=0,
        gpu_memory_fraction=0.85,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0.0, 1.0))
        self.mahalanobis_q = self.validator.check_float(
            "mahalanobis_q",
            mahalanobis_q,
            (0.0, 1.0),
        )
        self.compute_device = compute_device
        self.backend = MaCRODETBackend(compute_device, gpu_device_id, gpu_memory_fraction)
        self._epoch_pop_pos = None
        self._epoch_close = None
        self._epoch_far = None
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "mahalanobis_q", "compute_device"])
        self.sort_flag = False
        self.support_parallel_modes = True

    def _positions(self, pop):
        return np.array([agent.solution for agent in pop], dtype=float)

    def _covariance_matrix(self, pop_pos):
        n_dims = self.problem.n_dims
        sigma = np.cov(pop_pos, rowvar=False)
        if np.ndim(sigma) == 0:
            sigma = np.array([[float(sigma)]], dtype=float)
        if sigma.shape != (n_dims, n_dims):
            sigma = np.eye(n_dims, dtype=float) * 1e-6
        return (sigma + sigma.T) / 2.0 + 1e-6 * np.eye(n_dims)

    def _covariance_inverse(self, sigma):
        raise NotImplementedError

    def _mahalanobis_dist2(self, pop_pos):
        _, _, dist2 = self.backend.mahalanobis_cpu(
            pop_pos,
            self.problem.n_dims,
            self.covariance_inverse_method,
        )
        return dist2

    @property
    def covariance_inverse_method(self):
        raise NotImplementedError

    def _mahalanobis_threshold(self):
        return chi2.ppf(self.mahalanobis_q, self.problem.n_dims)

    def _close_indices(self, pop_pos):
        close, _ = self._close_far_indices(pop_pos)
        return close

    def _close_far_indices(self, pop_pos):
        if pop_pos is self._epoch_pop_pos and self._epoch_close is not None:
            return self._epoch_close, self._epoch_far
        threshold = self._mahalanobis_threshold()
        return self.backend.close_far_indices(
            pop_pos,
            self.problem.n_dims,
            threshold,
            self.covariance_inverse_method,
            include_distances=False,
        )

    def _mutation_pool_indices(self, pop_pos, current_idx):
        raise NotImplementedError

    def _valid_candidates(self, pool_indices, current_idx):
        return pool_indices[pool_indices != current_idx]

    def _fallback_candidates(self, current_idx):
        return np.array(
            [idx for idx in range(self.pop_size) if idx != current_idx],
            dtype=int,
        )

    def _sample_mutation_indices(self, pop_pos, current_idx):
        pool_indices = self._mutation_pool_indices(pop_pos, current_idx)
        candidates = self._valid_candidates(pool_indices, current_idx)
        if candidates.size < 3:
            candidates = self._fallback_candidates(current_idx)
        return self.generator.choice(candidates, 3, replace=False)

    def _binomial_crossover(self, parent_pos, mutant_pos):
        trial = parent_pos.copy()
        j0 = self.generator.integers(0, self.problem.n_dims)
        cross_mask = self.generator.random(self.problem.n_dims) <= self.cr
        cross_mask[j0] = True
        trial[cross_mask] = mutant_pos[cross_mask]
        return self.correct_solution(trial)

    def evolve(self, epoch):
        pop_pos = self._positions(self.pop)
        # The population is fixed throughout this DE generation. Compute the
        # covariance/classification kernel once and return only compact indices.
        self._epoch_pop_pos = pop_pos
        self._epoch_close, self._epoch_far = self._close_far_indices(pop_pos)
        pop_new = []

        for idx in range(self.pop_size):
            idxs = self._sample_mutation_indices(pop_pos, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            mutant = self.correct_solution(x1 + self.wf * (x2 - x3))
            trial = self._binomial_crossover(self.pop[idx].solution, mutant)
            candidate = Agent(solution=trial)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(trial)
                self.pop[idx] = self.get_better_agent(
                    candidate,
                    self.pop[idx],
                    self.problem.minmax,
                )
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop,
                pop_new,
                self.problem.minmax,
            )
        self._epoch_pop_pos = None
        self._epoch_close = None
        self._epoch_far = None


class _CECDEMC(_CECMahalanobisDEBase):
    """DE-M with Cholesky-solve Mahalanobis distances."""

    def _covariance_inverse(self, sigma):
        n_dims = self.problem.n_dims
        try:
            chol = np.linalg.cholesky(sigma)
            return np.linalg.solve(
                chol.T,
                np.linalg.solve(chol, np.eye(n_dims)),
            )
        except np.linalg.LinAlgError:
            return np.linalg.pinv(sigma)

    def _mutation_pool_indices(self, pop_pos, current_idx):
        close = self._close_indices(pop_pos)
        if self._valid_candidates(close, current_idx).size >= 3:
            return close
        return np.arange(self.pop_size)

    @property
    def covariance_inverse_method(self):
        return "cholesky_solve"


class MaCRO_DE_t(_CECDEMC):
    """MaCRO-DE-t: independent interface port of CEC DE-MC-CF."""

    CANONICAL_NAME = "MaCRO-DE-t"
    IMPLEMENTATION_REVISION = "awad-close-far-v2"

    def initialize_variables(self):
        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_max_seen = None
        self.div_norm_for_update = 1.0
        self._awad_pair_indices = None
        self.routing_counts = {"close": 0, "far": 0, "fallback": 0}

    def before_main_loop(self):
        pop_pos = self._positions(self.pop)
        div0 = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_max_seen = max(div0, self.EPSILON)

    def _awad(self, pop_pos, lb, ub):
        """Match MaCRO-DE's AWAD calculation and safeguards exactly."""
        _ = lb, ub
        npop, n_dims = pop_pos.shape
        med_dim = np.median(pop_pos, axis=0)
        div_dim = np.mean(np.abs(pop_pos - med_dim), axis=0)
        div = float(np.sum(div_dim) / max(n_dims, 1))

        unique_count = np.unique(pop_pos, axis=0).shape[0]
        non_repeat_percent = (unique_count * 100.0) / max(npop, 1)

        std_devs = np.std(pop_pos, axis=0)
        std_devs[std_devs == 0] = 1e-5
        if npop <= 1:
            min_distance = 0.0
        else:
            pair_count = npop * (npop - 1) // 2
            if (
                self._awad_pair_indices is None
                or self._awad_pair_indices[0].size != pair_count
            ):
                self._awad_pair_indices = np.triu_indices(npop, k=1)
            left, right = self._awad_pair_indices
            diff = (pop_pos[right] - pop_pos[left]) / std_devs
            min_distance = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
            if not np.isfinite(min_distance):
                min_distance = 0.0

        penalty_factor = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return float(div * 0.1 * non_repeat_percent * penalty_factor)

    @staticmethod
    def _route_for_diversity(div_norm):
        return "close" if float(div_norm) >= 0.5 else "far"

    @classmethod
    def routing_diagnostic(cls):
        """Tiny side-effect-free check for the two diversity routing branches."""
        routes = {
            "high_diversity": cls._route_for_diversity(0.5),
            "low_diversity": cls._route_for_diversity(0.499999),
        }
        routes["passed"] = (
            routes["high_diversity"] == "close"
            and routes["low_diversity"] == "far"
        )
        return routes

    def _mutation_pool_indices(self, pop_pos, current_idx):
        close, far = self._close_far_indices(pop_pos)
        route = self._route_for_diversity(self.div_norm_for_update)
        selected = close if route == "close" else far
        if self._valid_candidates(selected, current_idx).size >= 3:
            self.routing_counts[route] += 1
            return selected
        self.routing_counts["fallback"] += 1
        return np.arange(self.pop_size)

    def evolve(self, epoch):
        if self.div_max_seen is None:
            self.before_main_loop()
        super().evolve(epoch)

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch - 1] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(
            np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0)
        )
        self.div_norm_hist[epoch - 1] = div_norm_now
        self.div_norm_for_update = div_norm_now

    @property
    def covariance_inverse_method(self):
        # Preserve this separate ablation's existing inverse-based arithmetic.
        return "cholesky"
