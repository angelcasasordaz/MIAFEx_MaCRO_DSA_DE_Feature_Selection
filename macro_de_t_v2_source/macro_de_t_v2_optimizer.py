"""MaCRO-DE-t-v2 with adaptive Mahalanobis-Cholesky control."""
import numpy as np
from mealpy.utils.agent import Agent
from .macro_de_t_optimizer import MaCRO_DE_t as DE_MC_CF


class DE_MC_CF_V2(DE_MC_CF):
    """Scale coordinate-wise F by MaCRO diversity D; adapt pcr from target dM."""

    IMPLEMENTATION_REVISION = "macro-d-scaled-coordinate-adaptive-pcr-v5"
    CANONICAL_NAME = "MaCRO-DE-t-v2"
    SCIENTIFIC_PARAMETERS = ("epoch", "pop_size", "beta_min", "beta_max", "mahalanobis_q")

    def __init__(self, epoch=1000, pop_size=50, beta_min=0.10, beta_max=0.60,
                 mahalanobis_q=0.50, compute_device="cpu", **kwargs):
        if {"wf", "cr", "pcr"}.intersection(kwargs):
            raise TypeError("MaCRO-DE-t-v2 uses beta bounds and adaptive pcr; wf/cr/pcr are unsupported")
        beta_min, beta_max = map(float, (beta_min, beta_max))
        if not (np.isfinite(beta_min) and np.isfinite(beta_max)
                and 0 <= beta_min <= beta_max):
            raise ValueError("Require finite 0 <= beta_min <= beta_max")
        super().__init__(epoch=epoch, pop_size=pop_size, cr=0.9,
                         mahalanobis_q=mahalanobis_q,
                         compute_device=compute_device, **kwargs)
        self.pcr = self.cr = 0.10
        self.beta_min, self.beta_max = beta_min, beta_max
        self.set_parameters(["epoch", "pop_size", "beta_min", "beta_max",
                             "mahalanobis_q", "compute_device"])

    def initialize_variables(self):
        super().initialize_variables()
        self.dm_hist = np.full((self.epoch, self.pop_size), np.nan)
        self.pcr_hist = np.full((self.epoch, self.pop_size), np.nan)
        self.f_hist = np.full((self.epoch, self.pop_size), np.nan)
        self.fmean_hist = np.full(self.epoch, np.nan)
        self.d_hist = np.full(self.epoch, np.nan)

    @staticmethod
    def _normalized_mahalanobis(dist2):
        """Normalize distances from the population mean to [0, 1] per generation."""
        distances = np.sqrt(np.maximum(dist2, 0.0))
        maximum = float(np.max(distances))
        return distances / maximum if maximum > 0.0 else np.zeros_like(distances)

    def _adaptive_control(self, dM, D):
        f = self._sample_scale_factors(D)
        pcr = 0.1 + 0.25 * (1.0 - dM)
        return f, pcr

    def _sample_scale_factors(self, D):
        """Use MaCRO-DE's multiplicative D control and coordinate clipping."""
        scale_f = float(np.clip(1.5 - D, 0.5, 1.5))
        draws = self.generator.uniform(self.beta_min, self.beta_max, self.problem.n_dims)
        return np.clip(draws * scale_f, 0.1, 1.5)

    # Frozen-generation mechanics copied from MahalanobisDEBase.evolve, with
    # only scalar wf replaced by an independent vector draw per mutation.
    # Historical description above refers to the superseded fixed-pcr version.
    # The following scalar-control note records the intermediate v3 behavior.
    # Active control now uses one scalar random draw and target Mahalanobis dM.
    # Active v4 restores independent coordinate-wise F; only pcr depends on dM.
    # Active v5 adds MaCRO-DE's D scaling/clipping to those coordinate draws.
    def evolve(self, epoch):
        if self.div_max_seen is None:
            self.before_main_loop()
        # Same delayed cumulative-max-normalized diversity D as MaCRO_DE.evolve.
        # Inherited AWAD bookkeeping supplies D; covariance distances define groups.
        D = float(np.clip(self.div_norm_for_update, 0.0, 1.0))
        self.d_hist[epoch - 1] = D
        pop_pos = self._positions(self.pop)
        # The population is fixed throughout this DE generation. Compute the
        # covariance/classification kernel once and return only compact indices.
        self._epoch_pop_pos = pop_pos
        self._epoch_close, self._epoch_far, dist2 = self.backend.close_far_indices(
            pop_pos, self.problem.n_dims, self._mahalanobis_threshold(),
            self.covariance_inverse_method, include_distances=True,
        )
        dM = self._normalized_mahalanobis(dist2)
        self.dm_hist[epoch - 1] = dM
        pop_new = []

        for idx in range(self.pop_size):
            idxs = self._sample_mutation_indices(pop_pos, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            f, self.pcr = self._adaptive_control(float(dM[idx]), D)
            self.cr = self.pcr  # Reuse the original forced-binomial-crossover method.
            self.f_hist[epoch - 1, idx] = float(np.mean(f))  # Per-target mean of coordinate scales.
            self.pcr_hist[epoch - 1, idx] = self.pcr
            mutant = self.correct_solution(x1 + f * (x2 - x3))
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
        self.fmean_hist[epoch - 1] = float(np.mean(self.f_hist[epoch - 1]))

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch - 1] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(
            np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0)
        )
        self.div_norm_hist[epoch - 1] = div_norm_now
        self.div_norm_for_update = div_norm_now


# Public FS identity; retain the CEC class name for source traceability.
MaCRO_DE_t_v2 = DE_MC_CF_V2
