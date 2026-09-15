"""Deterministic CEC parity and integration checks; no dataset experiments.

Run: python -m unittest discover -s tests -p test_optimizer_integrations.py -v
CEC_REFERENCE_DIR may point to the supplied CEC directory. The default is the
sibling project; a hash-verified source snapshot makes the tests portable.
"""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from mealpy import FloatVar, TransferBinaryVar
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target
from scipy.stats import chi2

import algorithm_acronym_list as registry
import main_best as framework
from dsade_awad_optimizer import DSADE
from macro_de_t_optimizer import MaCRO_DE_t


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(os.environ.get(
    "CEC_REFERENCE_DIR",
    ROOT.parent / "Adaptive_Mahalanobis-Cholesky_DIfferential_Evolution",
))
SOURCE_HASHES = {
    "compute_backend": "a79aa07a8a53a8a2aa7a149a09e7174fd89bd346c0e26664e33c08830f036850",
    "de_ablation_base": "ae7ab5ef49eee6a5c536b56e5117ba7c6695562b9a590263e2477144a8d610f0",
    "de_mc_optimizer": "424907cd4c97bf84540851a45775e83583c6cc21706fd81cdce902dc4a1a6750",
    "de_mc_cf_optimizer": "0501f80af33f3bf6d5926b45d88fa545afb261ab3c487697ca8d914c6c42f5eb",
}


def original_imports(source):
    for module in SOURCE_HASHES:
        source = source.replace(f"from .{module} import", f"from {module} import")
    return source


def load_cec_reference():
    modules = {}
    # Isolate original absolute imports from the framework's existing MaCRO-DE.
    with patch.dict(sys.modules):
        for name, digest in SOURCE_HASHES.items():
            path = REFERENCE_DIR / f"{name}.py"
            if REFERENCE_DIR.is_dir() or "CEC_REFERENCE_DIR" in os.environ:
                source = path.read_text()
            else:
                path = ROOT / "cec_de_mc_cf" / f"{name}.py"
                source = original_imports(path.read_text())
            if hashlib.sha256(source.encode()).hexdigest() != digest:
                raise AssertionError(f"CEC source changed: {path}")
            module = ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = modules[name] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
    return modules


CEC = load_cec_reference()
Reference = CEC["de_mc_cf_optimizer"].DE_MC_CF


def sphere(vector):
    return float(np.sum(np.asarray(vector) ** 2))


def prepared(cls, positions, *, mode="single", seed=1234, epochs=1, q=0.68,
             binary=False):
    positions = np.asarray(positions, dtype=float)
    n_dims = positions.shape[1]
    model = cls(epoch=epochs, pop_size=len(positions), mahalanobis_q=q)
    bounds = (TransferBinaryVar(n_vars=n_dims, tf_func="vstf_01", lb=-8, ub=8,
                                all_zeros=False, name="my_var") if binary else
              FloatVar(lb=(-5.0,) * n_dims, ub=(5.0,) * n_dims))
    model.check_problem(dict(bounds=bounds, minmax="min", obj_func=sphere, log_to=None), seed)
    model.check_mode_and_workers(mode, None)
    model.initialize_variables()
    model.pop = [Agent(solution=p.copy(), target=Target(sphere(p))) for p in positions]
    model.before_main_loop()
    return model


class RecordingGenerator:
    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.calls = {"choice": [], "integers": [], "random": []}

    def __getattr__(self, name):
        method = getattr(self.generator, name)
        if name not in self.calls:
            return method

        def record(*args, **kwargs):
            value = method(*args, **kwargs)
            self.calls[name].append(np.asarray(value).copy())
            return value
        return record


class TraceMixin:
    def before_main_loop(self):
        super().before_main_loop()
        self.donors, self.snapshots, self.mutants, self.trials = [], [], [], []

    def _sample_mutation_indices(self, pop_pos, current_idx):
        selected = super()._sample_mutation_indices(pop_pos, current_idx)
        self.donors.append(selected.copy())
        self.snapshots.append(pop_pos.copy())
        return selected

    def _binomial_crossover(self, parent_pos, mutant_pos):
        self.mutants.append(mutant_pos.copy())
        trial = super()._binomial_crossover(parent_pos, mutant_pos)
        self.trials.append(trial.copy())
        return trial


class TraceAdapter(TraceMixin, MaCRO_DE_t):
    pass


class TraceReference(TraceMixin, Reference):
    pass


class CECParityTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.random.default_rng(991).normal(size=(10, 5))
        self.models = [prepared(cls, self.positions) for cls in (MaCRO_DE_t, Reference)]

    def test_supplied_sources_unchanged_except_package_imports(self):
        for name, digest in SOURCE_HASHES.items():
            source = original_imports((ROOT / "cec_de_mc_cf" / f"{name}.py").read_text())
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digest, name)

    def test_awad_duplicates_constant_dimensions_singleton_and_safeguards(self):
        populations = [self.positions, np.ones((10, 5)), np.array([[0.], [0.], [1.], [1.]]),
                       np.ones((1, 5)), np.c_[self.positions[:, 0], np.ones(10)],
                       np.array([[0.], [np.inf]])]
        for positions in populations:
            with self.subTest(shape=positions.shape), np.errstate(all="ignore"):
                values = [m._awad(positions, -5, 5) for m in self.models]
                np.testing.assert_allclose(values[0], values[1], rtol=0, atol=0)
        self.assertEqual(self.models[0]._awad(np.ones((10, 5)), -5, 5), 0.)
        self.assertAlmostEqual(self.models[0]._awad(populations[2], -5, 5), 0.025)

    def test_cumulative_max_normalization_and_delayed_route(self):
        for cls in (MaCRO_DE_t, Reference):
            model = prepared(cls, self.positions, epochs=3)
            base = next(c for c in cls.__mro__ if c.__name__ == "MahalanobisDEBase")
            with patch.object(model, "_awad", side_effect=[2., 1., 4., 1.]), \
                    patch.object(base, "evolve", return_value=None):
                model.before_main_loop()
                routes = []
                for epoch in range(1, 4):
                    routes.append(model._route_for_diversity(model.div_norm_for_update))
                    model.evolve(epoch)
            np.testing.assert_allclose(model.div_norm_hist, [.5, 1., .25])
            np.testing.assert_array_equal(model.div_awad_hist, [1., 4., 1.])
            self.assertEqual(model.div_max_seen, 4.)
            self.assertEqual(routes, ["close", "far", "close"])
            self.assertEqual(model._route_for_diversity(model.div_norm_for_update), "far")
            zero = prepared(cls, np.zeros((10, 5)))
            zero.evolve(1)
            self.assertEqual(zero.div_norm_hist[0], 0.)

    def test_routing_boundary_fallback_and_donor_eligibility(self):
        # Pool size is checked AFTER excluding the target (3 including target fails).
        scenarios = [(0.5, [0, 1, 2, 3], [4, 5, 6, 7, 8, 9], 0, "close"),
                     (0.499999, [0, 1, 2, 3], [4, 5, 6, 7, 8, 9], 4, "far"),
                     (1., [0, 1, 2], list(range(3, 10)), 0, "fallback"),
                     (0., list(range(7)), [7, 8, 9], 7, "fallback"),
                     (1., [], list(range(10)), 0, "fallback")]
        for diversity, close, far, target, route in scenarios:
            donors = []
            for model in self.models:
                model.div_norm_for_update = diversity
                model.generator = np.random.default_rng(88)
                previous = model.routing_counts[route]
                with patch.object(model, "_close_far_indices", return_value=(np.array(close), np.array(far))):
                    selected = model._sample_mutation_indices(self.positions, target)
                self.assertEqual(model.routing_counts[route], previous + 1)
                eligible = (close if route == "close" else far) if route != "fallback" else range(10)
                self.assertTrue(set(selected).issubset(set(eligible) - {target}))
                self.assertEqual(len(set(selected)), 3)
                donors.append(selected)
                # Exercise the base sampler's independent fallback safeguard too.
                with patch.object(model, "_mutation_pool_indices", return_value=np.array([target])):
                    selected = model._sample_mutation_indices(self.positions, target)
                    self.assertNotIn(target, selected)
                    self.assertEqual(len(set(selected)), 3)
            np.testing.assert_array_equal(*donors)

    def test_covariance_cholesky_threshold_and_pseudoinverse_fallback(self):
        for positions in (self.positions, np.ones((10, 5)), self.positions[:, :1]):
            models = [prepared(cls, positions) for cls in (MaCRO_DE_t, Reference)]
            outputs = []
            for model in models:
                dims = positions.shape[1]
                expected_cov = np.atleast_2d(np.cov(positions, rowvar=False)) + 1e-6 * np.eye(dims)
                np.testing.assert_array_equal(model._covariance_matrix(positions), expected_cov)
                with patch.object(np.linalg, "solve", wraps=np.linalg.solve) as solve:
                    inverse = model.backend.covariance_inverse(expected_cov, "cholesky")
                    self.assertEqual(solve.call_count, 2)
                np.testing.assert_allclose(inverse, np.linalg.inv(expected_cov), rtol=1e-12, atol=1e-12)
                self.assertEqual(model.covariance_inverse_method, "cholesky")
                self.assertEqual(model._mahalanobis_threshold(), chi2.ppf(.68, dims))
                sigma, chol, distances = model.backend.mahalanobis_cpu(positions, dims, "cholesky")
                np.testing.assert_array_equal(sigma, expected_cov)
                np.testing.assert_allclose(chol @ chol.T, sigma, atol=1e-14)
                delta = positions - positions.mean(axis=0)
                np.testing.assert_allclose(distances, np.sum((delta @ inverse) * delta, axis=1))
                close, far = model._close_far_indices(positions)
                np.testing.assert_array_equal(close, np.flatnonzero(distances <= chi2.ppf(.68, dims)))
                np.testing.assert_array_equal(far, np.flatnonzero(distances > chi2.ppf(.68, dims)))
                with patch.object(np.linalg, "cholesky", side_effect=np.linalg.LinAlgError):
                    fallback = model.backend.covariance_inverse(sigma, "cholesky")
                    np.testing.assert_array_equal(fallback, np.linalg.pinv(sigma))
                    np.testing.assert_array_equal(model._covariance_inverse(sigma), fallback)
                    fallback_dist = model._mahalanobis_dist2(positions)
                    np.testing.assert_allclose(fallback_dist, np.sum((delta @ fallback) * delta, axis=1))
                    fallback_pools = model._close_far_indices(positions)
                    np.testing.assert_array_equal(fallback_pools[0], np.flatnonzero(fallback_dist <= chi2.ppf(.68, dims)))
                outputs.append((distances, close, far))
            for left, right in zip(*outputs):
                np.testing.assert_array_equal(left, right)
        self.models[0].mahalanobis_q = .95
        self.assertEqual(self.models[0]._mahalanobis_threshold(), chi2.ppf(.95, 5))

    def test_forced_crossover_coordinate(self):
        for model in self.models:
            model.cr = 0.0
            with patch.object(model, "generator") as generator:
                generator.integers.return_value = 2
                generator.random.return_value = np.ones(5)
                trial = model._binomial_crossover(np.zeros(5), np.ones(5))
            np.testing.assert_array_equal(trial, [0, 0, 1, 0, 0])

    def test_one_generation_donors_mutation_crossover_frozen_greedy_outcome(self):
        for mode in ("single", "swarm"):
            for diversity in (1., .499999):
                with self.subTest(mode=mode, diversity=diversity):
                    models = [prepared(cls, self.positions, mode=mode) for cls in (TraceAdapter, TraceReference)]
                    for model in models:
                        model.div_norm_for_update = diversity
                        model.generator = RecordingGenerator(1234)
                        with patch.object(model.backend, "close_far_indices", wraps=model.backend.close_far_indices) as classify:
                            model.evolve(1)
                            self.assertEqual(classify.call_count, 1)
                        self.assertEqual((model.wf, model.cr), (.5, .9))
                        for i, donors in enumerate(model.donors):
                            np.testing.assert_array_equal(model.snapshots[i], self.positions)
                            self.assertNotIn(i, donors)
                            self.assertEqual(len(set(donors)), 3)
                            a, b, c = self.positions[donors]
                            mutant = np.clip(a + .5 * (b - c), -5., 5.)
                            np.testing.assert_array_equal(model.mutants[i], mutant)
                            mask = model.generator.calls["random"][i] <= .9
                            mask[int(model.generator.calls["integers"][i])] = True
                            np.testing.assert_array_equal(model.trials[i], np.where(mask, mutant, self.positions[i]))
                        parent_fit = np.array([sphere(p) for p in self.positions])
                        trial_fit = np.array([sphere(p) for p in model.trials])
                        self.assertTrue(np.any(trial_fit < parent_fit))
                        self.assertTrue(np.any(trial_fit >= parent_fit))
                        expected = np.where((trial_fit < parent_fit)[:, None], model.trials, self.positions)
                        np.testing.assert_array_equal(model._positions(model.pop), expected)
                        np.testing.assert_array_equal([a.target.fitness for a in model.pop], np.minimum(parent_fit, trial_fit))
                        self.assertIsNone(model._epoch_pop_pos)
                        self.assertIsNone(model._epoch_close)
                        self.assertIsNone(model._epoch_far)
                    for attr in ("donors", "mutants", "trials", "div_awad_hist", "div_norm_hist"):
                        np.testing.assert_array_equal(getattr(models[0], attr), getattr(models[1], attr))
                    self.assertEqual(models[0].routing_counts, models[1].routing_counts)

    def test_transfer_binary_interface_preserves_cec_one_generation(self):
        positions = (self.positions > 0).astype(float)
        positions[:, 0] = 1.
        models = [prepared(cls, positions, binary=True) for cls in (MaCRO_DE_t, Reference)]
        for model in models:
            model.evolve(1)
            for agent in model.pop:
                self.assertTrue(np.isin(agent.solution, [0, 1]).all())
                self.assertGreater(np.count_nonzero(agent.solution), 0)
        np.testing.assert_array_equal(models[0]._positions(models[0].pop), models[1]._positions(models[1].pop))


class DSADECanonicalTests(unittest.TestCase):
    def test_all_scientific_methods_identical_to_original_awad_source(self):
        tree = ast.parse((ROOT / "dsade_awad_optimizer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = "\n".join(ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef))
        self.assertEqual(hashlib.sha256(methods.encode()).hexdigest(),
                         "d061fd6b985f934ee1a5e402abff9d61b1b059755c3fe11fd3b66e0e7dcc2ca4")

    def test_awad_survivor_can_accept_worse_fitness_for_diversity(self):
        model = prepared(DSADE, np.zeros((10, 2)))
        parent = Agent(solution=np.zeros(2), target=Target(0.))
        offspring = Agent(solution=np.ones(2), target=Target(1.))
        self.assertIs(model.diversity_selection(parent, offspring, np.zeros((9, 2))), offspring)
        better = Agent(solution=np.zeros(2), target=Target(-1.))
        self.assertIs(model.diversity_selection(parent, better, np.zeros((9, 2))), better)


class OptimizerIntegrationTests(unittest.TestCase):
    def args(self, name, source="mafese"):
        return framework.parse_args(["--dataset-source", source, "--optimizers", name,
                                     "--fs-epochs", "1", "--pop-size", "10"])

    def test_resolution_registry_and_mafese_acceptance_for_both_sources(self):
        for source in ("miafex", "mafese"):
            for name, cls in (("DSADE", DSADE), ("DSA-DE", DSADE), ("DSA_DE", DSADE),
                              ("MaCRO-DE-t", MaCRO_DE_t)):
                with self.subTest(source=source, name=name):
                    args = self.args(name, source)
                    model = framework.build_optimizer(name, args)
                    self.assertIs(type(model), cls)
                    selector = framework.MhaSelector(problem="classification", estimator="knn", optimizer=model)
                    self.assertIs(selector._set_optimizer(model), model)
                    self.assertEqual(registry.optimizer_acronym(name), "DSADE" if cls is DSADE else "MaCRO-DE-t")
                    if cls is MaCRO_DE_t:
                        self.assertEqual((model.wf, model.cr, model.mahalanobis_q), (.5, .9, .68))
                        self.assertEqual(model.compute_device, "cpu")
                    else:
                        self.assertEqual(cls.__module__, "dsade_awad_optimizer")
        for name in ("DSADE_AWAD", "DSADE-AWAD"):
            self.assertNotIn(name, registry.CUSTOM_OPTIMIZERS)
            with self.assertRaises(ValueError):
                framework.build_optimizer(name, self.args(name))
        self.assertNotIn("MaCRO-DE-t", framework.OPTIMIZERS)
        self.assertIs(type(framework.build_optimizer("MaCRO-DE", self.args("MaCRO-DE"))), framework.MaCRO_DE)

    def test_revisions_change_scientific_cache_identity_for_both_sources(self):
        for source in ("miafex", "mafese"):
            signatures = []
            for name, cls, revision in (("DSADE", DSADE, "awad-survivor-v2"),
                                        ("DSA-DE", DSADE, "awad-survivor-v2"),
                                        ("DSA_DE", DSADE, "awad-survivor-v2"),
                                        ("MaCRO-DE-t", MaCRO_DE_t, "awad-close-far-v2")):
                args = self.args(name, source)
                canonical = registry.resolve_optimizer_name(name)
                self.assertEqual(framework.optimizer_implementation_revisions(args), {canonical: revision})
                signature = framework.build_cache_signature(args)
                signatures.append(signature)
                with patch.object(cls, "IMPLEMENTATION_REVISION", "different-science"):
                    self.assertNotEqual(signature, framework.build_cache_signature(args))
                with patch.object(framework, "optimizer_implementation_revisions", return_value={}):
                    self.assertNotEqual(signature, framework.build_cache_signature(args))
                self.assertEqual(framework.legacy_cache_signatures(args), [])
            self.assertNotEqual(signatures[0], signatures[-1])
        macro = self.args("MaCRO-DE")
        self.assertEqual(framework.optimizer_implementation_revisions(macro), {})
        self.assertTrue(framework.legacy_cache_signatures(macro))

    def test_reject_old_greedy_and_unversioned_caches_in_all_lookup_paths(self):
        for name in ("DSADE", "DSA-DE", "DSA_DE", "MaCRO-DE-t"):
            for source in ("miafex", "mafese"):
                with self.subTest(name=name, source=source), tempfile.TemporaryDirectory() as directory:
                    args = self.args(name, source)
                    args.output_root, args.exp_id, args.reuse_cache_from_exp_id = directory, 900, 899
                    paths = framework.make_paths(args)
                    source_paths = framework.make_paths(args, exp_id=899)
                    signature = framework.build_cache_signature(args)
                    with patch.object(framework, "optimizer_implementation_revisions", return_value={}):
                        obsolete = [framework.build_cache_signature(args), *framework.legacy_cache_signatures(args)]
                    for location in (paths, source_paths):
                        for old in obsolete:
                            for filename in framework.cache_files(location, "Tiny", "knn", old):
                                framework.save_cache(filename, {"obsolete": {"CompletedRuns": 1}})
                    for figures_only in (False, True):
                        with redirect_stdout(io.StringIO()):
                            payload = framework.resolve_cached_payload(paths, args, "Tiny", "knn", signature,
                                                                       figures_only=figures_only)
                        self.assertIsNone(payload)
                    current_file, progress_file = framework.cache_files(paths, "Tiny", "knn", signature)
                    self.assertFalse(Path(current_file).exists())
                    valid = {"current": {"CompletedRuns": 1}}
                    framework.save_cache(progress_file, valid)
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(framework.resolve_cached_payload(paths, args, "Tiny", "knn", signature), valid)

    def test_old_greedy_module_removed(self):
        self.assertFalse((ROOT / "dsade_optimizer.py").exists())
        self.assertIsNone(framework.importlib.util.find_spec("dsade_optimizer"))


if __name__ == "__main__":
    unittest.main()
