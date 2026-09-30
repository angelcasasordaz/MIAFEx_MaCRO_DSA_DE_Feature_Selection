"""Small synthetic runs verify the production path, masks, seeds and histories."""
import copy
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mafese.wrapper.mha as mha
from mealpy import TransferBinaryVar
from mealpy.human_based.BRO import OriginalBRO
from mealpy.optimizer import Optimizer
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

import main_best as f
from corrected_binary import BinaryRespawnBRO, CorrectedTransferBinaryVar, corrected_transfer_binary


class BinaryContractTests(unittest.TestCase):
    def test_adapter_methods_match_authoritative_implementation(self):
        root = Path(__file__).resolve().parents[1]
        def methods(path, classname):
            tree = ast.parse(path.read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
            return {node.name: ast.dump(node, include_attributes=False) for node in cls.body
                    if isinstance(node, ast.FunctionDef)}
        for classname in ('CorrectedTransferBinaryVar', 'BinaryRespawnBRO'):
            self.assertEqual(methods(root / 'corrected_binary.py', classname),
                             methods(root / 'diagnostics/corrected_binary_final/run_corrected.py', classname))

    def test_authoritative_results_and_source_fingerprints(self):
        root = Path(__file__).resolve().parents[1]
        reference = root / 'diagnostics/corrected_binary_final'
        manifest = json.loads((reference / 'manifest.json').read_text())
        for filename, expected in {**manifest['input_hashes'], **manifest['dependency_and_science_hashes']}.items():
            self.assertEqual(f.scientific_cache.file_digest(root / filename), expected, filename)
        self.assertEqual(f.BINARY_REPRESENTATION_REVISIONS['default'], manifest['binary_revision'])
        self.assertEqual(f.OPTIMIZER_REPAIR_REVISIONS['OriginalBRO'], manifest['bro_revision'])
        self.assertEqual(f.SEED_POLICY_REVISION, manifest['seed_revision'])
        results = json.loads((reference / 'results.json').read_text())
        self.assertEqual(len(results), 15)
        for result in results:
            self.assertEqual(result['curve'][-1], result['g_best_fitness'])
            self.assertEqual(result['g_best_fitness'], result['independent_g_best_fitness'])
            self.assertEqual(result['FitRun'], result['independent_g_best_fitness'])
            self.assertEqual(np.flatnonzero(result['winning_mask']).tolist(), result['winning_feature_indices'])
            self.assertEqual(sum(result['winning_mask']), result['selected_features'])
            self.assertEqual(len(result['curve']), result['epochs'])
            self.assertTrue(np.all(np.diff(result['curve']) <= 0))

    def test_bounds_decode_rng_and_proposal_sampling(self):
        var = CorrectedTransferBinaryVar(6, "my_var", "vstf_01", -8, 8, False)
        np.testing.assert_array_equal(var.lb, np.full(6, -8.))
        np.testing.assert_array_equal(var.ub, np.full(6, 8.))
        var.seed = 1234
        mask = np.array([1, 0, 1, 0, 0, 1])
        state = copy.deepcopy(var.generator.bit_generator.state)
        for _ in range(32):
            decoded = var.decode(mask)
            np.testing.assert_array_equal(decoded, mask)
            self.assertFalse(np.shares_memory(decoded, mask))
        self.assertEqual(state, var.generator.bit_generator.state)
        for bad in (np.zeros(6), np.full(6, .5), np.full(6, np.nan)):
            with self.assertRaises(ValueError):
                var.decode(bad)
        repaired = var.correct(np.linspace(-8, 8, 6))
        self.assertTrue(np.isin(repaired, [0, 1]).all() and repaired.any())
        self.assertNotEqual(state, var.generator.bit_generator.state)
        other = CorrectedTransferBinaryVar(n_vars=6, lb=-3, ub=5)
        np.testing.assert_array_equal(other.lb, np.full(6, -3.))
        np.testing.assert_array_equal(other.ub, np.full(6, 5.))

    def test_binding_restored_after_failure(self):
        original = mha.TransferBinaryVar
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with corrected_transfer_binary():
                self.assertIs(mha.TransferBinaryVar, CorrectedTransferBinaryVar)
                raise RuntimeError("interrupted")
        self.assertIs(mha.TransferBinaryVar, original)
        self.assertIs(original, TransferBinaryVar)

    def test_bro_respawn_repaired_before_target_without_changing_evolve(self):
        model = BinaryRespawnBRO(epoch=3, pop_size=10)
        evaluated = []

        def objective(solution):
            self.assertTrue(np.isin(solution, [0, 1]).all() and solution.any())
            evaluated.append(solution.copy())
            return float(solution.sum())

        model.check_problem({'bounds': CorrectedTransferBinaryVar(n_vars=6, all_zeros=False),
                             'minmax': 'min', 'obj_func': objective, 'log_to': None}, seed=1234)
        with patch.object(model, 'correct_solution', wraps=model.correct_solution) as repair:
            for solution in (np.linspace(-2, 3, 6), np.zeros(6), np.ones(6)):
                agent = model.generate_agent(solution)
                np.testing.assert_array_equal(evaluated[-1], agent.solution)
            self.assertEqual(repair.call_count, 2)
        self.assertIs(BinaryRespawnBRO.evolve, OriginalBRO.evolve)


class ProductionRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(n_samples=80, n_features=12, n_informative=5, random_state=41)
        cls.data = f.Data(X, y)
        cls.data.split_train_test(test_size=.2, random_state=42)
        cls.args = f.parse_args(['--epochs', '3', '--pop-size', '10', '--parallel', 'no'])

    def run_checked(self, method, estimator='knn', seed=1234):
        selectors = []

        class RecordingSelector(f.MhaSelector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                selectors.append(self)

        with patch.object(f, 'MhaSelector', RecordingSelector), \
                patch.object(Optimizer, 'solve', autospec=True, side_effect=Optimizer.solve) as solve:
            result = f.run_single(self.data, estimator, method, 'vstf_01', self.args, seed)
        self.assertEqual(solve.call_args.kwargs['seed'], seed)
        selector = selectors[0]
        opt = selector.optimizer
        self.assertEqual(selector.seed, seed)
        self.assertEqual(opt.problem.seed, seed)
        self.assertIsInstance(opt.problem.bounds[0], CorrectedTransferBinaryVar)
        np.testing.assert_array_equal(opt.problem.lb, np.full(12, -8.))
        np.testing.assert_array_equal(opt.problem.ub, np.full(12, 8.))
        mask = opt.g_best.solution
        np.testing.assert_array_equal(selector.selected_feature_solution, mask)
        np.testing.assert_array_equal(selector.selected_feature_indexes, np.flatnonzero(mask))
        for agent in opt.pop:
            self.assertTrue(np.isin(agent.solution, [0, 1]).all() and agent.solution.any())
        state = copy.deepcopy(opt.problem.bounds[0].generator.bit_generator.state)
        for _ in range(8):
            np.testing.assert_array_equal(opt.problem.decode_solution(mask)['my_var'], mask)
        self.assertEqual(state, opt.problem.bounds[0].generator.bit_generator.state)
        self.assertEqual(f.validate_selector_solution(selector, 12, seed), int(mask.sum()))
        inner = opt.problem.data
        columns = np.flatnonzero(mask)
        fresh = clone(opt.problem.estimator)
        with f.wrapper_resources():
            fresh.fit(np.take(inner.X_train, columns, axis=1), inner.y_train)
            accuracy = accuracy_score(inner.y_test, fresh.predict(np.take(inner.X_test, columns, axis=1)))
        independent = .9 * (1 - accuracy) + .1 * (mask.sum() / len(mask))
        self.assertEqual(result['fit_final'], independent)
        self.assertEqual(result['fit_final'], opt.g_best.target.fitness)
        self.assertEqual(result['n_features'], len(columns))
        np.testing.assert_array_equal(result['curve'], opt.history.list_global_best_fit)
        self.assertEqual(len(result['curve']), 3)
        return result, mask.copy()

    def test_all_original_optimizers_use_corrected_contract(self):
        for method in f.OPTIMIZERS:
            with self.subTest(method=method):
                self.run_checked(method)

    def test_pso_bro_repeatable_with_knn_and_svm(self):
        for method in ('PSO', 'BRO'):
            for estimator in ('knn', 'svm'):
                with self.subTest(method=method, estimator=estimator):
                    left, left_mask = self.run_checked(method, estimator, seed=1235)
                    np.random.seed(9999)
                    np.random.random(200)
                    right, right_mask = self.run_checked(method, estimator, seed=1235)
                    np.testing.assert_array_equal(left_mask, right_mask)
                    for key in ('curve', 'fit_final', 'n_features', 'as_test'):
                        np.testing.assert_array_equal(left[key], right[key])


class WinningMaskValidationTests(unittest.TestCase):
    def test_seed_mask_indices_and_scored_count_must_agree(self):
        from types import SimpleNamespace
        mask = np.array([1, 0, 1])
        selector = SimpleNamespace(
            seed=1234, selected_feature_solution=mask.copy(), selected_feature_masks=mask.astype(bool),
            selected_feature_indexes=np.array([0, 2]),
            optimizer=SimpleNamespace(
                problem=SimpleNamespace(seed=1234, decode_solution=lambda x: {'my_var': x.copy()}),
                g_best=SimpleNamespace(solution=mask, target=SimpleNamespace(objectives=[.1, .9, 2]))))
        self.assertEqual(f.validate_selector_solution(selector, 3, 1234), 2)
        for attribute, changed in [('seed', 1235), ('selected_feature_solution', np.array([0, 1, 1])),
                                   ('selected_feature_masks', np.array([True, True, False])),
                                   ('selected_feature_indexes', np.array([1, 2]))]:
            with self.subTest(attribute=attribute):
                invalid = copy.deepcopy(selector); setattr(invalid, attribute, changed)
                with self.assertRaises(ValueError): f.validate_selector_solution(invalid, 3, 1234)
        invalid = copy.deepcopy(selector); invalid.optimizer.g_best.target.objectives[2] = 1
        with self.assertRaisesRegex(ValueError, 'feature count'): f.validate_selector_solution(invalid, 3, 1234)


if __name__ == '__main__':
    unittest.main()
