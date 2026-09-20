"""Convergence validation/recovery with synthetic data; never execute optimizers."""
import copy
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main_best as f


def cached_row(ids=(0, 1, 2), *, evidence=False):
    finals = [0.1 + run * 0.01 for run in ids]
    metrics = [[80 + run for run in ids], [0.8] * len(ids), [0.8] * len(ids),
               [0.8] * len(ids), finals, [2] * len(ids), [1] * len(ids)]
    return f.build_label_payload(
        'knn', *metrics, [[0.5, 0.3, value] for value in finals], 3,
        completed_run_ids=list(ids),
        convergence_runs={run: f.convergence_metadata(3, value) for run, value in zip(ids, finals)} if evidence else None,
    )


class ConvergenceValidationTests(unittest.TestCase):
    def test_rejects_bad_shape_nonfinite_increases_and_wrong_final(self):
        for curve, best in (([], .1), ([.5, .1], .1), ([.5, .3, .2, .1], .1),
                            ([[.5, .3, .1]], .1), ([.5, float('nan'), .1], .1),
                            ([float('inf'), .3, .1], .1), ([.5, .2, .3], .3),
                            ([.5, .3, .1], .2), ([.5, .3, .1], float('nan'))):
            with self.subTest(curve=curve, best=best), self.assertRaises(ValueError):
                f.validate_convergence_curve(curve, 3, best)
        f.np.testing.assert_array_equal(f.validate_convergence_curve([5., 5., 4.], 3, 4.), [5., 5., 4.])

    def test_strict_mean_never_pads_or_truncates(self):
        curves = [[.8, .5, .1], [.7, .4, .2]]
        f.np.testing.assert_array_equal(f.pad_mean_curves(curves, 3), f.np.mean(f.np.stack(curves), axis=0))
        for invalid in ([], [[.5, .1]], [[.5, .3, .2, .1]], [[.5, float('nan'), .1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                f.pad_mean_curves(invalid, 3)

    def test_new_run_uses_independent_optimizer_best(self):
        args = f.parse_args(['--epochs', '3'])
        data = SimpleNamespace(X_train=f.np.ones((4, 2)), y_train=f.np.array([0, 1, 0, 1]))
        selector = Mock()
        selector.optimizer.history.list_global_best_fit = [.5, .3, .1]
        selector.optimizer.g_best.target.fitness = .2
        selector.transform.return_value = data.X_train
        selector.evaluate.return_value = {'AS_test': .8, 'PS_test': .8, 'RS_test': .8, 'F1S_test': .8}
        with patch.object(f, 'build_optimizer', return_value='OriginalPSO'), \
                patch.object(f, 'MhaSelector', return_value=selector):
            with self.assertRaisesRegex(ValueError, 'does not match final fitness'):
                f.run_single(data, 'knn', 'PSO', 'vstf_01', args, 1234)
            selector.evaluate.assert_not_called()
            selector.optimizer.g_best.target.fitness = .1
            result = f.run_single(data, 'knn', 'PSO', 'vstf_01', args, 1234)
        self.assertEqual(result['fit_final'], .1)
        self.assertEqual(result['convergence'], f.convergence_metadata(3, .1))
        self.assertEqual(result['convergence']['final_best_source'], 'optimizer.g_best.target.fitness')

    def test_legacy_and_out_of_order_new_rows(self):
        for evidence in (False, True):
            row = cached_row((2, 0, 1), evidence=evidence)
            self.assertEqual(row['CompletedRunIDs'], [0, 1, 2])
            self.assertEqual(f.cached_convergence_errors(row, 3), ({}, []))
            self.assertIs(f.prepare_cached_convergence(row, 3), row)
            self.assertEqual('ConvergenceRuns' in row, evidence)

    def test_invalid_run_only_is_removed_with_aligned_metrics_and_evidence(self):
        row = cached_row(evidence=True)
        row['CurvesAll'][1] = [.5, .3]
        before = copy.deepcopy(row)
        invalid, _ = f.cached_convergence_errors(row, 3)
        self.assertEqual(set(invalid), {1})
        repaired = f.prepare_cached_convergence(row, 3)
        self.assertEqual(repaired['CompletedRunIDs'], [0, 2])
        self.assertEqual(set(repaired['ConvergenceRuns']), {0, 2})
        for key in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs', 'FitRuns', 'FeatRuns', 'TimeRuns', 'CurvesAll'):
            f.np.testing.assert_array_equal(repaired[key], [before[key][0], before[key][2]])
        self.assertEqual(len(row['CurvesAll'][1]), 2)  # Input is not mutated.

    def test_missing_curve_and_bad_independent_evidence_are_identified(self):
        row = cached_row(evidence=True)
        row['CurvesAll'].pop()
        row['ConvergenceRuns'][1]['final_best'] = .25
        invalid, _ = f.cached_convergence_errors(row, 3)
        self.assertEqual(set(invalid), {1, 2})
        self.assertIn('final fitness', invalid[1])
        self.assertIn('missing individual curve', invalid[2])

    def test_mean_only_damage_needs_no_reruns(self):
        row = cached_row()
        row['Curve'] = f.np.array([.9, .8, .7])
        invalid, aggregate = f.cached_convergence_errors(row, 3)
        self.assertFalse(invalid)
        self.assertTrue(aggregate)
        repaired = f.prepare_cached_convergence(row, 3)
        self.assertEqual(repaired['CompletedRunIDs'], [0, 1, 2])
        self.assertNotIn('ConvergenceRuns', repaired)
        self.assertEqual(f.cached_convergence_errors(repaired, 3), ({}, []))


class ConvergenceResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = f.parse_args([
            '--dataset-name', 'Tiny', '--feature-dataset-root', str(root / 'features'),
            '--miafex-dataset-root', str(root / 'offline-images'), '--output-root', str(root),
            '--optimizers', 'PSO', '--estimators', 'knn', '--runs', '3', '--epochs', '3',
            '--pipeline-mode', 'full', '--parallel', 'no',
        ])
        self.args.optimizers = f.resolve_optimizers(self.args)
        metadata_args = copy.copy(self.args)
        metadata_args.figures_only = True
        self.scoped = f.resolve_miafex_dataset_args(metadata_args)
        for csv in f.miafex_feature_paths(self.scoped['Tiny']).values():
            Path(csv).parent.mkdir(parents=True, exist_ok=True)
            Path(csv).write_text('f0,f1,label\n1,2,0\n3,4,1\n')
        self.paths = f.make_paths(self.args)
        self.sig = f.build_cache_signature(self.scoped['Tiny'])
        self.label = f.build_alg_label(self.args.optimizers[0], 'vstf_01', 'knn', False, False)
        self.cache = f.cache_files(self.paths, 'Tiny', 'knn', self.sig)

    def write(self, row):
        for path in self.cache:
            f.save_cache(path, {self.label: row})

    def invoke(self, figures_only=False, parallel=False):
        self.args.figures_only = figures_only
        self.args.parallel = 'yes' if parallel else 'no'
        backend = f.ExecutionConfig('cpu', 'auto', 'sklearn', 'cpu', f.BackendAvailability(False, False, None, False, False))
        result = dict(as_test=81., ps_test=.8, rs_test=.8, f1_test=.8, fit_final=.11,
                      n_features=2, runtime=1., curve=[.5, .3, .11], convergence=f.convergence_metadata(3, .11))
        with ExitStack() as stack:
            for name in ('train_miafex', 'extract_miafex_features', 'build_optimizer'):
                stack.enter_context(patch.object(f, name, side_effect=AssertionError(name)))
            stack.enter_context(patch.object(f, 'parse_args', return_value=self.args))
            stack.enter_context(patch.object(f, 'resolve_execution_config', return_value=backend))
            stack.enter_context(patch.object(f, 'print_backend_report'))
            stack.enter_context(patch.object(f, 'export_reporting_outputs', return_value=([], 'summary.csv', [])))
            run = stack.enter_context(patch.object(f, 'run_single', return_value=result))
            def synthetic_parallel(data, estimator, method, tf, args, pending, **kwargs):
                for run_id in reversed(pending):
                    kwargs['on_run_complete'](run_id, result)
            pool = stack.enter_context(patch.object(f, 'execute_pending_runs', side_effect=synthetic_parallel))
            with redirect_stdout(io.StringIO()):
                f.main()
        return run, pool

    def test_normal_execution_reruns_only_invalid_id_and_preserves_legacy_rows(self):
        row = cached_row()
        row['CurvesAll'][1] = [.5, .3]
        self.write(row)
        before = copy.deepcopy(row)
        run, pool = self.invoke()
        run.assert_called_once()
        pool.assert_not_called()
        self.assertEqual(run.call_args.args[-1], self.args.seed_base + 1)
        stored = f.load_cache(self.cache[0])[self.label]
        self.assertEqual(stored['CompletedRunIDs'], [0, 1, 2])
        self.assertEqual(set(stored['ConvergenceRuns']), {1})
        self.assertEqual(f.cached_convergence_errors(stored, 3), ({}, []))
        for key in ('AccRuns', 'FitRuns', 'FeatRuns', 'TimeRuns', 'CurvesAll'):
            for index in (0, 2):
                f.np.testing.assert_array_equal(stored[key][index], before[key][index])
        self.assertEqual(f.build_cache_signature(self.scoped['Tiny']), self.sig)

    def test_parallel_recovery_keeps_valid_run_and_sorts_new_evidence(self):
        row = cached_row()
        row['CurvesAll'][0] = []
        row['CurvesAll'][2] = [.5, .3]
        self.write(row)
        run, pool = self.invoke(parallel=True)
        run.assert_not_called()
        self.assertEqual(pool.call_args.args[5], [0, 2])
        stored = f.load_cache(self.cache[0])[self.label]
        self.assertEqual(stored['CompletedRunIDs'], [0, 1, 2])
        self.assertEqual(set(stored['ConvergenceRuns']), {0, 2})
        self.assertEqual(f.cached_convergence_errors(stored, 3), ({}, []))

    def test_figures_only_rejects_invalid_cache_without_writes_or_reruns(self):
        row = cached_row()
        row['CurvesAll'][1] = [.5, .1, .11]
        self.write(row)
        snapshots = [(Path(p).read_bytes(), Path(p).stat().st_mtime_ns) for p in self.cache]
        with patch.object(f, 'save_cache', side_effect=AssertionError('No writes')), \
                self.assertRaisesRegex(ValueError, r'run 2:.*increases.*FIGURES_ONLY'):
            self.invoke(figures_only=True)
        self.assertEqual(snapshots, [(Path(p).read_bytes(), Path(p).stat().st_mtime_ns) for p in self.cache])

    def test_valid_legacy_complete_cache_is_not_rewritten(self):
        self.write(cached_row())
        with patch.object(f, 'save_cache', side_effect=AssertionError('No writes')):
            for figures_only in (False, True):
                run, pool = self.invoke(figures_only=figures_only)
                run.assert_not_called()
                pool.assert_not_called()
        self.assertNotIn('ConvergenceRuns', f.load_cache(self.cache[0])[self.label])

    def test_normal_execution_repairs_only_the_mean_without_reruns(self):
        row = cached_row()
        row['Curve'] = [.9, .8, .7]
        self.write(row)
        run, pool = self.invoke()
        run.assert_not_called()
        pool.assert_not_called()
        stored = f.load_cache(self.cache[0])[self.label]
        self.assertEqual(f.cached_convergence_errors(stored, 3), ({}, []))
        self.assertNotIn('ConvergenceRuns', stored)


if __name__ == '__main__':
    unittest.main()
