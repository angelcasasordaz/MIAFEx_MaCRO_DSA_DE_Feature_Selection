"""Reporting and CPU transport tests with synthetic caches; no scientific runs."""
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, redirect_stdout
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import main_best as framework


def worker_probe(value):
    """Spawn-safe diagnostic task; never invokes an optimizer or classifier."""
    data = framework._RUN_WORKER_DATA_SPLIT
    return os.getpid(), id(data["X_train"]), data["X_train"].flags.writeable, data["X_train"].sum(), value


class ComparisonTests(unittest.TestCase):
    def test_friedman_ranks_and_exact_wilcoxon_holm(self):
        matrix = pd.DataFrame({"DE": np.arange(8), "PSO": np.arange(8) + 1, "GWO": np.arange(8) + 2})
        analysis = framework.calculate_friedman_analysis(matrix, "FULL")
        summary = analysis["summary"].iloc[0]
        self.assertAlmostEqual(summary["Friedman statistic"], 16)
        self.assertAlmostEqual(summary["p-value"], np.exp(-8))
        self.assertEqual(summary["Significant"], "YES")
        self.assertEqual(list(analysis["ranks"]["Average rank"]), [1, 2, 3])
        self.assertEqual(len(analysis["posthoc"]), 3)
        np.testing.assert_allclose(analysis["posthoc"]["Raw p-value"], 1 / 128)
        np.testing.assert_allclose(analysis["posthoc"]["Holm adjusted p-value"], 3 / 128)
        self.assertTrue((analysis["posthoc"]["Significant at alpha=0.05"] == "YES").all())

    def test_holm_restores_pair_order_and_is_monotone(self):
        np.testing.assert_allclose(framework._holm_adjusted_pvalues([0.04, 0.001, 0.03, 0.9]),
                                   [0.09, 0.004, 0.09, 0.9])
        self.assertEqual(framework._holm_adjusted_pvalues([]).size, 0)

    def test_insufficient_nonfinite_and_nonsignificant_do_not_run_posthoc(self):
        matrices = [pd.DataFrame([[1, 2], [2, 3]]),
                    pd.DataFrame([[1, 2, 3], [np.nan, 1, 2], [1, np.inf, 2]]),
                    pd.DataFrame(columns=["DE", "PSO", "GWO"]),
                    pd.DataFrame([[1, 2, 3], [2, 3, 1], [3, 1, 2]])]
        with patch.object(framework, "wilcoxon", side_effect=AssertionError("No post-hoc expected")):
            for matrix in matrices:
                with self.subTest(matrix=matrix.shape):
                    result = framework.calculate_friedman_analysis(matrix, "FULL")
                    self.assertEqual(result["summary"].iloc[0]["Post-hoc method"], "Not performed")
                    self.assertTrue(result["posthoc"]["Raw p-value"].isna().all())
                    if matrix.shape[0] == 0 or matrix.shape[1] < 3 or not np.isfinite(matrix.to_numpy()).all():
                        self.assertTrue(np.isnan(result["summary"].iloc[0]["p-value"]))

    def test_tied_ranks_and_degenerate_friedman(self):
        matrix = pd.DataFrame({"DE": [1] * 8, "PSO": [1] * 8, "GWO": [2] * 8})
        result = framework.calculate_friedman_analysis(matrix, "FULL")
        np.testing.assert_allclose(result["block_ranks"].iloc[0], [1.5, 1.5, 3])
        identical = result["posthoc"].iloc[0]
        self.assertEqual((identical["Wilcoxon statistic"], identical["Raw p-value"]), (0, 1))
        result = framework.calculate_friedman_analysis(pd.DataFrame(np.ones((4, 3))), "FULL")
        self.assertIn("all optimizers tied", result["summary"].iloc[0]["Conclusion"])
        self.assertTrue(np.isnan(result["summary"].iloc[0]["p-value"]))

    def test_analysis_workbook_keeps_classifier_and_transfer_separate(self):
        args = framework.parse_args(["--optimizers", "DE", "PSO", "GWO", "--estimators", "knn", "svm",
                                     "--transfer-functions", "vstf_01", "sstf_02"])
        datasets = [f"D{i}" for i in range(8)]
        results = {dataset: {} for dataset in datasets}
        for ci, classifier in enumerate(args.estimators):
            for ti, tf in enumerate(args.transfer_functions):
                for di, dataset in enumerate(datasets):
                    for mi, method in enumerate(args.optimizers):
                        value = di + (mi + ci + ti) % 3
                        label = framework.build_alg_label(method, tf, classifier, True, True)
                        results[dataset][label] = {"Estimator": classifier, "FitMean": -999,
                                                  "FitRuns": [value - 0.25, value + 0.25, np.nan, np.inf]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.xlsx"
            framework.export_friedman_analysis(results, datasets, args.optimizers, args, str(path))
            tables = pd.read_excel(path, sheet_name=None)
        self.assertEqual(list(tables), ["FULL_Friedman", "FULL_Average_Ranks", "FULL_PostHoc_Holm",
                                        "FULL_Block_Fitness", "FULL_Block_Ranks"])
        summaries = tables["FULL_Friedman"]
        self.assertEqual(len(summaries), 4)
        for row in summaries.itertuples(index=False, name=None):
            classifier, tf = row[:2]
            ci, ti = args.estimators.index(classifier), args.transfer_functions.index(tf)
            expected = args.optimizers[(-(ci + ti)) % 3]
            matching = summaries[(summaries.Classifier == classifier) & (summaries.TransferFunction == tf)].iloc[0]
            self.assertEqual(matching["Best average rank optimizer(s)"], expected)
            self.assertEqual(matching["Complete blocks used"], 8)
        # A configured missing method must not silently be dropped to manufacture a test.
        matrix = framework.build_friedman_fitness_matrix(results, datasets, ["DE", "PSO", "WOA"], args,
                                                         classifier="knn", transfer_function="vstf_01")
        self.assertTrue(matrix["WOA"].isna().all())
        self.assertEqual(framework.calculate_friedman_analysis(matrix, "FULL")["summary"].iloc[0]["Complete blocks used"], 0)


class CompleteCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="complete-cache-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = framework.parse_args([
            "--miafex-datasets", "One", "Two", "--pipeline-mode", "full",
            "--miafex-dataset-root", str(self.root / "offline-images"),
            "--feature-dataset-root", str(self.root / "offline-features"),
            "--output-root", str(self.root), "--exp-id", "604", "--reuse-cache-from-exp-id", "none",
            "--optimizers", "DE", "PSO", "--estimators", "knn", "svm",
            "--transfer-functions", "vstf_01", "sstf_02", "--runs", "3",
        ])
        self.args.optimizers = framework.resolve_optimizers(self.args)
        metadata_args = copy.deepcopy(self.args)
        metadata_args.figures_only = True
        self.scoped = framework.resolve_miafex_dataset_args(metadata_args)
        self.names = list(self.scoped)
        self.signatures = {name: framework.build_cache_signature(scoped) for name, scoped in self.scoped.items()}
        self.paths = framework.make_paths(self.args)

    def write_caches(self, ids=(0, 1, 2), paths=None, legacy=False):
        paths = paths or self.paths
        for dataset in self.names:
            for classifier in self.args.estimators:
                payload = {}
                for method in self.args.optimizers:
                    for tf in self.args.transfer_functions:
                        label = framework.build_alg_label(method, tf, classifier, True, True)
                        payload[label] = framework.build_label_payload(
                            classifier, *[[run + 1.0 for run in ids] for _ in range(7)],
                            [np.full(self.args.epochs, run + 1.0) for run in ids], self.args.epochs,
                            completed_run_ids=None if legacy else list(ids),
                        )
                filename = framework.cache_files(paths, dataset, classifier, self.signatures[dataset])[0]
                framework.save_cache(filename, payload)

    def is_complete(self):
        with redirect_stdout(io.StringIO()):
            return framework.experiment_cache_is_complete(self.paths, self.args, self.names, self.signatures, self.scoped)

    def test_checks_ids_not_counts_and_all_combinations(self):
        self.write_caches(ids=(0, 1, 3))
        self.assertFalse(self.is_complete())
        self.write_caches()
        self.assertTrue(self.is_complete())
        path = framework.cache_files(self.paths, "Two", "svm", self.signatures["Two"])[0]
        payload = framework.load_cache(path)
        missing = payload.pop(next(iter(payload)))
        framework.save_cache(path, payload)
        self.assertFalse(self.is_complete())
        self.write_caches()
        payload = framework.load_cache(path)
        next(iter(payload.values()))["TimeRuns"] = []
        framework.save_cache(path, payload)
        self.assertFalse(self.is_complete())
        self.write_caches(legacy=True)
        self.assertTrue(self.is_complete())

    def test_only_current_exp_and_reuse_policy_are_consulted(self):
        source = framework.make_paths(self.args, exp_id=603)
        self.write_caches(paths=source)
        self.args.reuse_cache_from_exp_id = 603
        for scoped in self.scoped.values():
            scoped.reuse_cache_from_exp_id = 603
        self.assertFalse(self.is_complete())
        self.assertEqual(list(Path(self.paths.cache_dir).iterdir()), [])
        self.write_caches()
        for scoped in self.scoped.values():
            scoped.reuse_cache = False
        self.assertFalse(self.is_complete())
        for path in Path(self.paths.cache_dir).glob('*_results.pkl'):
            path.rename(str(path).replace('_results.pkl', '_progress.pkl'))
        self.assertTrue(self.is_complete())

    def test_revision_guards_reject_unversioned_cache(self):
        self.args.optimizers = ["MaCRO-DE-t"]
        for dataset, scoped in self.scoped.items():
            scoped.optimizers = ["MaCRO-DE-t"]
            self.signatures[dataset] = framework.build_cache_signature(scoped)
        self.write_caches()
        for dataset in self.names:
            legacy_sig = framework._hash_cache_settings(framework._legacy_cache_settings(self.scoped[dataset]))
            for classifier in self.args.estimators:
                path = Path(framework.cache_files(self.paths, dataset, classifier, self.signatures[dataset])[0])
                path.rename(framework.cache_files(self.paths, dataset, classifier, legacy_sig)[0])
        self.assertFalse(self.is_complete())

    def test_complete_and_figures_only_regenerate_all_outputs_offline(self):
        self.write_caches()
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in Path(self.paths.cache_dir).glob('*.pkl')}
        for figures_only in (False, True):
            with self.subTest(figures_only=figures_only), ExitStack() as stack:
                args = copy.deepcopy(self.args)
                args.figures_only = figures_only
                stack.enter_context(patch.object(framework, "parse_args", return_value=args))
                stack.enter_context(patch.object(framework, "resolve_execution_config"))
                stack.enter_context(patch.object(framework, "print_backend_report"))
                stack.enter_context(patch.object(framework, "validate_execution_config"))
                charts = stack.enter_context(patch.object(framework, "generate_seven_global_charts", return_value=[]))
                for name in ("resolve_miafex_csv", "load_miafex_feature_data", "get_dataset", "train_miafex",
                             "extract_miafex_features", "run_single", "execute_pending_runs", "save_cache"):
                    stack.enter_context(patch.object(framework, name, side_effect=AssertionError(name)))
                output = io.StringIO()
                with redirect_stdout(output):
                    framework.main()
                charts.assert_called_once()
                self.assertEqual('[experiment-complete]' in output.getvalue(), not figures_only)
                for filename in ("Global_Results_EXP604.xlsx", "Statistical_Results_EXP604.xlsx",
                                 "Full_Friedman_Analysis_EXP604.xlsx", "RESUMEN_GRAFICAS_EXP604.csv"):
                    self.assertTrue((Path(self.paths.res_dir) / filename).is_file(), filename)
                for path, snapshot in before.items():
                    self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)


class WorkerInitializerTests(unittest.TestCase):
    def setUp(self):
        self.split = {"X_train": np.array([[1., 2.], [3., 4.]]), "y_train": np.array([0, 1]),
                      "X_test": np.array([[5., 6.]]), "y_test": np.array([1])}

    def test_spawn_worker_reuses_immutable_data_across_tasks(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=framework.get_context("spawn"),
                                 initializer=framework.initialize_run_worker, initargs=(self.split,)) as pool:
            first = pool.submit(worker_probe, 1).result(timeout=45)
            second = pool.submit(worker_probe, 2).result(timeout=45)
        self.assertEqual(first[:4], second[:4])
        self.assertNotEqual(first[0], os.getpid())
        self.assertEqual(first[2:4], (False, 10))
        self.assertEqual((first[-1], second[-1]), (1, 2))
        self.assertTrue(self.split['X_train'].flags.writeable)

    def test_submission_data_only_in_initializer_and_seeds_unchanged(self):
        args = framework.parse_args(["--parallel", "yes", "--n-workers", "2"])
        data = framework.Data().set_train_test(**self.split)
        executor = Mock()
        def submit(fn, task):
            self.assertIs(fn, framework.run_single_parallel_task)
            self.assertNotIn('data_split', task)
            future = Future()
            future.set_result((task['run'], task['seed']))
            return future
        executor.submit.side_effect = submit
        context = Mock()
        context.__enter__ = Mock(return_value=executor)
        context.__exit__ = Mock(return_value=False)
        with patch.object(framework, 'ProcessPoolExecutor', return_value=context) as pool:
            results = framework.execute_pending_runs(data, 'knn', 'DE', 'vstf_01', args, [4, 1, 3])
        self.assertEqual(results, [(run, args.seed_base + run) for run in [1, 3, 4]])
        self.assertIs(pool.call_args.kwargs['initializer'], framework.initialize_run_worker)
        self.assertEqual(pool.call_args.kwargs['mp_context'].get_start_method(), 'spawn')
        self.assertEqual(pool.call_args.kwargs['max_workers'], 2)
        self.assertEqual(len(pool.call_args.kwargs['initargs']), 1)
        self.assertIs(pool.call_args.kwargs['initargs'][0]['X_train'], data.X_train)

    def test_worker_calls_existing_run_single_with_unchanged_partitions(self):
        args = framework.parse_args([])
        with patch.object(framework, '_RUN_WORKER_DATA_SPLIT', None), \
                patch.object(framework, '_RUN_WORKER_THREAD_LIMIT', None), \
                patch.object(framework, 'threadpool_limits') as limits, \
                patch.object(framework, 'run_single', return_value={'synthetic': True}) as run:
            framework.initialize_run_worker(self.split)
            for index in (2, 4):
                task = dict(run=index, estimator='knn', method='DE', tf='vstf_01', args=args, seed=args.seed_base + index)
                self.assertEqual(framework.run_single_parallel_task(task), (index, {'synthetic': True}))
                data = run.call_args.args[0]
                for name, expected in self.split.items():
                    np.testing.assert_array_equal(getattr(data, name), expected)
                self.assertEqual(run.call_args.args[-1], args.seed_base + index)
            limits.assert_called_once_with(limits=1)


if __name__ == '__main__':
    unittest.main()
