"""Focused FULL figure regressions; synthetic cached observations, no science."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import main_best as m
from reporting import core, figures, paper_tables, statistics


class FullPlottingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(m.plt.close, 'all')
        self.args = m.parse_args(['--output-root', self.temp.name, '--exp-id', '913',
                                 '--optimizers', 'DE', 'PSO', 'JADE', '--estimators', 'knn', 'svm',
                                 '--runs', '3', '--epochs', '3', '--pop-size', '5',
                                 '--pipeline-mode', 'feature_selection'])
        self.row = m.build_label_payload(
            'knn', [95, 80, 90], [.97, .71, .85], [.96, .72, .86], [.94, .70, .84],
            [.30, .10, .20], [2, 8, 5], [1, 9, 4],
            [[.8, .5, .3], [.9, .4, .1], [.7, .3, .2]], 3, completed_run_ids=[0, 1, 2],
        )
        self.results, indexed = {}, {}
        self.args.feature_dataset_root = str(Path(self.temp.name) / 'features')
        for dataset in ('First', 'Second'):
            directory = Path(self.args.feature_dataset_root) / dataset
            directory.mkdir(parents=True)
            for split in ('train', 'test'):
                (directory / f'{split}_features.csv').write_text(
                    ','.join([f'f{i}' for i in range(20)] + ['label']) + '\n')
            self.results[dataset] = {}
            for classifier in self.args.estimators:
                for algorithm in self.args.optimizers:
                    row = copy.deepcopy(self.row)
                    row['Estimator'] = classifier
                    self.results[dataset][f'{algorithm}_{classifier.upper()}'] = row
                    indexed[dataset, classifier, algorithm] = row
        self.report = core.CompletedReport(self.args, self.results, indexed, list(self.results),
                                           self.args.estimators, self.args.optimizers,
                                           list(paper_tables.METRICS), 'synthetic-identity', {}, {})

    def test_real_run_alignment_for_both_publication_and_legacy_figures(self):
        before = pickle.dumps(self.results)
        for mode, index in (('best', 1), ('worst', 0), ('mean', None)):
            with self.subTest(mode=mode), patch.object(m, 'PLOT_RUN_AGGREGATION', mode):
                self.assertEqual(m._plot_run_index(self.row), index)
                plot = m.generate_plot_dataframe(self.results, self.args)
                for metric in self.report.metrics:
                    expected = (np.mean(self.row[metric.run_key]) if index is None
                                else self.row[metric.run_key][index]) / metric.scale
                    np.testing.assert_allclose(figures.metric_matrix(self.report, 'knn', metric), expected)
                expected_accuracy = self.row['AccMean'] / 100 if index is None else self.row['AccRuns'][index] / 100
                np.testing.assert_allclose(plot.AS_test, expected_accuracy)
                if mode == 'best':
                    self.assertEqual(expected_accuracy, .80)
                    self.assertNotEqual(expected_accuracy, .95)
                for column, key in [('PS_test', 'PSRuns'), ('RS_test', 'RSRuns'), ('F1_test', 'F1Runs'),
                                    ('N_Features_Selected', 'FeatRuns'), ('Runtime', 'TimeRuns')]:
                    expected = np.mean(self.row[key]) if index is None else self.row[key][index]
                    np.testing.assert_allclose(plot[column], expected)
                curve = self.row['Curve'] if index is None else self.row['CurvesAll'][index]
                np.testing.assert_array_equal(m.build_curve_dataframe(self.results, self.args, 'knn').iloc[0].Curve, curve)
                fig = figures.convergence_figure(self.report, 'knn', ['First'])
                for line in fig.axes[0].lines:
                    np.testing.assert_array_equal(line.get_ydata(), curve)
                m.plt.close(fig)
        self.assertEqual(pickle.dumps(self.results), before)

    def test_publication_artists_use_selected_metrics_and_all_runs_for_distributions(self):
        with patch.object(m, 'PLOT_RUN_AGGREGATION', 'best'):
            fig = figures.summary_figure(self.report, 'svm')
            self.assertEqual(fig.axes[0].patches[0].get_height(), .80)
            m.plt.close(fig)
            accuracy = next(metric for metric in self.report.metrics if metric.run_key == 'AccRuns')
            fig = figures.heatmap_figure(self.report, 'knn', accuracy)
            np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), np.full((3, 2), .8))
            m.plt.close(fig)
            observations = figures.run_observations(self.report, 'knn', accuracy)
            np.testing.assert_array_equal(observations, np.tile([.95, .8, .9, .95, .8, .9], (3, 1)))
            fig = figures.violin_figure(observations, self.report.algorithms, 'knn', 'Accuracy')
            clouds = [collection for collection in fig.axes[0].collections
                      if len(collection.get_offsets()) == 6]
            self.assertEqual(len(clouds), 0)
            self.assertEqual(sum(text.get_text() == '0.883' for text in fig.axes[0].texts), 3)
            m.plt.close(fig)

    def test_accuracy_files_style_order_and_individual_paths(self):
        captured = {}
        def sink(fig, path):
            captured[Path(path).name] = (Path(path).parent, fig)
            Path(path).touch()
        with patch.object(m, 'PLOT_RUN_AGGREGATION', 'best'), patch.object(figures, 'save_png', side_effect=sink):
            files = m.generate_seven_global_charts(m.generate_summary_dataframe(self.results, self.args),
                                                   self.results, self.temp.name, self.args.optimizers, self.args,
                                                   estimator_filter=m.PLOT_GLOBAL_ESTIMATOR)
        self.assertEqual([path for path in files if 'heatmap' in path], ['06_heatmap_accuracy_knn.png'])
        heat = captured['06_heatmap_accuracy_knn.png'][1].axes[0]
        np.testing.assert_array_equal(heat.images[0].get_array(), np.full((3, 2), .8))
        self.assertEqual([tick.get_text() for tick in heat.get_xticklabels()], ['First', 'Second'])
        self.assertEqual([tick.get_text() for tick in heat.get_yticklabels()], self.args.optimizers)
        for classifier in ('knn', 'svm'):
            for dataset in self.results:
                for family in ('radar', 'features_runtime', 'convergence'):
                    filename = f'{family}_{dataset}_{classifier}.png'
                    self.assertIn('individual/' + filename, files)
                    self.assertEqual(captured[filename][0], Path(self.temp.name) / 'individual')
        self.assertIn('07_violin_accuracy_knn.png', files)
        self.assertIn('individual/generic_radar_svm.png', files)
        self.assertEqual(len(files), len(set(files)))

    def test_export_boundary_passes_selected_data_and_preserves_summary_tables(self):
        paths = m.make_paths(self.args)
        before = pickle.dumps(self.results)
        for mode, accuracy in (('best', .8), ('worst', .95), ('mean', .8833333333333333)):
            with patch.object(m, 'PLOT_RUN_AGGREGATION', mode), \
                    patch.object(m, 'export_global_excel', return_value=[]) as global_table, \
                    patch.object(m, 'export_statistical_excel') as stats_table, \
                    patch.object(m, 'export_friedman_analysis') as friedman, \
                    patch.object(m, 'generate_seven_global_charts', return_value=[]) as chart, \
                    patch.object(statistics, 'export', return_value=[]), \
                    redirect_stdout(io.StringIO()):
                m.export_reporting_outputs(paths, self.args, list(self.results), self.results, self.results)
                np.testing.assert_allclose(chart.call_args.args[0].AS_test, accuracy)
                self.assertEqual(Path(chart.call_args.args[2]), Path(paths.fig_dir) / 'full')
                for exporter in (global_table, stats_table, friedman):
                    self.assertEqual(pickle.dumps(exporter.call_args.args[0]), before)
                summary = pd.read_csv(Path(paths.res_dir) / 'RESUMEN_GRAFICAS_EXP913.csv')
                np.testing.assert_allclose(summary.AS_test, self.row['AccMean'] / 100)
        self.assertEqual(pickle.dumps(self.results), before)

    def test_scientific_workbooks_and_statistics_keep_their_mean_and_per_metric_reducers(self):
        baseline = None
        for mode in ('best', 'worst', 'mean'):
            folder = Path(self.temp.name) / mode
            folder.mkdir()
            with patch.object(m, 'PLOT_RUN_AGGREGATION', mode):
                m.export_global_excel(self.results, list(self.results), str(folder / 'global.xlsx'))
                m.export_statistical_excel(self.results, list(self.results), self.args.optimizers, self.args, str(folder / 'stats.xlsx'))
                m.export_friedman_analysis(self.results, list(self.results), self.args.optimizers, self.args, str(folder / 'friedman.xlsx'))
                current = {f'{file.name}/{sheet}': frame for file in folder.glob('*.xlsx')
                           for sheet, frame in pd.read_excel(file, sheet_name=None).items()}
                if baseline is None:
                    baseline = current
                else:
                    for sheet, frame in current.items():
                        pd.testing.assert_frame_equal(frame, baseline[sheet])
                _, matrix = statistics.matched_block_matrix(self.report.indexed, self.report.datasets,
                                                             self.report.classifiers, self.report.algorithms,
                                                             next(x for x in self.report.metrics if x.run_key == 'F1Runs'))
                np.testing.assert_allclose(matrix, np.mean(self.row['F1Runs']))
        np.testing.assert_allclose(baseline['global.xlsx/Accuracy'].iloc[:, 1:], 88.33333333333333)
        # Statistical Best remains max Accuracy (.95), independently of best fitness (.80).
        self.assertIn(95., baseline['stats.xlsx/Accuracy'].select_dtypes('number').to_numpy())

    def test_combined_figures_only_dispatch_skips_tables_and_scientific_execution(self):
        args = copy.deepcopy(self.args)
        args.report_only = args.figures_only = True
        with patch.object(m, 'parse_args', return_value=args), \
                patch('reporting.core.run_full_figures', return_value={'optimization_calls': 0}) as full, \
                patch('reporting.core.run_report', side_effect=AssertionError('No table export')), \
                patch.object(m, 'resolve_execution_config', side_effect=AssertionError('No science')):
            self.assertEqual(m.main(), {'optimization_calls': 0})
            full.assert_called_once_with(args)

    def test_full_cache_only_folder_migration_and_statistical_location(self):
        root = Path(self.temp.name)
        source = root / 'Figures/EXP913'
        source.mkdir(parents=True)
        (source / '02_radar_First_knn.png').write_bytes(b'old figure')
        (source / '06_heatmap_f1_knn.png').write_bytes(b'old heatmap')
        (source / 'generic_average_rank.png').write_bytes(b'old rank')
        scientific = root / 'Results/EXP913/cache'
        scientific.mkdir(parents=True)
        (scientific / 'protected.pkl').write_bytes(b'protected cache')
        with patch.object(core, 'load_completed_cache', return_value=self.report), \
                patch.object(figures, 'generate', return_value=[]) as generate, \
                patch.object(figures, 'save_png', side_effect=lambda fig, path: m.plt.close(fig)), \
                redirect_stdout(io.StringIO()):
            result = core.run_full_figures(self.args)
        destination = source / 'full'
        self.assertTrue((destination / 'individual/02_radar_First_knn.png').is_file())
        self.assertTrue((destination / 'individual/06_heatmap_f1_knn.png').is_file())
        self.assertTrue((destination / 'statistics/generic_average_rank.png').is_file())
        self.assertFalse(list(source.glob('*.png')))
        self.assertIs(generate.call_args.args[0], self.report)
        self.assertEqual(generate.call_args.args[1], destination)
        self.assertEqual(result['optimization_calls'], 0)
        self.assertTrue(all(name.startswith('statistics/') for name in result['generated']))
        self.assertEqual((scientific / 'protected.pkl').read_bytes(), b'protected cache')
        self.assertFalse(list((root / 'Results').rglob('*.xlsx')))


if __name__ == '__main__':
    unittest.main()
