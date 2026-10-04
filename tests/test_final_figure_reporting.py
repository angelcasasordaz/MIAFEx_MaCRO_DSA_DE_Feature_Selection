"""Absolute MIAFEx radar ratios, explicit classifier selection and figure-only moves."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main_best as m
from reporting import core, figures, statistics
from tests import test_full_plotting


class SelectedFeatureRatioTests(unittest.TestCase):
    def test_requested_coordinates_and_bounds(self):
        for selected, expected in ((768, 1.), (384, .5), (5, 5 / 768), (0, 0.)):
            with self.subTest(selected=selected):
                self.assertEqual(float(m.selected_feature_ratio(selected, 768)), expected)
        self.assertAlmostEqual(float(m.selected_feature_ratio(5, 768)), .0065104167, places=10)
        values = m.selected_feature_ratio(np.linspace(0, 768, 1001), 768)
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_invalid_counts_are_rejected_without_clipping(self):
        for selected, original in ((-1, 768), (769, 768), (np.nan, 768),
                                   (5, 0), (5, np.inf), (1, 3.5)):
            with self.subTest(selected=selected, original=original), self.assertRaises(ValueError):
                m.selected_feature_ratio(selected, original)

    def test_true_dataset_widths_label_handling_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            args = m.parse_args(['--feature-dataset-root', directory])
            for dataset, columns in [('A', 'label,f0,f1,f2'), ('B', 'f0,f1,f2,f3,f4,target')]:
                root = Path(directory, dataset)
                root.mkdir()
                for split in ('train', 'test'):
                    (root / f'{split}_features.csv').write_text(columns + '\n')
            before = copy.deepcopy(vars(args))
            with core.report_guard(()):
                self.assertEqual(m.plot_original_feature_counts(args, ['A', 'B']), {'A': 3, 'B': 5})
            self.assertEqual(vars(args), before)
            args.dataset_name = 'Custom'
            args.train_features_csv = str(Path(directory, 'B/train_features.csv'))
            args.test_features_csv = str(Path(directory, 'B/test_features.csv'))
            self.assertEqual(m.plot_original_feature_counts(args, ['Custom']), {'Custom': 5})
            Path(args.test_features_csv).write_text('f0,f1,target\n')
            with self.assertRaisesRegex(ValueError, 'train/test feature columns'):
                m.plot_original_feature_counts(args, ['Custom'])


class FinalFigureReportingTests(unittest.TestCase):
    setUp = test_full_plotting.FullPlottingTests.setUp

    def test_radar_artists_use_same_representative_run_and_absolute_width(self):
        before = pickle.dumps(self.results)
        for mode, index in [('best', 1), ('worst', 0), ('mean', None)]:
            with self.subTest(mode=mode), patch.object(m, 'PLOT_RUN_AGGREGATION', mode):
                table = m.generate_plot_dataframe(self.results, self.args)
                plot, opts, colors, _ = m.prepare_plot_groups(table[table.Estimador == 'knn'], self.args.optimizers)
                methods = plot.drop_duplicates('GrupoGrafica').set_index('GrupoGrafica').Optimizador.to_dict()
                selected = np.mean(self.row['FeatRuns']) if index is None else self.row['FeatRuns'][index]
                for original in (20, 768):
                    fig, ax = m.plt.subplots(subplot_kw={'polar': True})
                    m._draw_dataset_radar(ax, 'First', plot, opts, colors, methods, original_features=original)
                    for line in ax.lines:
                        self.assertEqual(line.get_ydata()[4], selected / original)
                    self.assertEqual(ax.get_xticklabels()[4].get_text(), 'Selected Feature Ratio')
                    m.plt.close(fig)
                labels, values = figures.radar_values(self.report, 'knn')
                self.assertEqual(labels[-1], 'Selected Feature Ratio')
                np.testing.assert_array_equal(values[:, :, -1], np.full((3, 2), selected / 20))
                metric = next(item for item in self.report.metrics if item.run_key == 'AccRuns')
                self.assertEqual(figures.run_observations(self.report, 'knn', metric).shape, (3, 6))
        self.assertEqual(pickle.dumps(self.results), before)

    def test_global_setting_selects_classifier_for_all_single_classifier_figures(self):
        for (dataset, classifier, algorithm), row in self.report.indexed.items():
            if classifier == 'svm':
                row['AccRuns'] = np.asarray([65., 60., 70.])
                row['AccMean'] = 65.
                row['FeatRuns'] = np.asarray([10., 12., 14.])
                row['FeatMean'] = 12.
        before = pickle.dumps(self.results)
        for classifier, expected_accuracy, expected_features in [('knn', .8, 8.), ('svm', .6, 12.)]:
            self.args.plot_global_estimator = classifier
            captured = {}
            def sink(fig, path):
                captured[Path(path).name] = [line.get_ydata().copy() for line in fig.axes[0].lines]
                Path(path).touch()
                m.plt.close(fig)
            with self.subTest(classifier=classifier), \
                    patch.object(m, 'PLOT_GLOBAL_ESTIMATOR', classifier), \
                    patch.object(m, 'PLOT_RUN_AGGREGATION', 'best'), \
                    patch.object(core, 'load_completed_cache', return_value=self.report), \
                    patch.object(figures, 'statistical_figures', return_value=[]), \
                    patch.object(figures, 'save_png', side_effect=sink), \
                    redirect_stdout(io.StringIO()):
                result = core.run_full_figures(self.args)
                self.assertEqual(result['optimization_calls'], 0)
                radar_path = f'02_radar_por_dataset_{classifier}.png'
                self.assertIn(radar_path, result['generated'])
                for line in captured[radar_path]:
                    self.assertEqual(line[0], expected_accuracy)
                    self.assertEqual(line[4], expected_features / 20)
                selected = next(metric for metric in self.report.metrics if metric.run_key == 'FeatRuns')
                np.testing.assert_array_equal(figures.metric_matrix(self.report, classifier, selected), expected_features)
                self.assertEqual([name for name in result['generated'] if 'heatmap' in name],
                                 [f'06_heatmap_accuracy_{classifier}.png'])
                self.assertIn(f'07_violin_accuracy_{classifier}.png', result['generated'])
                self.assertIn(f'05_convergence_por_dataset_{classifier}.png', result['generated'])
                self.assertEqual(set(self.report.classifiers), {'knn', 'svm'})
        self.assertEqual(pickle.dumps(self.results), before)

    def test_global_setting_does_not_change_identities_tables_or_friedman_inputs(self):
        scoped = m.resolve_miafex_dataset_args(self.args)['First']
        identity = m.build_combination_identity(scoped, 'First', 'knn', 'DE', 'vstf_01')
        signature = m.build_cache_signature(scoped)
        legacy = m.build_legacy_group_signature(scoped)
        summary = m.generate_summary_dataframe(self.results, self.args)
        metric = next(item for item in self.report.metrics if item.run_key == 'F1Runs')
        blocks, matrix = statistics.matched_block_matrix(
            self.report.indexed, self.report.datasets, self.report.classifiers, self.report.algorithms, metric)
        for classifier in ('knn', 'svm'):
            with patch.object(m, 'PLOT_GLOBAL_ESTIMATOR', classifier):
                self.assertEqual(m.build_combination_identity(scoped, 'First', 'knn', 'DE', 'vstf_01'), identity)
                self.assertEqual(m.build_cache_signature(scoped), signature)
                self.assertEqual(m.build_legacy_group_signature(scoped), legacy)
                m.pd.testing.assert_frame_equal(m.generate_summary_dataframe(self.results, self.args), summary)
                actual_blocks, actual = statistics.matched_block_matrix(
                    self.report.indexed, self.report.datasets, self.report.classifiers, self.report.algorithms, metric)
                self.assertEqual(actual_blocks, blocks)
                np.testing.assert_array_equal(actual, matrix)

    def test_existing_full_tree_moves_secondary_figures_without_data_changes_or_duplicates(self):
        root = Path(self.temp.name, 'Figures/EXP913')
        full = root / 'full'
        full.mkdir(parents=True)
        keep = ['06_heatmap_accuracy_knn.png', '07_violin_accuracy_knn.png']
        move = ['06_heatmap_accuracy_svm.png', '06_heatmap_f1_knn.png',
                '07_violin_accuracy_svm.png', '07_violin_recall_knn.png']
        for name in keep + move:
            (full / name).write_bytes(name.encode())
        (full / 'particular').mkdir()
        (full / 'particular/02_radar_First_knn.png').write_bytes(b'radar')
        (full / 'generic_holm_heatmap.png').write_bytes(b'statistic')
        # Identical pre-existing duplicate is removed; no scientific file is touched.
        (full / 'individual').mkdir()
        (full / 'individual' / move[0]).write_bytes(move[0].encode())
        protected = Path(self.temp.name, 'Results/EXP913/cache/protected.pkl')
        protected.parent.mkdir(parents=True)
        protected.write_bytes(b'scientific results')
        before = pickle.dumps(self.results)
        core.organize_full_figure_tree(root, full, self.report.datasets, self.report.classifiers)
        for name in keep:
            self.assertEqual((full / name).read_bytes(), name.encode())
        for name in move:
            self.assertFalse((full / name).exists())
            self.assertEqual((full / 'individual' / name).read_bytes(), name.encode())
        self.assertFalse((full / 'particular').exists())
        self.assertEqual((full / 'individual/02_radar_First_knn.png').read_bytes(), b'radar')
        self.assertEqual((full / 'statistics/generic_holm_heatmap.png').read_bytes(), b'statistic')
        self.assertEqual(protected.read_bytes(), b'scientific results')
        self.assertEqual(pickle.dumps(self.results), before)
        self.assertEqual(core.organize_full_figure_tree(root, full, self.report.datasets, self.report.classifiers), [])


if __name__ == '__main__':
    unittest.main()
