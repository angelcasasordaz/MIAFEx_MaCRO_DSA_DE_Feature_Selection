"""Publication/statistical smoke fixtures; never train, fit, or run an optimizer."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import main_best as m
from reporting import core, figures, statistics, paper_tables
import full_plot_style as style
from figure_layout import main_figure_names, full_figure_category, ROOT, STATISTICS
from plot_labels import plot_display_label, primary_algorithm


ALGORITHMS = ['MaCRO-DE-t-v2', 'DE', 'JADE', 'SHADE', 'PSO', 'GWO', 'WOA', 'HHO',
              'DBO', 'RUN', 'FOX', 'BRO', 'MaCRO-DE', 'DSADE']


def fixture(count=5, datasets=('First', 'Second'), epochs=40):
    args = m.parse_args(['--report-only', '--exp-id', '913', '--optimizers', *ALGORITHMS[:count],
                         '--estimators', 'knn', 'svm', '--runs', '3', '--epochs', str(epochs)])
    indexed, results = {}, {ds: {} for ds in datasets}
    for di, ds in enumerate(datasets):
        for ci, classifier in enumerate(args.estimators):
            for ai, algorithm in enumerate(ALGORITHMS[:count]):
                fitness = np.array([.30, .10, .20]) + .002*ai
                row = {'Estimator': classifier, 'CompletedRuns': 3, 'CompletedRunIDs': [0, 1, 2],
                       'FitRuns': fitness, 'AccRuns': np.array([99., 75., 88.])-di-ci,
                       'PSRuns': np.array([.95, .61, .85])-.001*ai,
                       'RSRuns': np.array([.97, .62, .87])-.001*ai,
                       'F1Runs': np.array([.96, .60, .86])-.001*ai,
                       'FeatRuns': np.array([1., 7., 4.])+ai, 'TimeRuns': np.array([1., 9., 4.])+ci}
                for metric in paper_tables.METRICS:
                    row[metric.run_key.replace('Runs', 'Mean')] = float(np.mean(row[metric.run_key]))
                row['CurvesAll'] = [np.r_[np.linspace(.9, fit, epochs//2), np.full(epochs-epochs//2, fit)] for fit in fitness]
                row['Curve'] = np.mean(row['CurvesAll'], axis=0)
                indexed[ds, classifier, algorithm] = row
                results[ds][f'{algorithm}_{classifier.upper()}'] = row
    return core.CompletedReport(args, results, indexed, list(datasets), args.estimators, ALGORITHMS[:count],
                                list(paper_tables.METRICS), 'synthetic', {}, {})


class PublicationPresentationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')
        self.report = fixture()
        self.enterContext(patch.object(m, 'plot_original_feature_counts', return_value={'First': 768, 'Second': 768, 'Accuracy': 768}))

    def test_best_worst_mean_share_one_run_and_leave_tables_and_caches_unchanged(self):
        report = self.report
        before = pickle.dumps(report.results)
        summary = m.generate_summary_dataframe(report.results, report.args)
        baseline_blocks = statistics.matched_block_matrix(report.indexed, report.datasets, report.classifiers,
                                                          report.algorithms, statistics.selected_metric(report))[1]
        for mode, index in [('best', 1), ('worst', 0), ('mean', None)]:
            with self.subTest(mode=mode), patch.object(m, 'PLOT_RUN_AGGREGATION', mode):
                for metric in report.metrics:
                    expected = [[(np.mean(report.indexed[ds, 'knn', a][metric.run_key]) if index is None
                                  else report.indexed[ds, 'knn', a][metric.run_key][index])/metric.scale
                                 for ds in report.datasets] for a in report.algorithms]
                    np.testing.assert_allclose(figures.metric_matrix(report, 'knn', metric), expected)
                fig = figures.summary_figure(report)
                np.testing.assert_allclose([bar.get_height() for bar in fig.axes[0].patches],
                                          figures.metric_matrix(report, 'knn', report.metrics[0]).mean(axis=1))
                plt.close(fig)
                np.testing.assert_array_equal(figures.run_observations(report, 'knn', report.metrics[0])[0],
                    np.concatenate([report.indexed[ds, 'knn', report.algorithms[0]]['AccRuns']/100 for ds in report.datasets]))
                matrix = statistics.matched_block_matrix(report.indexed, report.datasets, report.classifiers,
                                                         report.algorithms, statistics.selected_metric(report))[1]
                np.testing.assert_array_equal(matrix, baseline_blocks)
                pd.testing.assert_frame_equal(summary, m.generate_summary_dataframe(report.results, report.args))
        self.assertEqual(pickle.dumps(report.results), before)

    def test_nine_figures_adapt_to_5_11_14_algorithms_and_variable_datasets(self):
        for count, datasets in [(5, ('First',)), (11, ('First', 'Second')), (14, ('First', 'Second', 'Third', 'Fourth'))]:
            report = fixture(count, datasets)
            with patch.object(m, 'plot_original_feature_counts', return_value={ds: 768 for ds in datasets}):
                names, skipped = [], []
                for stem, fig in figures.base_publication_figures(report, skipped):
                    names.append(stem+'.png')
                    if stem.startswith('02_'):
                        self.assertEqual(sum(ax.get_visible() for ax in fig.axes), len(datasets))
                        self.assertEqual(len(fig.axes[0].lines), count)
                    if stem.startswith('05_'):
                        self.assertEqual(len(fig.axes[0].lines), count)
                        self.assertEqual(len(fig.axes[0].child_axes), 1)
                    fig.set_dpi(45); fig.canvas.draw(); plt.close(fig)
                self.assertEqual(tuple(names), main_figure_names('knn', 'accuracy'))
                self.assertFalse(skipped)

    def test_classifier_metric_and_language_switch_without_changing_identities(self):
        report = self.report
        before = pickle.dumps(report.results)
        for classifier, metric, language in [('knn', 'accuracy', 'en'), ('svm', 'f1', 'es'),
                                             ('knn', 'precision', 'es'), ('svm', 'recall', 'en')]:
            report.args.plot_global_estimator, report.args.plot_global_metric, report.args.figure_language = classifier, metric, language
            names = []
            for stem, fig in figures.base_publication_figures(report, []):
                names.append(stem+'.png')
                if stem.startswith('06_'):
                    key = {'accuracy': 'AccRuns', 'precision': 'PSRuns', 'recall': 'RSRuns', 'f1': 'F1Runs'}[metric]
                    selected = next(v for v in report.metrics if v.run_key == key)
                    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), figures.metric_matrix(report, classifier, selected))
                    self.assertEqual(fig.axes[0].get_xlabel(), 'Dataset' if language == 'en' else 'Conjunto de datos')
                plt.close(fig)
            self.assertEqual(tuple(names), main_figure_names(classifier, metric))
        self.assertEqual(pickle.dumps(report.results), before)
        report.args.plot_global_estimator = 'absent'
        with self.assertRaises(ValueError): figures.base_classifier(report)

    def test_spanish_does_not_translate_dataset_or_algorithm_identities(self):
        report = fixture(datasets=('Accuracy',))
        report.args.figure_language = 'es'
        figures_list = figures.base_publication_figures(report, [])
        texts = []
        for stem, fig in figures_list:
            texts.extend(t.get_text() for t in fig.findobj(match=Text))
            if stem.startswith('06_'):
                self.assertEqual(fig.axes[0].get_xticklabels()[0].get_text(), 'Accuracy')
            plt.close(fig)
        self.assertIn('Exactitud', texts)
        self.assertIn('Precisión', texts)
        self.assertIn('Media', texts)
        self.assertIn('Mediana', texts)
        self.assertIn('Proporción de características seleccionadas', texts)

    def test_every_palette_is_deterministic_centralized_and_safely_extended(self):
        for name in style.PALETTES:
            colors = style.palette(ALGORITHMS, name)
            self.assertEqual(len(set(colors.values())), 14)
            reordered = style.palette(list(reversed(ALGORITHMS)), name)
            subset = style.palette(ALGORITHMS[::3], name)
            self.assertEqual(colors, reordered)
            self.assertEqual(subset, {a: colors[a] for a in ALGORITHMS[::3]})
            self.report.args.plot_color_palette = name
            labels, values = figures.radar_values(self.report, 'knn')
            fig = figures.radar_figure(self.report, 'knn', labels, values)
            for line in fig.axes[0].lines:
                algorithm = next(a for a in self.report.algorithms if plot_display_label(a) == line.get_label())
                self.assertEqual(to_hex(line.get_color()), to_hex(colors[algorithm]))
            plt.close(fig)
            with patch.object(m, 'PLOT_COLOR_PALETTE', name):
                for algorithm in ALGORITHMS:
                    self.assertEqual(m.optimizer_plot_color(algorithm), colors[algorithm])
        with self.assertRaises(ValueError): style.palette(ALGORITHMS, 'unknown')

    def test_primary_from_configuration_drawn_last_without_numerical_changes(self):
        report = self.report
        report.algorithms = report.algorithms[1:] + report.algorithms[:1]
        report.args.optimizers = ['DE', 'MaCRO-DE-t-v2', 'PSO']
        self.assertEqual(primary_algorithm(report.algorithms, report.args), 'MaCRO-DE-t-v2')
        labels, values = figures.radar_values(report, 'knn')
        fig = figures.radar_figure(report, 'knn', labels, values)
        lines = fig.axes[0].lines
        self.assertEqual(lines[-1].get_label(), 'MaCRO-DE-t-v2')
        self.assertGreater(lines[-1].get_linewidth(), lines[0].get_linewidth())
        np.testing.assert_array_equal(lines[-1].get_ydata(), np.r_[values[-1, 0], values[-1, 0, 0]])
        self.assertEqual(len(lines), len(report.algorithms))
        self.assertNotEqual(plot_display_label('MaCRO-DE-t'), 'DSA-DE')

    def test_convergence_zoom_low_band_last_quarter_and_never_favors_primary_values(self):
        report = fixture(count=5, epochs=200)
        for a, final in zip(report.algorithms, [.8, .10, .101, .6, .7]):
            for ds in report.datasets:
                row = report.indexed[ds, 'knn', a]
                row['FitRuns'] = np.array([final+.002, final, final+.001])
                row['CurvesAll'] = [np.r_[np.linspace(.95, fit, 100), np.full(100, fit)] for fit in row['FitRuns']]
        fig = figures.convergence_figure(report, 'knn', report.datasets)
        inset = fig.axes[0].child_axes[0]
        self.assertEqual(inset.get_xlim(), (151., 200.))
        self.assertLess(inset.get_ylim()[1], .11)
        self.assertEqual(len(inset.lines), 5)
        self.assertEqual(fig.axes[0].lines[-1].get_label(), 'MaCRO-DE-t-v2')
        self.assertEqual(fig.axes[0].lines[-1].get_ydata()[-1], .8)
        row = report.indexed['First', 'knn', report.algorithms[1]]
        row.pop('CurvesAll')
        row['Curve'] = [float('nan')]*200
        skipped = []
        fig = figures.convergence_figure(report, 'knn', ['First'], skipped=skipped)
        self.assertEqual(len(fig.axes[0].lines), 4)
        self.assertFalse(fig.axes[0].child_axes)
        self.assertTrue(any(item['output'].startswith('Convergence inset') for item in skipped))

    def test_violin_box_use_real_samples_no_internal_points_and_centered_mean_numbers(self):
        observed = np.array([[.1, .2, .9, .4], [.4, .4, .4, .4]])
        algorithms = ['MaCRO-DE-t-v2', 'DE']
        for creator in (figures.violin_figure, figures.boxplot_figure):
            fig = creator(observed, algorithms, 'knn', reference=algorithms[0])
            ax = fig.axes[0]
            self.assertFalse(any(isinstance(c, PathCollection) for c in ax.collections))
            labels = [text for text in ax.texts if text.get_text() in {f'{x:.3f}' for x in observed.mean(axis=1)}]
            self.assertEqual(len(labels), 2)
            self.assertTrue(all(text.get_ha() == 'center' for text in labels))
            np.testing.assert_allclose([text.xy[1] for text in labels], observed.mean(axis=1))
            plt.close(fig)
        fig, ax = plt.subplots()
        boxes = figures.draw_boxplot(ax, observed, algorithms, reference=algorithms[0])
        np.testing.assert_allclose([line.get_ydata()[0] for line in boxes['medians']], np.median(observed, axis=1))
        np.testing.assert_allclose([line.get_ydata()[0] for line in boxes['means']], np.mean(observed, axis=1))
        self.assertEqual(boxes['whiskers'][1].get_ydata()[-1], .9)
        self.assertEqual(len(boxes['fliers']), 0)

    def test_one_principal_heatmap_only_and_correct_output_categories(self):
        report, skipped = self.report, []
        names = []
        for generator in (figures.base_publication_figures(report, skipped), figures.publication_figures(report, skipped),
                          figures.per_dataset_figures(report, skipped)):
            for stem, fig in generator:
                names.append(stem+'.png'); plt.close(fig)
        heatmaps = [name for name in names if 'heatmap' in name]
        self.assertEqual(heatmaps, ['06_heatmap_accuracy_knn.png'])
        self.assertEqual(full_figure_category(heatmaps[0]), ROOT)
        self.assertEqual(full_figure_category('generic_holm_heatmap.png'), STATISTICS)


class MatchedBlockReportingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')

    def test_cells_are_run_means_and_blocks_are_identical_dataset_classifier_keys(self):
        report = fixture()
        metric = statistics.selected_metric(report)
        blocks, matrix = statistics.matched_block_matrix(report.indexed, report.datasets, report.classifiers,
                                                         report.algorithms, metric, expected_runs=3)
        self.assertEqual(blocks, [('First', 'knn'), ('First', 'svm'), ('Second', 'knn'), ('Second', 'svm')])
        self.assertEqual(matrix.shape, (4, 5))
        self.assertAlmostEqual(matrix[0, 0], np.mean([.96, .60, .86]))
        del report.indexed['Second', 'svm', report.algorithms[-1]]
        blocks, matrix = statistics.matched_block_matrix(report.indexed, report.datasets, report.classifiers,
                                                         report.algorithms, metric, expected_runs=3)
        analysis = statistics.analyze(matrix, report.algorithms)
        self.assertEqual(analysis['omitted_blocks'], [3])
        self.assertEqual(analysis['x'].shape, (3, 5))
        np.testing.assert_array_equal(analysis['x'], matrix[:3])
        report.indexed['First', 'knn', report.algorithms[0]]['CompletedRuns'] = 2
        self.assertTrue(np.isnan(statistics.matched_block_matrix(report.indexed, report.datasets, report.classifiers,
                                                                report.algorithms, metric, expected_runs=3)[1][0, 0]))

    def test_friedman_average_tied_ranks_and_complete_holm_pair_family(self):
        x = np.array([[.9, .8, .7], [.8, .7, .6], [.7, .6, .7], [.6, .7, .5], [.8, .5, .4]])
        result = statistics.analyze(x, ['A', 'B', 'C'])
        np.testing.assert_array_equal(result['ranks'], rankdata(-x, method='average', axis=1))
        omnibus = friedmanchisquare(*x.T)
        self.assertAlmostEqual(result['friedman']['statistic'], omnibus.statistic)
        self.assertAlmostEqual(result['friedman']['pvalue'], omnibus.pvalue)
        self.assertEqual(result['family_size'], 3)
        for row in result['pairs'].itertuples():
            i, j = ['A', 'B', 'C'].index(row.Algorithm_A), ['A', 'B', 'C'].index(row.Algorithm_B)
            test = wilcoxon(x[:, i]-x[:, j], alternative='two-sided', zero_method='wilcox', method=row.Method)
            self.assertAlmostEqual(row.Raw_p, test.pvalue)
        raw = result['pairs'].Raw_p.to_numpy()
        order = np.argsort(raw)
        expected = np.empty(3)
        for r, i in enumerate(order): expected[i] = min(1., max((3-j)*raw[order[j]] for j in range(r+1)))
        np.testing.assert_allclose(result['pairs'].Holm_adjusted_p, expected)
        self.assertEqual(statistics.analyze([[1, 1, 1], [2, 2, 2]], ['A', 'B', 'C'])['pairs'].Raw_p.tolist(), [1., 1., 1.])
        self.assertIsNone(statistics.analyze([[1, 1, 1], [2, 2, 2]], ['A', 'B', 'C'])['friedman']['pvalue'])
        self.assertTrue(statistics.analyze([[1, 2]], ['A', 'B'])['pairs'].Raw_p.isna().all())

    def test_reference_comparisons_include_pairs_where_proposed_is_algorithm_b(self):
        algorithms = ['DE', 'MaCRO-DE-t-v2', 'PSO']
        analysis = statistics.analyze(np.arange(18).reshape(6, 3), algorithms)
        names = []
        for stem, fig in figures.statistical_figures(analysis, algorithms, 'F1-Score', reference=algorithms[1]):
            names.append(stem)
            if stem == 'generic_reference_comparisons':
                self.assertEqual({t.get_text() for t in fig.axes[0].get_yticklabels()}, {'DE', 'PSO'})
            plt.close(fig)
        self.assertEqual(len(names), 4)

    def test_statistical_exports_report_omitted_keys_metric_and_no_false_equivalence(self):
        report = fixture()
        del report.indexed['Second', 'svm', report.algorithms[-1]]
        with tempfile.TemporaryDirectory() as root:
            def save(fig, path):
                plt.close(fig); Path(path).write_bytes(b'fixture-png')
            with patch.object(figures, 'save_png', side_effect=save):
                statistics.export(report, Path(root)/'figures', Path(root)/'results')
            results = Path(root)/'results'
            self.assertEqual({p.name for p in results.iterdir()}, {'statistical_summary.csv', 'pairwise_wilcoxon_holm.csv',
                              'matched_block_means.csv', 'omitted_blocks.csv', 'statistical_report.txt'})
            summary = pd.read_csv(results/'statistical_summary.csv')
            self.assertTrue((summary.Complete_Blocks == 3).all())
            self.assertEqual(pd.read_csv(results/'omitted_blocks.csv').Dataset.tolist(), ['Second'])
            text = (results/'statistical_report.txt').read_text()
            self.assertIn("('Second', 'svm')", text)
            self.assertIn('Non-significance does not establish equivalence', text)
            self.assertIn('ranks do not measure effect magnitude', text)
            self.assertIn('Optimization calls: 0', text)
            report.args.statistical_metric = 'precision'
            self.assertEqual(statistics.selected_metric(report).run_key, 'PSRuns')


class ReportingScienceIsolationTests(unittest.TestCase):
    def test_presentation_options_do_not_change_scientific_or_legacy_signatures(self):
        report = fixture()
        report.args.dataset_source = 'mafese'
        with patch.object(m, 'get_dataset', return_value=m.Data([[1], [2]], [0, 1])):
            baseline = m.build_combination_identity(report.args, 'Tiny', 'knn', 'DE', 'vstf_01')
            cache = m.build_cache_signature(report.args)
            legacy = m.build_legacy_group_signature(report.args)
            for palette_name in style.PALETTES:
                report.args.plot_color_palette = palette_name
                report.args.figure_language = 'es'
                report.args.plot_global_metric = 'recall'
                report.args.plot_global_estimator = 'svm'
                report.args.statistical_metric = 'precision'
                self.assertEqual(m.build_combination_identity(report.args, 'Tiny', 'knn', 'DE', 'vstf_01'), baseline)
                self.assertEqual(m.build_cache_signature(report.args), cache)
                self.assertEqual(m.build_legacy_group_signature(report.args), legacy)

    def test_guarded_report_only_dispatch_has_zero_optimizer_calls_and_append_only_outputs(self):
        from tests.test_generic_reporting import arguments, caches
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            args = arguments(Path(root))
            args.pipeline_mode = 'feature_selection'
            caches(args)
            before = {p: (core.sha256(p), p.stat().st_mtime_ns) for p in Path(root).rglob('*.pkl')}
            args.report_output_root = str(Path(root)/'review')
            for name in ('build_optimizer', 'run_single', 'execute_pending_runs', 'train_miafex',
                         'extract_miafex_features', 'resolve_miafex_csv', 'save_cache', 'save_combination'):
                mock = stack.enter_context(patch.object(m, name, side_effect=AssertionError('Scientific execution forbidden')))
                self.addCleanup(mock.assert_not_called)
            def tiny_figures(report, skipped):
                for name in figures.base_figure_names(report):
                    yield Path(name).stem, plt.figure(figsize=(.3, .3))
            stack.enter_context(patch.object(figures, 'base_publication_figures', side_effect=tiny_figures))
            stack.enter_context(patch.object(figures, 'publication_figures', return_value=iter(())))
            stack.enter_context(patch.object(figures, 'per_dataset_figures', return_value=iter(())))
            def tiny_stats(*args, **kwargs):
                for name in ('generic_average_rank', 'generic_reference_comparisons', 'generic_holm_heatmap', 'generic_block_distribution'):
                    yield name, plt.figure(figsize=(.3, .3))
            stack.enter_context(patch.object(figures, 'statistical_figures', side_effect=tiny_stats))
            stack.enter_context(patch.object(m, 'parse_args', return_value=args))
            stack.enter_context(redirect_stdout(io.StringIO()))
            manifest = m.main()
            self.assertEqual(manifest['optimization_calls'], 0)
            self.assertTrue(manifest['protected_files_unchanged'])
            self.assertEqual(before, {p: (core.sha256(p), p.stat().st_mtime_ns) for p in before})
            destination = Path(manifest['figures_destination'])
            self.assertEqual({p.name for p in destination.glob('*.png')}, set(main_figure_names()))
            self.assertEqual(len(list((destination/'statistics').glob('*.png'))), 4)
            self.assertEqual(manifest['reports'][0]['presentation']['statistical_metric'], 'F1Runs')


if __name__ == '__main__':
    unittest.main()
