"""All publication types on small variable-sized data, without large PNG renders."""
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from reporting import core, figures, statistics, paper_tables
from tests.test_paper_tables import fixture


def report_fixture(datasets, algorithms, classifiers):
    args, results = fixture(datasets, algorithms, classifiers)
    indexed = {}
    for ds, rows in results.items():
        for label, row in rows.items():
            row['Curve'] = [.9, .7, .5]
            indexed[ds, row['Estimator'], label.rsplit('_', 1)[0]] = row
    return core.CompletedReport(args, results, indexed, list(datasets), list(classifiers),
                                list(algorithms), list(paper_tables.METRICS), 'fixture', {})


class GenericPublicationTypesTests(unittest.TestCase):
    def test_eight_types_adapt_to_actual_dimensions_and_classifiers(self):
        types = {'summary', 'radar', 'heatmap', 'precision', 'accuracy_boxplot',
                 'recall_violin', 'convergence', 'features_runtime'}
        for datasets, algorithms, classifiers in [(('Only',), ('DE',), ('tree',)),
                                                   (('A', 'B', 'C', 'D'), ('DE', 'PSO', 'JADE'), ('knn', 'rf'))]:
            report = report_fixture(datasets, algorithms, classifiers)
            skipped, names = [], []
            for stem, fig in figures.publication_figures(report, skipped):
                try:
                    names.append(stem)
                    if 'convergence' in stem:
                        self.assertEqual(sum(ax.get_visible() for ax in fig.axes), len(datasets))
                        self.assertTrue(all(len(ax.lines) == len(algorithms) for ax in fig.axes if ax.get_visible()))
                    if 'radar' in stem:
                        self.assertEqual(sum(ax.get_visible() for ax in fig.axes), len(datasets))
                    fig.canvas.draw()  # Small in-memory default DPI only.
                finally:
                    plt.close(fig)
            for ci in range(1, len(classifiers) + 1):
                for kind in types:
                    self.assertTrue(any(name.startswith(f'generic_{kind}_c{ci}') for name in names), kind)
            self.assertEqual(len(names), (7 + len(report.metrics)) * len(classifiers))
            if len(datasets) == 1:
                self.assertTrue(any(item['output'].startswith('Precision CI') for item in skipped))
                self.assertTrue(any(item['output'].startswith('Violin density') for item in skipped))
            else:
                self.assertEqual(skipped, [])

    def test_dynamic_ci_and_constant_violin_observations(self):
        values = np.array([[.4, .5, .6, .7], [.2, .3, .4, .5]])
        means, ci = figures.dataset_mean_ci(values)
        np.testing.assert_allclose(ci, t.ppf(.975, 3) * values.std(axis=1, ddof=1) / 2)
        self.assertIsNone(figures.dataset_mean_ci(values[:, :1])[1])
        fig = figures.violin_figure(np.full((2, 4), .7), ['Alpha', 'Beta'], 'custom')
        try:
            # Constant samples retain four scatter observations per algorithm.
            clouds = [c for c in fig.axes[0].collections if hasattr(c, 'get_offsets') and len(c.get_offsets()) == 4]
            self.assertEqual(len(clouds), 2)
        finally: plt.close(fig)

    def test_four_statistical_types_use_actual_algorithm_count(self):
        for n, k in ((4, 2), (5, 5)):
            algorithms = [f'Algorithm {i}' for i in range(k)]
            x = np.arange(n*k, dtype=float).reshape(n, k)**1.3
            analysis = statistics.analyze(x, algorithms)
            names = []
            for stem, fig in figures.statistical_figures(analysis, algorithms, 'Metric'):
                names.append(stem)
                plt.close(fig)
            self.assertEqual(set(names), {'generic_average_rank', 'generic_reference_comparisons',
                                         'generic_holm_heatmap', 'generic_block_distribution'})

    def test_palette_extends_and_runtime_package_has_no_experiment_branches(self):
        self.assertEqual(len(figures.palette([f'A{i}' for i in range(19)])), 19)
        package = Path(core.__file__).parent
        self.assertFalse((package / 'presets').exists())
        self.assertEqual({p.name for p in package.glob('*.py')},
                         {'__init__.py', 'core.py', 'figures.py', 'statistics.py', 'paper_tables.py'})
        for path in package.glob('*.py'):
            self.assertNotIn('627', path.read_text())
