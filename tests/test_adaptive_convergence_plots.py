"""Plotting-only regressions: no optimizers, training, or cache writes."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

import main_best as f


class AdaptiveConvergenceTests(unittest.TestCase):
    def tearDown(self):
        f.plt.close("all")

    @staticmethod
    def clustered_curves():
        # Separate until iteration 51, then persistently close together.
        return np.array([np.r_[np.full(50, value), np.full(50, .1 + i * .01)]
                         for i, value in enumerate((1., 2., 3., 4.))])

    def test_window_is_later_sustained_and_order_independent(self):
        curves = self.clustered_curves()
        original = curves.copy()
        window = f._convergence_zoom_window(curves)
        self.assertEqual(window[:2], (76, 100))
        self.assertEqual(window, f._convergence_zoom_window(curves[::-1]))
        self.assertEqual(window, f._convergence_zoom_window(curves))
        np.testing.assert_array_equal(curves, original)
        scaled = f._convergence_zoom_window(curves * 7 + 3)
        self.assertEqual(scaled[:2], window[:2])
        np.testing.assert_allclose(scaled[2:], np.array(window[2:]) * 7 + 3)

    def test_no_inset_for_short_identical_or_invalid_histories(self):
        for curves in ([], [np.ones(100)], [np.ones(100), np.ones(100)],
                       [np.ones(7), np.zeros(7)], [np.ones(20), np.zeros(21)],
                       [np.full(100, np.nan), np.zeros(100)]):
            with self.subTest(curves=curves):
                self.assertIsNone(f._convergence_zoom_window(curves))

    def test_unseparated_curves_keep_full_final_envelope(self):
        curves = [np.linspace(3, 2, 100), np.linspace(2, 1, 100)]
        start, end, low, high = f._convergence_zoom_window(curves)
        self.assertEqual((start, end), (76, 100))
        self.assertLess(low, 1.)
        self.assertGreater(high, curves[0][75])

    def test_persistent_outlier_does_not_hide_a_majority_cluster(self):
        curves = self.clustered_curves()
        curves[-1] = 10.
        window = f._convergence_zoom_window(curves)
        self.assertEqual(window[:2], (76, 100))
        self.assertLess(window[3], 1.)

    def test_two_high_outliers_do_not_expand_low_fitness_zoom(self):
        low_curves = np.array([np.linspace(1., end, 200) for end in (.001, .002, .003)])
        # The entire final stage is close, irrespective of early convergence.
        low_curves[:, 150:] = np.array([np.linspace(.004, end, 50) for end in (.001, .002, .003)])
        curves = np.vstack([low_curves, np.full(200, .04), np.full(200, .06)])
        start, end, low, high = f._convergence_zoom_window(curves)
        self.assertEqual((start, end), (151, 200))
        self.assertLess(low, .001)
        self.assertGreater(high, .004)
        self.assertLess(high, .04)
        self.assertEqual((start, end, low, high), f._convergence_zoom_window(curves[::-1]))

    def test_plot_uses_exact_samples_sparse_markers_styles_and_logs(self):
        curves = self.clustered_curves()
        methods = ["MaCRO-DE-t", "DE", "PSO", "GWO"]
        table = f.pd.DataFrame([
            {"Archivo": "Example", "Estimador": "knn", "Optimizador": method, "Curve": curve}
            for method, curve in zip(methods, curves)
        ])
        table, opts, colors, labels = f.prepare_plot_groups(table, methods)
        fig, ax = f.plt.subplots()
        output = io.StringIO()
        with redirect_stdout(output):
            f._draw_dataset_convergence(ax, "Example", table, opts, colors)
        self.assertIn(f"dataset=Example classifier=knn aggregation={f.PLOT_RUN_AGGREGATION} start_iteration=76 end_iteration=100",
                      output.getvalue())
        self.assertEqual(len(ax.child_axes), 1)
        inset = ax.child_axes[0]
        for line, zoom_line in zip(ax.lines, inset.lines):
            original = curves[methods.index(line.get_label())]
            np.testing.assert_array_equal(line.get_ydata(), original)
            np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 101))
            np.testing.assert_array_equal(zoom_line.get_ydata(), original[75:100])
            np.testing.assert_array_equal(zoom_line.get_xdata(), np.arange(76, 101))
            self.assertEqual(line.get_color(), f.optimizer_plot_color(line.get_label()))
            self.assertEqual(line.get_markevery()[0], 0)
            self.assertEqual(line.get_markevery()[-1], 99)
            self.assertLessEqual(len(line.get_markevery()), 9)
        macro = next(line for line in ax.lines if line.get_label() == "MaCRO-DE-t")
        self.assertEqual(macro.get_linestyle(), "-")
        self.assertTrue(macro.get_path_effects())
        self.assertGreater(macro.get_linewidth(), ax.lines[1].get_linewidth())
        f._convergence_legend(fig, opts, colors, labels)
        fig.canvas.draw()
        text_rows = [text.get_window_extent().y0 for text in fig.legends[0].get_texts()]
        self.assertLess(max(text_rows) - min(text_rows), 1)

    def test_compact_inset_title_and_main_curves_preserved(self):
        methods = ["MaCRO-DE-t", "DE", "JADE", "PSO", "BRO"]
        curves = [np.linspace(.004, value, 100) for value in (.001, .002, .003)]
        curves += [np.full(100, .04), np.full(100, .06)]
        table = f.pd.DataFrame([{"Archivo": "Example", "Estimador": "knn", "Optimizador": method,
                                "Curve": curve} for method, curve in zip(methods, curves)])
        table, opts, colors, _ = f.prepare_plot_groups(table, methods)
        fig, ax = f.plt.subplots()
        f._draw_dataset_convergence(ax, "Example", table, opts, colors)
        inset = ax.child_axes[0]
        self.assertEqual(inset.get_title(), "Final stage (76–100)")
        self.assertLess(inset.get_ylim()[1], .04)
        self.assertEqual(len(ax.lines), 5)
        self.assertEqual(len(inset.lines), 5)
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_ydata(), curves[methods.index(line.get_label())])
        fig.canvas.draw()

    def test_best_is_one_real_run_and_aggregation_keeps_signature(self):
        args = f.parse_args(["--epochs", "3"])
        row = {"Estimator": "knn", "Curve": [.7, .4, .15], "FitRuns": [.2, .1],
               "CurvesAll": [[.5, .3, .2], [.9, .5, .1]]}
        signature = f.build_cache_signature(args)
        for mode, expected in (("mean", np.mean(row["CurvesAll"], axis=0)), ("best", row["CurvesAll"][1]),
                               ("worst", row["CurvesAll"][0])):
            with patch.object(f, "PLOT_RUN_AGGREGATION", mode):
                table = f.build_curve_dataframe({"Example": {"DE_KNN": row}}, args, "knn")
                np.testing.assert_array_equal(table.iloc[0]["Curve"], expected)
                self.assertEqual(f.build_cache_signature(args), signature)


if __name__ == "__main__":
    unittest.main()
