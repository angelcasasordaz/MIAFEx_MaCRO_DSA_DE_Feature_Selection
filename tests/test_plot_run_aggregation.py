"""Aligned representative runs, preserved means/distributions and cache science."""
import copy
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main_best as f


class PlotAggregationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(f.plt.close, "all")
        self.args = f.parse_args(["--dataset-source", "mafese", "--optimizers", "DSADE",
                                  "MaCRO-DE-t", "MaCRO-DE-t-v2", "--epochs", "3", "--runs", "3"])
        self.row = f.build_label_payload(
            "knn", [99, 75, 88], [.99, .7, .8], [.9, .6, .7], [.95, .65, .75],
            [.3, .1, .2], [1, 7, 4], [1, 9, 4],
            [[.8, .5, .3], [.9, .4, .1], [.7, .3, .2]], 3,
            completed_run_ids=[8, 2, 5],
        )
        self.results = {"Tiny": {"MaCRO-DE-t-v2_KNN": self.row}}

    def test_best_worst_take_every_value_from_one_real_noncontiguous_run(self):
        before = pickle.dumps(self.results)
        fields = {"AS_test": "Acc", "PS_test": "PS", "RS_test": "RS", "F1_test": "F1",
                  "N_Features_Selected": "Feat", "Runtime": "Time"}
        for mode, index, run_id in (("best", 0, 2), ("worst", 2, 8)):
            with patch.object(f, "PLOT_RUN_AGGREGATION", mode), \
                    patch.object(f, "build_optimizer", side_effect=AssertionError("No science")), \
                    patch.object(f, "run_single", side_effect=AssertionError("No runs")):
                self.assertEqual(self.row["CompletedRunIDs"][f._plot_run_index(self.row)], run_id)
                values = f.generate_plot_dataframe(self.results, self.args).iloc[0]
                for column, metric in fields.items():
                    expected = self.row[f"{metric}Runs"][index] / (100 if metric == "Acc" else 1)
                    self.assertEqual(values[column], expected)
                curve = f.build_curve_dataframe(self.results, self.args, "knn").iloc[0]["Curve"]
                np.testing.assert_array_equal(curve, self.row["CurvesAll"][index])
                self.assertEqual(curve[-1], self.row["FitRuns"][index])
                distributions = f.build_run_level_dataframe(self.results, self.args, "knn")
                self.assertEqual(len(distributions), 3)
                np.testing.assert_array_equal(distributions.AS_test, self.row["AccRuns"] / 100)
        self.assertEqual(pickle.dumps(self.results), before)

    def test_global_figure_entry_uses_representative_metrics_and_all_run_distributions(self):
        summary = f.generate_summary_dataframe(self.results, self.args)
        before = pickle.dumps(self.results)
        with tempfile.TemporaryDirectory() as directory:
            for mode, accuracy in (("best", .75), ("worst", .99),
                                   ("mean", self.row["AccMean"] / 100)):
                with self.subTest(mode=mode), patch.object(f, "PLOT_RUN_AGGREGATION", mode), \
                        patch.object(f, "PLOT_ESTIMATORS", ["knn"]), \
                        patch.object(f, "generate_classifier_metric_grid_chart", return_value=None) as grid, \
                        patch.object(f, "generate_individual_dataset_charts", return_value=[]) as individual, \
                        patch.object(f, "_draw_dataset_radar") as radar, \
                        patch.object(f, "_draw_dataset_features_runtime") as features, \
                        patch.object(f, "generate_global_accuracy_boxplot") as distribution, \
                        patch.object(f, "generate_global_features_runtime") as global_features, \
                        patch.object(f, "_save_chart"):
                    f.generate_seven_global_charts(summary, self.results, directory,
                                                  self.args.optimizers, self.args)
                    for table in (grid.call_args.args[0], individual.call_args.args[0],
                                  radar.call_args.args[2], features.call_args.args[2],
                                  global_features.call_args.args[0]):
                        self.assertEqual(table.iloc[0]["AS_test"], accuracy)
                    table = distribution.call_args.args[0]
                    self.assertEqual(len(table), 3)
                    np.testing.assert_array_equal(table.AS_test, self.row["AccRuns"] / 100)
                f.plt.close("all")
        self.assertEqual(pickle.dumps(self.results), before)

    def test_mean_is_existing_summary_and_mean_convergence_and_tables_stay_mean(self):
        summary = f.generate_summary_dataframe(self.results, self.args)
        for mode in ("best", "worst", "mean"):
            with patch.object(f, "PLOT_RUN_AGGREGATION", mode):
                f.pd.testing.assert_frame_equal(summary, f.generate_summary_dataframe(self.results, self.args))
                self.assertEqual(len(f.build_run_level_dataframe(self.results, self.args, "knn")), 3)
                if mode == "mean":
                    f.pd.testing.assert_frame_equal(summary, f.generate_plot_dataframe(self.results, self.args))
                    curve = f.build_curve_dataframe(self.results, self.args, "knn").iloc[0]["Curve"]
                    np.testing.assert_array_equal(curve, self.row["Curve"])

    def test_option_does_not_change_any_scientific_or_legacy_identity(self):
        with patch.object(f, "get_dataset", return_value=f.Data([[1], [2]], [0, 1])):
            identities = {m: f.build_combination_identity(self.args, "Tiny", "knn", m, "vstf_01")
                          for m in self.args.optimizers}
            signature = f.build_cache_signature(self.args)
            legacy = f.build_legacy_group_signature(self.args)
            before = copy.deepcopy(vars(self.args))
            for mode in ("best", "worst", "mean"):
                with patch.object(f, "PLOT_RUN_AGGREGATION", mode):
                    for exp in (604, 605):
                        self.args.exp_id = exp
                        for method, identity in identities.items():
                            self.assertEqual(identity, f.build_combination_identity(self.args, "Tiny", "knn", method, "vstf_01"))
                        self.assertEqual(signature, f.build_cache_signature(self.args))
                        self.assertEqual(legacy, f.build_legacy_group_signature(self.args))
            self.args.exp_id = before["exp_id"]
            self.assertEqual(vars(self.args), before)

    def test_invalid_mode_and_misaligned_runs_fail_without_guessing(self):
        with patch.object(f, "PLOT_RUN_AGGREGATION", "invalid"), self.assertRaisesRegex(ValueError, "PLOT_RUN_AGGREGATION"):
            f.generate_plot_dataframe(self.results, self.args)
        for changes in ({"AccRuns": [90]}, {"CompletedRunIDs": [2, 2, 5]}, {"FitRuns": [np.nan, .1, .2]}):
            bad = copy.deepcopy(self.results)
            bad["Tiny"]["MaCRO-DE-t-v2_KNN"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                f.generate_plot_dataframe(bad, self.args)


if __name__ == "__main__":
    unittest.main()
