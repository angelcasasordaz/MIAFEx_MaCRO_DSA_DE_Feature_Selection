"""Lightweight plotting/report-selection regressions; no experiment execution."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import main_best as framework


def panel_snapshot(ax):
    """Compare plotted values, styling and axes independently of panel geometry."""
    return {
        "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
        "xlim": ax.get_xlim(), "ylim": ax.get_ylim(),
        "xticks": list(ax.get_xticks()), "yticks": list(ax.get_yticks()),
        "labels": [tick.get_text() for tick in ax.get_xticklabels()],
        "lines": [(line.get_xydata().tolist(), matplotlib.colors.to_rgba(line.get_color()), line.get_linestyle(),
                   line.get_linewidth(), line.get_zorder()) for line in ax.lines],
        "patches": [(patch.get_path().vertices.tolist(), patch.get_facecolor(),
                     patch.get_alpha(), patch.get_hatch()) for patch in ax.patches],
        "bars": [[(bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height())
                  for bar in container] for container in ax.containers],
        "text": [text.get_text() for text in ax.texts],
    }


class IndividualPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(framework.plt.close, "all")
        self.args = framework.parse_args([
            "--optimizers", "MaCRO-DE", "DE", "PSO", "--estimators", "knn", "svm",
            "--output-root", self.temp.name,
        ])
        self.results = {}
        for di, dataset in enumerate(("Brain_MRI", "Other_Dataset")):
            self.results[dataset] = {}
            for ci, classifier in enumerate(self.args.estimators):
                for mi, method in enumerate(self.args.optimizers):
                    value = 0.7 + di * 0.02 + ci * 0.04 + mi * 0.01
                    self.results[dataset][f"{method}_{classifier.upper()}"] = {
                        "Estimator": classifier, "AccMean": value * 100, "PSMean": value,
                        "RSMean": value, "F1Mean": value, "FeatMean": 10 + mi,
                        "TimeMean": 2 + mi, "Curve": [value, value / 2, value / 3],
                        "AccRuns": [value * 100, (value + .01) * 100],
                        "PSRuns": [value, value + .01], "RSRuns": [value, value + .01],
                        "F1Runs": [value, value + .01], "FeatRuns": [10 + mi, 12 + mi],
                        "TimeRuns": [2 + mi, 3 + mi], "FitRuns": [value, value + .01],
                    }
        self.df = framework.generate_summary_dataframe(self.results, self.args)

    def capture(self, fig, directory, name):
        self.figures[name] = [panel_snapshot(ax) for ax in fig.axes]
        Path(directory, name).touch()  # The classifier grid is subsequently renamed.
        framework.plt.close(fig)

    def test_individual_panels_match_combined_for_each_classifier(self):
        before = copy.deepcopy(self.results)
        for estimator in ("knn", "svm"):
            self.figures = {}
            with self.subTest(estimator=estimator), \
                    patch.object(framework, "PLOT_ESTIMATORS", [estimator]), \
                    patch.object(framework, "_save_chart", side_effect=self.capture):
                saved = framework.generate_seven_global_charts(
                    self.df, self.results, self.temp.name, self.args.optimizers, self.args,
                )
            self.assertEqual(len(saved), 15)
            self.assertEqual(len(self.figures["09_resultados_clasificador_metrica_todos_datasets.png"]), 4)
            for family in ("02_radar", "03_features_runtime", "05_convergence"):
                combined = self.figures[f"{family}_por_dataset_knn.png"]
                for index, dataset in enumerate(sorted(self.results)):
                    single = self.figures[f"{family}_{dataset}_{estimator}.png"]
                    expected = [combined[index]]
                    if family == "03_features_runtime":
                        expected.append(combined[len(self.results) + index])
                    self.assertEqual(single, expected)
        self.assertEqual(self.results, before)

    def test_both_classifiers_and_missing_curves(self):
        self.results["Brain_MRI"]["DE_KNN"]["Curve"] = []
        self.figures = {}
        with patch.object(framework, "PLOT_ESTIMATORS", ["knn", "svm"]), \
                patch.object(framework, "_save_chart", side_effect=self.capture):
            saved = framework.generate_individual_dataset_charts(
                self.df, self.results, self.temp.name, self.args.optimizers, self.args,
            )
        self.assertEqual(len(saved), 12)
        self.assertEqual(len(self.figures["05_convergence_Brain_MRI_knn.png"][0]["lines"]), 2)
        self.assertEqual(len(self.figures["05_convergence_Brain_MRI_svm.png"][0]["lines"]), 3)

    def test_selection_filters_all_reports_without_changing_science(self):
        paths = framework.make_paths(self.args)
        original_args = copy.deepcopy(vars(self.args))
        original_results = copy.deepcopy(self.results)
        original_estimators = list(framework.ESTIMATORS)
        signature = framework.build_cache_signature(self.args)
        for selected in (["knn"], ["svm"], ["knn", "svm"]):
            with self.subTest(selected=selected), \
                    patch.object(framework, "PLOT_ESTIMATORS", selected), \
                    patch.object(framework, "generate_seven_global_charts", return_value=[]) as charts, \
                    redirect_stdout(io.StringIO()):
                framework.export_reporting_outputs(
                    paths, self.args, list(self.results), self.results, self.results,
                )
                self.assertEqual(framework.build_cache_signature(self.args), signature)
            self.assertEqual(set(charts.call_args.args[0].Estimador), set(selected))
            summary = framework.pd.read_csv(Path(paths.res_dir, "RESUMEN_GRAFICAS_EXP604.csv"))
            self.assertEqual(set(summary.Estimador), set(selected))
            analysis = framework.pd.read_excel(Path(paths.res_dir, "Full_Friedman_Analysis_EXP604.xlsx"))
            self.assertEqual(set(analysis.Classifier), set(selected))
            for filename, start_column in (("Global_Results_EXP604.xlsx", 1),
                                            ("Statistical_Results_EXP604.xlsx", 2)):
                tables = framework.pd.read_excel(Path(paths.res_dir, filename), sheet_name=None)
                for table in tables.values():
                    for column in table.columns[start_column:]:
                        self.assertTrue(any(est.upper() in column for est in selected), column)
                        self.assertFalse(any(est.upper() in column for est in {"knn", "svm"} - set(selected)))
        self.assertEqual(vars(self.args), original_args)
        self.assertEqual(framework.ESTIMATORS, original_estimators)
        self.assertEqual(self.results, original_results)


if __name__ == "__main__":
    unittest.main()
