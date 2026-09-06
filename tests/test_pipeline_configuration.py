"""Exercise orchestration with temporary artifacts and mocked neural/FS work."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main_best as framework


class PipelineConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = framework.parse_args([
            "--pipeline-mode", "full",
            "--miafex-dataset-root", str(self.root / "images"),
            "--miafex-checkpoint-root", str(self.root / "checkpoints"),
            "--feature-dataset-root", str(self.root / "features"),
        ])
        self.args.miafex_datasets = None  # Test discovery independently of the controlled-run defaults.
        self.args.miafex_device = "cpu"
        for name in ("A", "B"):
            for split in ("train", "test"):
                for label in ("one", "two"):
                    folder = self.root / "images" / name / split / label
                    folder.mkdir(parents=True)
                    # Discovery checks structure/extensions, not image decoding.
                    (folder / "image.jpg").touch()
        self.output = io.StringIO()
        self.addCleanup(patch.stopall)
        patch.object(framework, "MIAFEX_IMPORT_ERROR", None).start()
        self.train = patch.object(framework, "train_miafex", side_effect=self.fake_train).start()
        self.extract = patch.object(framework, "extract_miafex_features", side_effect=self.fake_extract).start()

    @staticmethod
    def write_file(path, content="artifact"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)

    def fake_train(self, **kwargs):
        return self.write_file(Path(kwargs["output_dir"]) / "miafex_checkpoint.pth")

    def fake_extract(self, **kwargs):
        content = "0,label\n10,0\n20,1\n" if Path(kwargs["data_dir"]).name == "train" else "0,label\n100,0\n200,1\n"
        path = self.write_file(Path(kwargs["output_dir"]) / "extracted_features.csv", content)
        self.write_file(Path(kwargs["output_dir"]) / "class_to_idx.json", '{"one": 0, "two": 1}')
        values = [[10], [20]] if Path(kwargs["data_dir"]).name == "train" else [[100], [200]]
        framework.np.save(Path(kwargs["output_dir"]) / "miafex_features.npy", values)
        return path

    def write_pair(self, scoped):
        self.write_file(scoped.train_features_csv, "0,label\n10,0\n20,1\n")
        self.write_file(scoped.test_features_csv, "0,label\n100,0\n200,1\n")

    def scoped(self, name="A"):
        return framework.resolve_miafex_dataset_args(self.args)[name]

    def prepare(self, args):
        with redirect_stdout(self.output):
            return framework.resolve_miafex_csv(args)

    def test_configuration_defaults_and_cli_overrides(self):
        args = framework.parse_args([])
        self.assertEqual((args.dataset_source, args.pipeline_mode), ("miafex", "feature_selection"))
        self.assertEqual(args.miafex_datasets, ["Brain_MRI"])
        self.assertEqual(args.optimizers, ["DE"])
        self.assertEqual(args.estimators, ["knn"])
        self.assertEqual((args.runs, args.epochs, args.pop_size), (1, 100, 50))
        self.assertEqual(args.transfer_functions, ["vstf_01"])
        self.assertEqual((args.miafex_epochs, args.miafex_batch_size, args.epochs), (10, 8, 100))
        self.assertEqual(args.random_state, 42)
        self.assertTrue(args.reuse_cache)
        self.assertEqual(args.parallel, "no")
        args = framework.parse_args(["--dataset-source", "mafese", "--pipeline-mode", "feature_selection",
                                     "--no-reuse-cache", "--parallel", "no", "--fs-epochs", "7",
                                     "--miafex-epochs", "3", "--n-workers", "2"])
        self.assertEqual((args.epochs, args.miafex_epochs, args.n_workers), (7, 3, 2))
        self.assertFalse(args.reuse_cache)
        self.assertEqual(args.parallel, "no")
        self.assertEqual(framework.resolve_mafese_dataset_names(args), framework.TEST_datasets_clasific_14)
        with patch.object(framework, "MIAFEX_DATASETS", ["B"]), patch.object(framework, "RUNS", 5):
            defaults = framework.parse_args([])
        self.assertEqual(defaults.miafex_datasets, ["B"])
        self.assertEqual(defaults.runs, 5)

    def test_discovery_selection_and_invalid_layouts(self):
        self.assertEqual(list(framework.resolve_miafex_dataset_args(self.args)), ["A", "B"])
        self.args.miafex_datasets = ["B"]
        self.assertEqual(list(framework.resolve_miafex_dataset_args(self.args)), ["B"])
        self.args.dataset_name = "A"  # Explicit single-name override wins over config.
        self.assertEqual(list(framework.resolve_miafex_dataset_args(self.args)), ["A"])
        self.args.dataset_name = "unknown"
        with self.assertRaises(ValueError):
            framework.resolve_miafex_dataset_args(self.args)
        (self.root / "images" / "B" / "test" / "extra").mkdir()
        self.assertEqual(list(framework.discover_miafex_datasets(self.args.miafex_dataset_root)), ["A"])

    def test_shared_single_dataset_overrides_rejected_for_multiple_datasets(self):
        self.args.features_csv = str(self.root / "shared.csv")
        with self.assertRaises(ValueError):
            framework.resolve_miafex_dataset_args(self.args)

    def test_auto_creates_isolated_artifacts_then_reuses_them(self):
        for name in ("A", "B"):
            scoped = self.scoped(name)
            self.assertEqual(self.prepare(scoped), {
                split: str(self.root / "features" / name / f"{split}_features.csv") for split in ("train", "test")
            })
            self.assertTrue((self.root / "checkpoints" / name / "miafex_checkpoint.pth").is_file())
            framework.np.testing.assert_array_equal(framework.np.load(self.root / "features" / name / "train_features.npy"), [[10], [20]])
            framework.np.testing.assert_array_equal(framework.np.load(self.root / "features" / name / "test_features.npy"), [[100], [200]])
            for split in ("train", "test"):
                self.assertTrue((self.root / "features" / name / f"{split}_class_to_idx.json").is_file())
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual(self.extract.call_count, 4)
        self.assertEqual(self.train.call_args.kwargs["batch_size"], 8)
        self.assertEqual(self.train.call_args.kwargs["device"], "cpu")
        self.assertTrue(all(Path(call.kwargs["train_root"]).name == "train" for call in self.train.call_args_list))
        self.assertEqual(self.extract.call_args.kwargs["data_dir"], str(self.root / "images/B/test"))
        self.assertFalse(self.extract.call_args.kwargs["run_ml_baselines"])
        with patch.object(framework, "MIAFEX_IMPORT_ERROR", ImportError("optional dependency unavailable")):
            self.prepare(self.scoped("A"))
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual(self.extract.call_count, 4)

    def test_auto_reuses_checkpoint_and_extracts_missing_csv(self):
        scoped = self.scoped()
        self.write_file(Path(scoped.miafex_output) / "miafex_checkpoint.pth")
        self.prepare(scoped)
        self.train.assert_not_called()
        self.assertEqual(self.extract.call_count, 2)

    def test_auto_requires_both_partitions_and_reextracts_both_if_one_is_missing(self):
        scoped = self.scoped()
        self.write_file(Path(scoped.miafex_output) / "miafex_checkpoint.pth")
        self.write_file(scoped.train_features_csv, "0,label\n-1,0\n-2,1\n")
        paths = self.prepare(scoped)
        self.train.assert_not_called()
        self.assertEqual([Path(call.kwargs["data_dir"]).name for call in self.extract.call_args_list], ["train", "test"])
        data = framework.load_miafex_feature_data(paths)
        framework.np.testing.assert_array_equal(data.X_train[:, 0], [10, 20])
        framework.np.testing.assert_array_equal(data.X_test[:, 0], [100, 200])

    def test_single_legacy_csv_is_not_a_complete_feature_dataset(self):
        self.args.dataset_name = "A"
        self.args.features_csv = self.write_file(self.root / "legacy/extracted_features.csv", "0,label\n1,0\n2,1\n")
        scoped = self.scoped()
        self.assertEqual(scoped.train_features_csv, str(self.root / "legacy/train_features.csv"))
        self.assertEqual(scoped.test_features_csv, str(self.root / "legacy/test_features.csv"))
        scoped.pipeline_mode = "feature_selection"
        with self.assertRaises(FileNotFoundError):
            self.prepare(scoped)
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_feature_selection_rejects_a_missing_test_partition(self):
        scoped = self.scoped()
        scoped.pipeline_mode = "feature_selection"
        self.write_file(scoped.train_features_csv)
        with self.assertRaises(FileNotFoundError):
            self.prepare(scoped)
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_failed_test_extraction_does_not_publish_a_partial_pair(self):
        scoped = self.scoped()
        self.write_pair(scoped)
        before = {path: Path(path).read_bytes() for path in framework.miafex_feature_paths(scoped).values()}
        scoped.extract_miafex = "yes"
        def extract(**kwargs):
            if Path(kwargs["data_dir"]).name == "test":
                raise RuntimeError("simulated test extraction failure")
            return self.fake_extract(**kwargs)
        self.extract.side_effect = extract
        with self.assertRaises(RuntimeError):
            self.prepare(scoped)
        self.assertEqual(before, {path: Path(path).read_bytes() for path in before})
        self.assertFalse(list((self.root / "features/A").glob(".extract-*")))

    def test_conflicting_extraction_label_maps_are_not_published(self):
        scoped = self.scoped()
        def extract(**kwargs):
            path = self.fake_extract(**kwargs)
            if Path(kwargs["data_dir"]).name == "test":
                self.write_file(Path(kwargs["output_dir"]) / "class_to_idx.json", '{"one": 1, "two": 0}')
            return path
        self.extract.side_effect = extract
        with self.assertRaisesRegex(ValueError, "class mappings"):
            self.prepare(scoped)
        self.assertFalse(Path(scoped.train_features_csv).exists())
        self.assertFalse(Path(scoped.test_features_csv).exists())

    def test_pair_loader_preserves_rows_and_uses_one_label_mapping(self):
        scoped = self.scoped()
        self.write_file(scoped.train_features_csv, "f1,f2,label\n10,11,zebra\n20,21,ant\n30,31,zebra\n")
        self.write_file(scoped.test_features_csv, "f1,f2,label\n100,101,zebra\n200,201,zebra\n")
        with patch.object(framework.Data, "split_train_test", side_effect=AssertionError("No outer resplit")):
            data = framework.load_miafex_feature_data(framework.miafex_feature_paths(scoped))
        framework.np.testing.assert_array_equal(data.X_train, [[10, 11], [20, 21], [30, 31]])
        framework.np.testing.assert_array_equal(data.X_test, [[100, 101], [200, 201]])
        framework.np.testing.assert_array_equal(data.y_train, [1, 0, 1])
        framework.np.testing.assert_array_equal(data.y_test, [1, 1])

    def test_pair_loader_rejects_schema_mismatch_and_unseen_test_labels(self):
        scoped = self.scoped()
        self.write_file(scoped.train_features_csv, "f1,f2,label\n10,11,zebra\n20,21,ant\n")
        self.write_file(scoped.test_features_csv, "f2,f1,label\n100,101,zebra\n")
        with self.assertRaisesRegex(ValueError, "feature columns"):
            framework.load_miafex_feature_data(framework.miafex_feature_paths(scoped))
        self.write_file(scoped.test_features_csv, "f1,f2,label\n100,101,new_class\n")
        with self.assertRaisesRegex(ValueError, "absent from the training"):
            framework.load_miafex_feature_data(framework.miafex_feature_paths(scoped))

    def test_selector_fitness_never_receives_prepared_test_rows(self):
        np = framework.np
        data = framework.Data().set_train_test(
            X_train=np.arange(40).reshape(20, 2), y_train=np.tile([0, 1], 10),
            X_test=np.arange(1000, 1008).reshape(4, 2), y_test=np.array([0, 1, 0, 1]),
        )
        optimizer = Mock()
        optimizer.history = SimpleNamespace(list_global_best_fit=[0.1])
        def solve(problem, **kwargs):
            # Exercise MAFESE's real fit/problem construction but no optimizer run.
            self.assertTrue(np.all(problem.data.X_train < 1000))
            self.assertTrue(np.all(problem.data.X_test < 1000))
            combined = np.concatenate((problem.data.X_train, problem.data.X_test))
            self.assertEqual({tuple(row) for row in combined}, {tuple(row) for row in data.X_train})
            optimizer.problem = problem
            return SimpleNamespace(solution=np.ones(2))
        optimizer.solve.side_effect = solve
        estimator = Mock()
        estimator.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
        def evaluate(**kwargs):
            self.assertIs(kwargs["data"], data)
            np.testing.assert_array_equal(kwargs["data"].X_test, np.arange(1000, 1008).reshape(4, 2))
            return {"AS_test": 0.5, "PS_test": 0.5, "RS_test": 0.5, "F1S_test": 0.5}
        with patch.object(framework, "build_optimizer", return_value="OriginalPSO"), \
                patch.object(framework.MhaSelector, "_set_optimizer", return_value=optimizer), \
                patch.object(framework.MhaSelector, "_set_estimator", return_value=estimator), \
                patch.object(framework.MhaSelector, "evaluate", side_effect=evaluate) as evaluation, \
                redirect_stdout(self.output):
            framework.run_single(data, "knn", "PSO", "vstf_01", self.args, 42)
        optimizer.solve.assert_called_once()
        evaluation.assert_called_once()

    def test_parallel_task_preserves_both_partitions(self):
        scoped = self.scoped()
        self.write_pair(scoped)
        data = framework.load_miafex_feature_data(framework.miafex_feature_paths(scoped))
        with patch.object(framework, "run_single", return_value={}) as run:
            framework.run_single_parallel_task({
                "data_split": {key: getattr(data, key) for key in ("X_train", "y_train", "X_test", "y_test")},
                "estimator": "knn", "method": "PSO", "tf": "vstf_01", "args": self.args, "seed": 42, "run": 0,
            })
        received = run.call_args.args[0]
        for key in ("X_train", "y_train", "X_test", "y_test"):
            framework.np.testing.assert_array_equal(getattr(received, key), getattr(data, key))

    def test_auto_prepares_missing_checkpoint_but_reuses_existing_csv(self):
        scoped = self.scoped()
        self.write_pair(scoped)
        self.prepare(scoped)
        self.train.assert_called_once()
        self.extract.assert_not_called()

    def test_yes_forces_stages_and_explicit_csv_filename_is_honored(self):
        scoped = self.scoped()
        scoped.train_features_csv = str(self.root / "custom/custom_train.csv")
        scoped.test_features_csv = str(self.root / "custom/custom_test.csv")
        self.prepare(scoped)
        scoped.train_miafex = scoped.extract_miafex = "yes"
        self.prepare(scoped)
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual(self.extract.call_count, 4)
        self.assertTrue(Path(scoped.train_features_csv).is_file())
        self.assertTrue(Path(scoped.test_features_csv).is_file())

    def test_no_requires_existing_artifacts_without_training(self):
        scoped = self.scoped()
        scoped.train_miafex = "no"
        with self.assertRaises(FileNotFoundError):
            self.prepare(scoped)
        scoped.extract_miafex = "no"
        with self.assertRaises(FileNotFoundError):
            self.prepare(scoped)
        self.write_pair(scoped)
        self.prepare(scoped)
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_feature_selection_never_trains_or_extracts_even_with_yes(self):
        scoped = self.scoped()
        scoped.pipeline_mode = "feature_selection"
        scoped.train_miafex = scoped.extract_miafex = "yes"
        with self.assertRaises(FileNotFoundError):
            self.prepare(scoped)
        self.write_pair(scoped)
        self.prepare(scoped)
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_feature_only_discovery_without_raw_images(self):
        self.write_file(self.root / "features/Offline/train_features.csv")
        self.write_file(self.root / "features/Offline/test_features.csv")
        self.args.miafex_dataset_root = str(self.root / "absent")
        self.args.pipeline_mode = "feature_selection"
        scoped = framework.resolve_miafex_dataset_args(self.args)
        self.assertEqual(list(scoped), ["Offline"])
        self.prepare(scoped["Offline"])
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_full_and_feature_only_share_cache_signature(self):
        scoped = self.scoped()
        signature = framework.build_cache_signature(scoped)
        scoped.pipeline_mode = "feature_selection"
        self.assertEqual(framework.build_cache_signature(scoped), signature)
        self.assertNotEqual(framework.build_cache_signature(self.scoped("B")), signature)

    def test_dataset_specific_cache_loading(self):
        self.args.output_root = str(self.root / "outputs")
        self.args.estimators = ["knn"]
        paths = framework.make_paths(self.args)
        signatures = {name: framework.build_cache_signature(self.scoped(name)) for name in ("A", "B")}
        for name in signatures:
            framework.save_cache(str(Path(paths.cache_dir) / f"{paths.exp_tag}_{name}_knn_{signatures[name]}_results.pkl"),
                                 {"sample": {"CompletedRuns": 1}})
        results = framework.load_results_from_cache(paths, self.args, ["A", "B"], signatures)
        self.assertEqual(set(results), {"A", "B"})
        self.assertEqual(results["A"]["sample"]["CompletedRuns"], 1)

    def run_main(self):
        availability = framework.BackendAvailability(False, False, None, False, False)
        backend = framework.ExecutionConfig("cpu", "auto", "sklearn", "cpu", availability)
        with patch.object(framework, "parse_args", return_value=self.args), \
                patch.object(framework, "resolve_execution_config", return_value=backend), \
                patch.object(framework, "print_backend_report"), redirect_stdout(self.output):
            framework.main()

    def test_extract_mode_exits_before_feature_selection_or_results(self):
        self.args.pipeline_mode = "extract"
        with patch.object(framework, "resolve_optimizers", side_effect=AssertionError("FS must not start")), \
                patch.object(framework, "make_paths", side_effect=AssertionError("No results expected")), \
                patch.object(framework, "run_single", side_effect=AssertionError("FS must not start")):
            self.run_main()
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual(self.extract.call_count, 4)

    def test_full_pipeline_then_feature_only_resumes_both_dataset_caches(self):
        self.args.output_root = str(self.root / "results")
        self.args.optimizers = ["PSO"]
        self.args.estimators = ["knn"]
        self.args.runs = 1
        self.args.epochs = 2
        self.args.parallel = "no"
        result = {"as_test": 75.0, "ps_test": 0.75, "rs_test": 0.75,
                  "f1_test": 0.75, "fit_final": 0.2, "n_features": 1,
                  "runtime": 0.01, "curve": [0.3, 0.2]}
        with patch.object(framework, "resolve_optimizers", return_value=["OriginalPSO"]), \
                patch.object(framework, "optimizer_display_label", return_value="PSO"), \
                patch.object(framework.Data, "split_train_test", side_effect=AssertionError("MIAFEx must never resplit its prepared partitions")), \
                patch.object(framework, "run_single", return_value=result) as run, \
                patch.object(framework, "export_global_excel", return_value=[]) as export, \
                patch.object(framework, "generate_summary_dataframe", return_value=framework.pd.DataFrame()), \
                patch.object(framework, "generate_seven_global_charts", return_value=[]):
            self.run_main()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(set(export.call_args.args[0]), {"A", "B"})
            self.assertEqual(self.train.call_count, 2)
            self.assertEqual(self.extract.call_count, 4)
            for call in run.call_args_list:
                data = call.args[0]
                framework.np.testing.assert_array_equal(data.X_train[:, 0], [10, 20])
                framework.np.testing.assert_array_equal(data.X_test[:, 0], [100, 200])
                framework.np.testing.assert_array_equal(data.y_train, [0, 1])
                framework.np.testing.assert_array_equal(data.y_test, [0, 1])
            self.args.pipeline_mode = "feature_selection"
            self.run_main()
            # Completed FS runs are resumed from their per-dataset caches.
            self.assertEqual(run.call_count, 2)
            self.assertEqual(self.train.call_count, 2)
            self.assertEqual(self.extract.call_count, 4)

    def test_figures_only_bypasses_neural_preparation(self):
        self.args.figures_only = True
        self.args.output_root = str(self.root / "results")
        with patch.object(framework, "resolve_optimizers", return_value=["OriginalPSO"]), \
                patch.object(framework, "resolve_miafex_csv", side_effect=AssertionError("No neural preparation")), \
                patch.object(framework, "regenerate_figures_from_cache", return_value=("summary.csv", [])) as regenerate:
            self.run_main()
        regenerate.assert_called_once()
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_mafese_extract_is_rejected_without_neural_work(self):
        self.args.dataset_source = "mafese"
        self.args.pipeline_mode = "extract"
        with self.assertRaises(ValueError):
            self.run_main()
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_mafese_keeps_its_existing_holdout_split(self):
        self.args.dataset_source = "mafese"
        self.args.output_root = str(self.root / "results")
        self.args.optimizers = ["PSO"]
        self.args.estimators = ["knn"]
        self.args.runs = 1
        self.args.epochs = 2
        self.args.parallel = "no"
        X = framework.np.arange(40).reshape(20, 2)
        y = framework.np.tile([0, 1], 10)
        real_split = framework.Data.split_train_test
        def split(data, **kwargs):
            return real_split(data, **kwargs)
        result = {"as_test": 75.0, "ps_test": 0.75, "rs_test": 0.75,
                  "f1_test": 0.75, "fit_final": 0.2, "n_features": 1,
                  "runtime": 0.01, "curve": [0.3, 0.2]}
        with patch.object(framework, "resolve_mafese_dataset_names", return_value=["Internal"]), \
                patch.object(framework, "get_dataset", return_value=SimpleNamespace(X=X, y=y)), \
                patch.object(framework, "resolve_optimizers", return_value=["OriginalPSO"]), \
                patch.object(framework, "optimizer_display_label", return_value="PSO"), \
                patch.object(framework.Data, "split_train_test", autospec=True, side_effect=split) as splitting, \
                patch.object(framework, "run_single", return_value=result) as run, \
                patch.object(framework, "export_global_excel", return_value=[]), \
                patch.object(framework, "generate_summary_dataframe", return_value=framework.pd.DataFrame()), \
                patch.object(framework, "generate_seven_global_charts", return_value=[]):
            self.run_main()
        splitting.assert_called_once()
        self.assertEqual(splitting.call_args.kwargs["test_size"], self.args.test_size)
        self.assertEqual(splitting.call_args.kwargs["random_state"], self.args.random_state)
        framework.np.testing.assert_array_equal(splitting.call_args.kwargs["stratify"], y)
        self.assertEqual((len(run.call_args.args[0].X_train), len(run.call_args.args[0].X_test)), (16, 4))
        self.train.assert_not_called()
        self.extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
