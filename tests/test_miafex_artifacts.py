"""Artifact/version safety with temporary files and mocked neural generation."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import main_best as framework
import miafex_artifacts as artifacts


class ArtifactVersionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.options = [
            "--miafex-artifact-tag", "test_v1", "--dataset-name", "Sample",
            "--miafex-dataset-root", str(self.root / "images"),
            "--miafex-checkpoint-root", str(self.root / "checkpoints"),
            "--feature-dataset-root", str(self.root / "features"),
            "--miafex-provenance-root", str(self.root / "provenance"),
            "--train-miafex", "auto", "--extract-miafex", "auto",
        ]
        for split in ("train", "test"):
            for label in ("a", "b"):
                directory = self.root / "images" / "Sample" / split / label
                directory.mkdir(parents=True)
                (directory / "sample.jpg").write_bytes(f"{split}-{label}".encode())
        self.args = self.scoped()
        self.train = self.enterContext(patch.object(framework, "train_miafex", side_effect=self.fake_train))
        self.extract = self.enterContext(patch.object(framework, "extract_miafex_features", side_effect=self.fake_extract))
        self.enterContext(patch.object(framework, "MIAFEX_IMPORT_ERROR", None))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def scoped(self, extra=()):
        args = framework.parse_args(self.options + list(extra))
        args.miafex_device = "cpu"
        return framework.resolve_miafex_dataset_args(args)["Sample"]

    @staticmethod
    def fake_train(**kwargs):
        path = Path(kwargs["output_dir"]) / "miafex_checkpoint.pth"
        path.write_bytes(b"fixed checkpoint")
        return str(path)

    @staticmethod
    def fake_extract(**kwargs):
        path = Path(kwargs["output_dir"]) / "extracted_features.csv"
        path.write_text("0,label\n1,0\n2,1\n")
        (path.parent / "class_to_idx.json").write_text('{"a": 0, "b": 1}')
        return str(path)

    def prepare(self, args=None):
        return framework.resolve_miafex_csv(args or self.args)

    def snapshot(self):
        return {str(path): path.read_bytes() for root in ("checkpoints", "features", "provenance")
                for path in (self.root / root).rglob("*") if path.is_file()}

    def test_exp_id_never_changes_artifact_roots(self):
        for exp_id in (607, 608, 609, 610):
            args = framework.parse_args(["--exp-id", str(exp_id)])
            self.assertEqual(args.miafex_artifact_tag, "exp607")
            self.assertEqual(args.miafex_checkpoint_root, "checkpoints/miafex_exp607")
            self.assertEqual(args.feature_dataset_root, "datasets_features/miafex_exp607")
            self.assertEqual((args.train_miafex, args.extract_miafex), ("auto", "auto"))
        args = framework.parse_args(["--exp-id", "610", "--miafex-artifact-tag", "paperlike_v2"])
        self.assertEqual(args.miafex_checkpoint_root, "checkpoints/miafex_paperlike_v2")
        self.assertEqual(args.feature_dataset_root, "datasets_features/miafex_paperlike_v2")
        self.assertEqual(args.miafex_provenance_root, "artifact_provenance/miafex_paperlike_v2")
        with self.assertRaises(ValueError):
            framework.parse_args(["--miafex-artifact-tag", "../escape"])

    def test_one_version_reused_across_experiments_and_all_twenty_runs(self):
        self.prepare()
        before = self.snapshot()
        for exp_id in (608, 609, 610):
            args = self.scoped(["--exp-id", str(exp_id)])
            self.assertEqual(args.runs, 20)
            self.prepare(args)
        self.assertEqual(self.train.call_count, 1)
        self.assertEqual(self.extract.call_count, 2)
        self.assertEqual(before, self.snapshot())
        text = self.output.getvalue()
        for state in ("TRAIN NEW", "REUSE CHECKPOINT", "EXTRACT NEW", "REUSE FEATURES", "VALIDATED"):
            self.assertIn(state, text)
        metadata = artifacts.read_metadata(self.args)
        self.assertEqual(metadata["configuration"]["image_source"]["train"]["image_count"], 2)
        self.assertEqual(set(metadata["artifact_sha256"]), {"checkpoint", "train_features_csv", "test_features_csv"})

    def test_configuration_conflicts_fail_without_generation_or_overwrite(self):
        self.prepare()
        before = self.snapshot()
        for flag, value in (("--miafex-epochs", "51"), ("--miafex-batch-size", "16"),
                            ("--miafex-learning-rate", "0.0002"), ("--miafex-artifact-tag", "other")):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "validation failed"):
                self.prepare(self.scoped([flag, value]))
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.train.call_count, 1)
        self.assertEqual(self.extract.call_count, 2)

    def test_neural_code_changes_invalidate_reuse(self):
        self.prepare()
        before = self.snapshot()
        real_digest = artifacts.file_digest
        def changed_digest(path):
            return "changed" if Path(path).name == "miafex_model.py" else real_digest(path)
        with patch.object(artifacts, "file_digest", side_effect=changed_digest):
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                self.prepare()
        self.assertEqual(before, self.snapshot())

    def test_split_or_image_content_changes_invalidate_reuse(self):
        self.prepare()
        image = self.root / "images/Sample/train/a/sample.jpg"
        image.write_bytes(b"changed content")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "image_source"):
            self.prepare()
        self.assertEqual(before, self.snapshot())

    def test_tampered_or_missing_artifacts_fail_even_in_feature_only_mode(self):
        self.prepare()
        args = self.scoped(["--pipeline-mode", "feature_selection"])
        for path in (args.train_features_csv, args.test_features_csv, artifacts.artifact_paths(args)["checkpoint"]):
            path = Path(path)
            original = path.read_bytes()
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.prepare(args)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.prepare(args)
            path.write_bytes(original)

    def test_unversioned_files_are_never_automatically_adopted(self):
        path = Path(self.args.miafex_output) / "miafex_checkpoint.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unknown history")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "no trusted provenance"):
            self.prepare()
        self.assertEqual(before, self.snapshot())
        self.train.assert_not_called()
        self.extract.assert_not_called()

    def test_yes_requires_unused_version_and_feature_only_ignores_stage_flags(self):
        args = self.scoped(["--train-miafex", "yes", "--extract-miafex", "yes"])
        self.prepare(args)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "TRAIN NEW"):
            self.prepare(args)
        args.train_miafex = "no"
        with self.assertRaisesRegex(ValueError, "EXTRACT NEW"):
            self.prepare(args)
        args.pipeline_mode = "feature_selection"
        args.train_miafex = "yes"
        self.prepare(args)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.train.call_count, 1)
        self.assertEqual(self.extract.call_count, 2)

    def test_no_requires_features_and_can_extract_from_validated_checkpoint(self):
        args = self.scoped(["--train-miafex", "no"])
        with self.assertRaises(FileNotFoundError):
            self.prepare(args)
        # Simulate a version whose checkpoint has finished but extraction has
        # not started. No inference from unversioned files is permitted.
        metadata = artifacts.new_metadata(args)
        path = Path(artifacts.artifact_paths(args)["checkpoint"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixed checkpoint")
        artifacts.record_outputs(args, metadata)
        self.prepare(args)
        self.train.assert_not_called()
        self.assertEqual(self.extract.call_count, 2)
        args.extract_miafex = "no"
        self.prepare(args)
        self.assertEqual(self.extract.call_count, 2)

    def test_failed_extraction_preserves_checkpoint_and_publishes_no_features(self):
        calls = 0
        def broken_extract(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test extraction failed")
            return self.fake_extract(**kwargs)
        self.extract.side_effect = broken_extract
        with self.assertRaises(RuntimeError):
            self.prepare()
        metadata = artifacts.validate_existing(self.args)
        self.assertEqual(set(metadata["artifact_sha256"]), {"checkpoint"})
        self.assertFalse(Path(self.args.train_features_csv).exists())
        self.assertFalse(Path(self.args.test_features_csv).exists())
        self.extract.side_effect = self.fake_extract
        self.prepare()
        self.assertEqual(self.train.call_count, 1)

    def test_offline_images_support_validated_feature_only_reuse(self):
        self.prepare()
        shutil.rmtree(self.root / "images/Sample")
        self.prepare(self.scoped(["--pipeline-mode", "feature_selection"]))
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_corrupt_metadata_and_concurrent_generation_fail(self):
        with artifacts.artifact_lock(self.args):
            with self.assertRaisesRegex(ValueError, "holds"):
                self.prepare()
        self.prepare()
        metadata = artifacts.metadata_path(self.args)
        metadata.write_text('{"schema_version": 1}')
        with self.assertRaisesRegex(ValueError, "invalid provenance"):
            self.prepare()

    def test_startup_and_completed_cache_branch_validate_before_reporting(self):
        self.prepare()
        before = self.snapshot()
        args = framework.parse_args(self.options + ["--exp-id", "610", "--pipeline-mode", "feature_selection"])
        backend = framework.ExecutionConfig("cpu", "auto", "sklearn", "cpu",
                                            framework.BackendAvailability(False, False, None, False, False))
        paths = framework.Paths("EXP610", str(self.root / "figures"), str(self.root / "results"), str(self.root / "cache"))
        with patch.object(framework, "parse_args", return_value=args), \
                patch.object(framework, "resolve_execution_config", return_value=backend), \
                patch.object(framework, "print_backend_report"), \
                patch.object(framework, "make_paths", return_value=paths), \
                patch.object(framework, "experiment_cache_is_complete", return_value=True) as complete, \
                patch.object(framework, "regenerate_figures_from_cache", return_value=("summary.csv", [])) as report, \
                patch.object(framework, "run_single", side_effect=AssertionError("No FS execution")):
            framework.main()
            complete.assert_called_once()
            report.assert_called_once()
            complete.reset_mock()
            report.reset_mock()
            args.miafex_learning_rate *= 2
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                framework.main()
            complete.assert_not_called()
            report.assert_not_called()
        self.assertEqual(before, self.snapshot())
        for label in ("EXP_ID: 610", "MIAFEX_ARTIFACT_TAG: test_v1", "MIAFEx checkpoint root:", "MIAFEx feature-dataset root:"):
            self.assertIn(label, self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
