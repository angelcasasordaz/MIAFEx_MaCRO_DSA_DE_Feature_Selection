"""Checkpoint/resume tests with tiny deterministic workers; no neural or FS work."""

from concurrent.futures import Future
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import main_best as framework


class SimulatedInterruption(Exception):
    pass


def synthetic_result(seed):
    rng = framework.np.random.RandomState(seed)
    values = rng.uniform(0.6, 0.9, 4)
    return {
        "as_test": float(100 * values[0]), "ps_test": float(values[1]),
        "rs_test": float(values[2]), "f1_test": float(values[3]),
        "fit_final": float(1 - values[0]), "n_features": int(rng.randint(1, 4)),
        # Fixed synthetic runtime lets us compare every cache field exactly.
        "runtime": 0.01 + seed / 100000,
        "curve": framework.np.array([0.5, 0.4, 1 - values[0]]),
    }


def synthetic_parallel_task(task):
    """Runs 2 and 4 finish first; other runs wait on a test-only release file."""
    args, run = task["args"], task["run"]
    marker = Path(args.test_worker_dir) / f"run-{run}.txt"
    marker.write_text(str(task["seed"]))
    if args.test_interrupt and run not in (1, 3):
        deadline = time.monotonic() + 15
        while not Path(args.test_release).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Test did not release blocked synthetic worker")
            time.sleep(0.01)
    return run, synthetic_result(task["seed"])


class ParallelResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="parallel-resume-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = framework.parse_args([
            "--dataset-name", "Synthetic", "--pipeline-mode", "full",
            "--dataset-root", str(self.root / "images"),
            "--miafex-checkpoint-root", str(self.root / "checkpoints"),
            "--feature-dataset-root", str(self.root / "features"),
            "--output-root", str(self.root / "results"),
            "--optimizers", "PSO", "--estimators", "knn", "--runs", "6",
            "--fs-epochs", "3", "--n-workers", "4", "--parallel", "yes",
        ])
        for split in ("train", "test"):
            for label in ("a", "b"):
                directory = self.root / "images" / split / label
                directory.mkdir(parents=True)
                (directory / "image.jpg").touch()
        scoped = framework.resolve_miafex_dataset_args(self.args)["Synthetic"]
        features = Path(scoped.train_features_csv).parent
        features.mkdir(parents=True)
        Path(scoped.train_features_csv).write_text("f0,f1,label\n1,2,0\n3,4,1\n")
        Path(scoped.test_features_csv).write_text("f0,f1,label\n100,101,0\n102,103,1\n")
        checkpoint = Path(scoped.miafex_output) / "miafex_checkpoint.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"Existing checkpoint, never loaded by these tests")
        self.neural_artifacts = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (Path(scoped.train_features_csv), Path(scoped.test_features_csv), checkpoint)
        }
        self.output = io.StringIO()

    def invoke_main(self, args, save=None):
        backend = framework.ExecutionConfig(
            "cpu", "auto", "sklearn", "cpu",
            framework.BackendAvailability(False, False, None, False, False),
        )
        with patch.object(framework, "parse_args", return_value=copy.deepcopy(args)), \
                patch.object(framework, "resolve_execution_config", return_value=backend), \
                patch.object(framework, "print_backend_report"), \
                patch.object(framework, "run_single_parallel_task", new=synthetic_parallel_task), \
                patch.object(framework, "train_miafex", side_effect=AssertionError("No neural training")), \
                patch.object(framework, "extract_miafex_features", side_effect=AssertionError("Reuse CSVs")), \
                patch.object(framework, "run_single", side_effect=AssertionError("No real FS")), \
                patch.object(framework, "save_cache", new=save or framework.save_cache), \
                patch.object(framework, "export_global_excel", return_value=[]), \
                patch.object(framework, "generate_seven_global_charts", return_value=[]), \
                redirect_stdout(self.output):
            framework.main()

    def configure_invocation(self, args, name, interrupt=False):
        args.test_worker_dir = str(self.root / name)
        Path(args.test_worker_dir).mkdir()
        args.test_release = str(self.root / "release")
        args.test_interrupt = interrupt

    def result_cache(self, args):
        return next(Path(args.output_root).glob("Results/*/cache/*_results.pkl"))

    def executed_runs(self, args):
        return {int(path.stem.split("-")[1]): int(path.read_text())
                for path in Path(args.test_worker_dir).glob("run-*.txt")}

    def assert_payload_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            framework.np.testing.assert_equal(actual[key], expected[key], err_msg=key)

    def test_interrupted_noncontiguous_runs_resume_only_missing_and_match_uninterrupted(self):
        self.configure_invocation(self.args, "interrupted", interrupt=True)
        real_save = framework.save_cache
        snapshots = []

        def interrupt_after_two(path, payload):
            real_save(path, payload)
            if str(path).endswith("_results.pkl"):
                row = next(iter(framework.load_cache(path).values()))
                snapshots.append(row["CompletedRunIDs"])
                if row["CompletedRunIDs"] == [1, 3]:
                    # Release the pool only after durable non-contiguous results exist.
                    Path(self.args.test_release).touch()
                    raise SimulatedInterruption("Stop after runs 2 and 4 were checkpointed")

        with self.assertRaises(SimulatedInterruption):
            self.invoke_main(self.args, save=interrupt_after_two)
        self.assertEqual(len(snapshots[0]), 1)
        self.assertEqual(snapshots[-1], [1, 3])
        cache = self.result_cache(self.args)
        interrupted = next(iter(framework.load_cache(cache).values()))
        self.assertEqual(interrupted["CompletedRunIDs"], [1, 3])
        self.assertEqual(interrupted["CompletedRuns"], 2)

        self.configure_invocation(self.args, "resumed")
        self.invoke_main(self.args)
        missing = [0, 2, 4, 5]
        self.assertEqual(self.executed_runs(self.args), {run: self.args.seed_base + run for run in missing})
        self.assertIn("completed=2/6 | recovered=2,4 | missing=1,3,5..6", self.output.getvalue())
        self.assertRegex(self.output.getvalue(), r"Run 0[1-6]/6 \| Accuracy=.* \| F1=.* \| Fitness=.* \| Features=.* \| run time=.* \| total elapsed=")
        resumed = next(iter(framework.load_cache(cache).values()))
        self.assertEqual(resumed["CompletedRunIDs"], list(range(6)))

        baseline_args = copy.deepcopy(self.args)
        baseline_args.output_root = str(self.root / "baseline-results")
        self.configure_invocation(baseline_args, "uninterrupted")
        self.invoke_main(baseline_args)
        self.assertEqual(self.executed_runs(baseline_args), {run: self.args.seed_base + run for run in range(6)})
        baseline = next(iter(framework.load_cache(self.result_cache(baseline_args)).values()))
        self.assert_payload_equal(resumed, baseline)
        # Verify the original aggregation, independent of completion order and new metadata.
        rows = [synthetic_result(self.args.seed_base + run) for run in range(6)]
        original = framework.build_label_payload(
            "knn", *[[row[key] for row in rows] for key in (
                "as_test", "ps_test", "rs_test", "f1_test", "fit_final", "n_features", "runtime", "curve",
            )], self.args.epochs,
        )
        self.assert_payload_equal({k: v for k, v in resumed.items() if k != "CompletedRunIDs"}, original)
        self.assertEqual(self.neural_artifacts, {
            path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.neural_artifacts
        })

        self.configure_invocation(self.args, "already-complete")
        self.invoke_main(self.args)
        self.assertEqual(self.executed_runs(self.args), {})
        self.assertIn("completed=6/6 | recovered=1..6 | missing=none", self.output.getvalue())

    def test_legacy_contiguous_cache_resumes_without_repeating_completed_runs(self):
        scoped = framework.resolve_miafex_dataset_args(self.args)["Synthetic"]
        scoped.optimizers = framework.resolve_optimizers(scoped)
        paths = framework.make_paths(self.args)
        signature = framework.build_cache_signature(scoped)
        label = framework.build_alg_label(scoped.optimizers[0], "vstf_01", "knn", False, False)
        rows = [synthetic_result(self.args.seed_base + run) for run in range(2)]
        legacy = framework.build_label_payload(
            "knn", *[[row[key] for row in rows] for key in (
                "as_test", "ps_test", "rs_test", "f1_test", "fit_final", "n_features", "runtime", "curve",
            )], self.args.epochs,
        )
        self.assertNotIn("CompletedRunIDs", legacy)
        # Exercise recovery from the old partial file with no final file present.
        path = Path(paths.cache_dir) / f"{paths.exp_tag}_Synthetic_knn_{signature}_progress.pkl"
        framework.save_cache(path, {label: legacy})
        self.configure_invocation(self.args, "legacy-resumed")
        self.invoke_main(self.args)
        self.assertEqual(self.executed_runs(self.args), {run: self.args.seed_base + run for run in range(2, 6)})
        result = framework.load_cache(self.result_cache(self.args))[label]
        self.assertEqual(result["CompletedRunIDs"], list(range(6)))
        framework.np.testing.assert_array_equal(result["AccRuns"][:2], legacy["AccRuns"])

    def test_heartbeat_every_sixty_seconds_even_without_completions(self):
        self.args.n_workers = 1
        first, later = Future(), Future()
        executor = Mock()
        executor.submit.side_effect = [first, later]
        executor_context = Mock()
        executor_context.__enter__ = Mock(return_value=executor)
        executor_context.__exit__ = Mock(return_value=False)
        clock = [0.0]
        events = iter([(60.0, None), (90.0, later), (120.0, None), (121.0, first)])
        completed = []
        timeouts = []

        def wait(futures, timeout, return_when):
            timeouts.append(timeout)
            clock[0], future = next(events)
            if future is later:
                future.set_result((2, synthetic_result(self.args.seed_base + 2)))
            if future is first:
                self.assertEqual([run for run, _ in completed], [2])
                future.set_result((0, synthetic_result(self.args.seed_base)))
            ready = {future} if future is not None else set()
            return ready, set(futures) - ready

        data = framework.Data().set_train_test(X_train=[[1]], y_train=[0], X_test=[[2]], y_test=[0])
        with patch.object(framework, "ProcessPoolExecutor", return_value=executor_context), \
                patch.object(framework, "wait", side_effect=wait), \
                patch.object(framework.time, "monotonic", side_effect=lambda: clock[0]), \
                redirect_stdout(self.output):
            result = framework.execute_pending_runs(
                data, "knn", "PSO", "vstf_01", self.args, [0, 2],
                on_run_complete=lambda run, out: completed.append((run, out)),
                dataset_name="Synthetic", completed_runs=4, started_at=0.0,
            )
        self.assertEqual(framework.PROGRESS_INTERVAL_SECONDS, 60.0)
        self.assertEqual(timeouts, [60.0, 60.0, 30.0, 60.0])
        lines = self.output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("PROGRESS | Synthetic | PSO_KNN | completed=4/6 | active=1 | pending=1 | elapsed=01:00 | status=running", lines[0])
        self.assertIn("completed=5/6 | active=1 | pending=0 | elapsed=02:00 | status=running", lines[1])
        self.assertEqual([run for run, _ in result], [0, 2])

    def test_atomic_writes_preserve_last_checkpoint_on_serialization_or_replace_failure(self):
        cache = self.root / "atomic.pkl"
        framework.save_cache(cache, {"saved": [1, 3]})
        before = cache.read_bytes()

        def broken_dump(payload, stream):
            stream.write(b"incomplete pickle")
            raise OSError("Simulated disk write failure")

        for target, effect in [("pickle.dump", broken_dump), ("os.replace", OSError("replace failed"))]:
            with self.subTest(target=target):
                module, attribute = target.split(".")
                with patch.object(getattr(framework, module), attribute, side_effect=effect), self.assertRaises(OSError):
                    framework.save_cache(cache, {"saved": [0, 1, 3]})
                self.assertEqual(cache.read_bytes(), before)
                self.assertEqual(framework.load_cache(cache), {"saved": [1, 3]})
                self.assertFalse(list(self.root.glob(".atomic.pkl.*.tmp")))
        real_replace = framework.os.replace

        def inspect_replace(source, destination):
            self.assertEqual(Path(source).parent, cache.parent)
            self.assertEqual(framework.load_cache(cache), {"saved": [1, 3]})
            self.assertEqual(framework.load_cache(source), {"saved": [0, 1, 3]})
            real_replace(source, destination)

        with patch.object(framework.os, "replace", side_effect=inspect_replace):
            framework.save_cache(cache, {"saved": [0, 1, 3]})
        self.assertEqual(framework.load_cache(cache), {"saved": [0, 1, 3]})


if __name__ == "__main__":
    unittest.main()
