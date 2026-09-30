"""Progress visibility and the optimized cache-only guard's protections."""
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from reporting.core import report_guard, report_stage, staged_version, next_report_version, safe_path


class FlushedOutput(StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


class ReportingProgressTests(unittest.TestCase):
    def test_stage_flushes_before_work_and_after_completion(self):
        output = FlushedOutput()
        with redirect_stdout(output):
            with report_stage('Actual work'):
                self.assertIn('Actual work ...', output.getvalue())
                self.assertEqual(output.flushes, 1)
        self.assertIn('Actual work complete', output.getvalue())
        self.assertEqual(output.flushes, 2)

    def test_failure_is_visible_and_propagates(self):
        output = FlushedOutput()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, 'original error'):
            with report_stage('Validation'):
                raise ValueError('original error')
        self.assertIn('Validation failed', output.getvalue())
        self.assertNotIn('complete', output.getvalue())
        self.assertEqual(output.flushes, 2)

    def test_cached_guard_keeps_name_and_library_checks(self):
        for name, filename in (('_run_single', 'framework.py'), ('build_optimizer', 'factory.py'),
                               ('load_dataset', 'framework.py'), ('solve', 'mealpy/optimizer.py'),
                               ('fit', 'mafese/selector.py'), ('fit', 'sklearn/base.py'),
                               ('run_single', 'main_best.py'), ('train_miafex', 'train_miafex.py'),
                               ('extract_miafex_features', 'extract_miafex_features.py'),
                               ('resolve_miafex_csv', 'main_best.py'), ('save_combination', 'main_best.py')):
            namespace = {'calls': []}
            exec(compile(f'def {name}():\n    calls.append("executed")', filename, 'exec'), namespace)
            with self.subTest(name=name), report_guard(()) as state:
                # Populate the allowed-code cache before visiting forbidden code.
                self.assertEqual(sum([1, 2]), 3)
                with self.assertRaisesRegex(RuntimeError, 'blocked scientific execution'):
                    namespace[name]()
            self.assertEqual(namespace['calls'], [])
            self.assertEqual(state['optimization_calls'], 1)

    def test_guard_restores_existing_profile(self):
        previous = sys.getprofile()
        with report_guard(()) as state:
            for _ in range(100):
                self.assertEqual(str(1), '1')
        self.assertIs(sys.getprofile(), previous)
        self.assertEqual(state['optimization_calls'], 0)


class ReportingProtectionTests(unittest.TestCase):
    def test_scientific_writes_and_subprocesses_are_blocked(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root / 'stage'; stage.mkdir()
            scientific = root / 'scientific.csv'; scientific.write_text('original')
            with report_guard((stage,)):
                with self.assertRaises(PermissionError): scientific.write_text('changed')
                with self.assertRaises(PermissionError): scientific.unlink()
                with self.assertRaises(PermissionError): os.replace(scientific, stage / 'moved.csv')
                for extension in ('.pkl', '.pth', '.ckpt', '.npy', '.pdf'):
                    with self.assertRaises(PermissionError): (stage / ('science' + extension)).write_bytes(b'bad')
                with self.assertRaises(RuntimeError): subprocess.run(['true'])
                (stage / 'report.txt').write_text('allowed')
            self.assertEqual(scientific.read_text(), 'original')

    def test_unpaired_malformed_redirected_and_conflicting_versions_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fig = root / 'Figures/EXP913/full_rep1'; fig.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Incomplete'): next_report_version(root, 913)
            res = root / 'Results/EXP913/full_rep1'; res.mkdir(parents=True)
            self.assertEqual(next_report_version(root, 913), 2)
            (res / 'validation.json').write_text('{"experiment_id": 914, "report_version": 1}')
            with self.assertRaisesRegex(ValueError, 'manifest identity'): next_report_version(root, 913)
            (res / 'validation.json').unlink()
            redirected = fig / 'redirect'; redirected.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Symlink'): next_report_version(root, 913)
            redirected.unlink()
            (fig.parent / 'full_rep01').mkdir()
            with self.assertRaisesRegex(ValueError, 'Noncanonical'): next_report_version(root, 913)
            with self.assertRaises(ValueError): safe_path(root / '..' / 'escape')

    def test_existing_destinations_and_failed_pair_publication_are_preserved(self):
        import reporting.core as core
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # A competing final destination is never replaced.
            with self.assertRaises(OSError):
                with staged_version(root, 913) as (_, fig, res, finals):
                    (fig / 'new.txt').write_text('new')
                    finals[0].mkdir(); (finals[0] / 'old.txt').write_text('old')
            final = root / 'Figures/EXP913/full_rep1'
            self.assertEqual((final / 'old.txt').read_text(), 'old')
            self.assertFalse((final / 'new.txt').exists())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rename = core._rename_new
            calls = []
            def fail_second(source, destination):
                calls.append(destination)
                if len(calls) == 2: raise OSError('second tree publication failed')
                return rename(source, destination)
            with patch.object(core, '_rename_new', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'second tree'):
                    with staged_version(root, 913) as (_, fig, res, finals):
                        (fig / 'new.txt').write_text('new')
                        (res / 'new.txt').write_text('new')
            self.assertEqual(next_report_version(root, 913), 1)
            self.assertFalse(list(root.rglob('.report-staging-*')))
            self.assertFalse(list(root.rglob('.report-allocation.lock')))


if __name__ == '__main__':
    unittest.main()
