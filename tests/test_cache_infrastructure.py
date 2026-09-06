"""Tiny same/cross-EXP orchestration; no neural training or optimizer runs."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

import main_best as framework


class WorkerSizingTests(unittest.TestCase):
    def test_cpu_ram_affinity_and_unknown_limits(self):
        mib = 1024**2
        for cpus, affinity, ram, expected in (
            (12, 12, 8 * 1024 * mib, 8), (12, 12, 512 * mib + 3 * 192 * mib, 3),
            (12, 3, 8 * 1024 * mib, 2), (1, 1, 0, 1), (None, 12, None, 1),
            (12, 12, None, 8), (12, 12, 511 * mib, 1),
        ):
            with self.subTest(cpus=cpus, ram=ram), \
                    patch.object(framework.os, 'cpu_count', return_value=cpus), \
                    patch.object(framework.os, 'sched_getaffinity', return_value=set(range(affinity))), \
                    patch.object(framework, 'available_memory_bytes', return_value=ram):
                self.assertEqual(framework.automatic_worker_count(), expected)
        self.assertEqual(framework.AUTO_WORKER_CPU_FRACTION, 2 / 3)
        self.assertEqual(framework.AUTO_WORKER_RAM_BYTES, 192 * mib)
        self.assertEqual(framework.AUTO_RAM_RESERVE_BYTES, 512 * mib)

    def test_memory_uses_available_ram_and_portable_fallbacks(self):
        with patch('builtins.open', mock_open(read_data='MemTotal: 9999 kB\nMemAvailable: 1234 kB\n')):
            self.assertEqual(framework.available_memory_bytes(), 1234 * 1024)
        psutil = Mock()
        psutil.virtual_memory.return_value.available = 4567
        with patch('builtins.open', side_effect=OSError), \
                patch.object(framework.importlib, 'import_module', return_value=psutil):
            self.assertEqual(framework.available_memory_bytes(), 4567)
        with patch('builtins.open', side_effect=OSError), \
                patch.object(framework.importlib, 'import_module', side_effect=ImportError), \
                patch.object(framework.os, 'sysconf', side_effect=[3, 4096]):
            self.assertEqual(framework.available_memory_bytes(), 3 * 4096)
        with patch('builtins.open', side_effect=OSError), \
                patch.object(framework.importlib, 'import_module', side_effect=ImportError), \
                patch.object(framework.os, 'sysconf', side_effect=ValueError):
            self.assertIsNone(framework.available_memory_bytes())


class CacheInfrastructureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cache-infrastructure-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = framework.parse_args([
            '--exp-id', '602', '--reuse-cache-from-exp-id', '601',
            '--output-root', str(self.root), '--pipeline-mode', 'feature_selection',
            '--dataset-name', 'Tiny', '--dataset-root', str(self.root / 'images'),
            '--feature-dataset-root', str(self.root / 'features'),
            '--optimizers', 'PSO', '--estimators', 'knn', '--runs', '4',
            '--fs-epochs', '3', '--pop-size', '5', '--parallel', 'no',
        ])
        self.args.optimizers = framework.resolve_optimizers(self.args)
        self.scoped = framework.resolve_miafex_dataset_args(self.args)['Tiny']
        for split, path in framework.miafex_feature_paths(self.scoped).items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('f0,f1,label\n1,2,0\n3,4,1\n' if split == 'train'
                                  else 'f0,f1,label\n10,20,0\n30,40,1\n')
        self.paths = framework.make_paths(self.args)
        self.source = framework.make_paths(self.args, exp_id=601)
        self.signature = framework.build_cache_signature(self.scoped)
        self.label = framework.build_alg_label(self.args.optimizers[0], 'vstf_01', 'knn', False, False)
        self.output = io.StringIO()

    def payload(self, ids, *, legacy=False):
        return {self.label: framework.build_label_payload(
            'knn', *[[0.5 + run / 100 for run in ids] for _ in range(7)],
            [[0.3, 0.2, 0.1] for _ in ids], 3,
            completed_run_ids=None if legacy else ids,
        )}

    def write(self, paths, ids, *, progress=False, signature=None, legacy=False):
        filename = framework.cache_files(paths, 'Tiny', 'knn', signature or self.signature)[int(progress)]
        framework.save_cache(filename, self.payload(ids, legacy=legacy))
        return Path(filename)

    def resolve(self, **kwargs):
        with redirect_stdout(self.output):
            return framework.resolve_cached_payload(
                self.paths, self.scoped, 'Tiny', 'knn', self.signature, **kwargs)

    def ids(self, payload):
        return framework.completed_run_ids(payload[self.label])

    def test_defaults_and_none_cli(self):
        args = framework.parse_args([])
        self.assertEqual((framework.EXP_ID, framework.REUSE_CACHE_FROM_EXP_ID), (602, 602))
        self.assertEqual((args.exp_id, args.reuse_cache_from_exp_id), (602, 602))
        self.assertEqual(args.n_workers, framework.N_WORKERS)
        self.assertEqual(args.figures_only, framework.FIGURES_ONLY)
        self.assertIsNone(framework.parse_args(['--reuse-cache-from-exp-id', 'none']).reuse_cache_from_exp_id)

    def test_signature_excludes_execution_settings_but_retains_science(self):
        for key, value in dict(exp_id=999, reuse_cache_from_exp_id=998, output_root='elsewhere',
                               n_workers=1, parallel='yes', reuse_cache=False, figures_only=True,
                               pipeline_mode='full', train_miafex='yes', extract_miafex='no',
                               miafex_output='other/checkpoint', dataset_root='offline',
                               features_csv='legacy-alias', compute_mode='cpu', ml_backend='sklearn',
                               miafex_device='cpu').items():
            changed = copy.deepcopy(self.scoped)
            setattr(changed, key, value)
            with self.subTest(key=key):
                self.assertEqual(framework.build_cache_signature(changed), self.signature)
        for key, value in dict(epochs=4, pop_size=6, runs=5, seed_base=999, test_size=0.3,
                               random_state=99, train_features_csv='different.csv',
                               test_features_csv='different-test.csv', transfer_functions=['sstf_01'],
                               dsade_beta_min=0.1, miafex_epochs=11).items():
            changed = copy.deepcopy(self.scoped)
            setattr(changed, key, value)
            with self.subTest(key=key):
                self.assertNotEqual(framework.build_cache_signature(changed), self.signature)

    def test_current_wins_and_richer_progress_wins_over_final(self):
        self.write(self.paths, [1])
        self.write(self.paths, [1, 3], progress=True)
        self.write(self.source, [0, 1, 2, 3])
        with patch.object(framework, 'make_paths', side_effect=AssertionError('Source must not be consulted')):
            self.assertEqual(self.ids(self.resolve()), [1, 3])
        self.assertIn('CACHE HIT CURRENT', self.output.getvalue())

    def test_cross_import_is_atomic_and_source_stays_byte_and_mtime_identical(self):
        source = self.write(self.source, [1, 3], progress=True)
        before = source.read_bytes(), source.stat().st_mtime_ns
        self.assertEqual(self.ids(self.resolve()), [1, 3])
        for path in framework.cache_files(self.paths, 'Tiny', 'knn', self.signature):
            self.assertEqual(self.ids(framework.load_cache(path)), [1, 3])
        self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns))
        self.assertFalse(list(Path(self.paths.cache_dir).glob('*.tmp')))
        self.assertIn('CACHE IMPORTED | EXP601 -> EXP602', self.output.getvalue())
        self.resolve()
        self.assertIn('CACHE HIT CURRENT', self.output.getvalue())

    def test_same_exp_and_none_never_consult_source_or_create_directories(self):
        self.write(self.source, [1])
        for source_id in (602, None, 987):
            self.scoped.reuse_cache_from_exp_id = source_id
            self.assertIsNone(self.resolve())
        self.assertFalse((self.root / 'Results/EXP987').exists())
        self.assertFalse((self.root / 'Figures/EXP987').exists())
        self.assertEqual(self.output.getvalue().count('CACHE MISS'), 3)

    def test_incompatible_science_is_not_imported(self):
        changed = copy.deepcopy(self.scoped)
        changed.epochs += 1
        self.write(self.source, [1], signature=framework.build_cache_signature(changed))
        self.assertIsNone(self.resolve())
        self.assertIn('CACHE SOURCE INCOMPATIBLE', self.output.getvalue())
        self.assertIn('CACHE MISS', self.output.getvalue())
        self.assertEqual(list(Path(self.paths.cache_dir).iterdir()), [])

    def test_legacy_hash_and_contiguous_rows_migrate_same_or_cross_exp(self):
        for paths in (self.source, self.paths):
            with self.subTest(exp=paths.exp_tag):
                legacy_sig = framework._hash_cache_settings(framework._legacy_cache_settings(self.scoped))
                legacy = self.write(paths, [0, 1], signature=legacy_sig, legacy=True, progress=True)
                before = legacy.read_bytes()
                self.assertEqual(self.ids(self.resolve()), [0, 1])
                self.assertEqual(legacy.read_bytes(), before)
                for path in framework.cache_files(self.paths, 'Tiny', 'knn', self.signature):
                    Path(path).unlink()
        self.assertIn('migrated legacy signature', self.output.getvalue())

    def test_obsolete_single_csv_partition_signature_is_rejected(self):
        payload = framework._legacy_cache_settings(self.scoped)
        for key in ('miafex_partition_mode', 'train_features_csv', 'test_features_csv'):
            payload.pop(key)
        self.write(self.source, [0], signature=framework._hash_cache_settings(payload))
        self.assertIsNone(self.resolve())

    def test_no_reuse_keeps_current_progress_resume_but_disables_source(self):
        self.scoped.reuse_cache = False
        self.write(self.paths, [0, 1, 2, 3])
        self.write(self.source, [0, 1, 2, 3])
        self.assertIsNone(self.resolve())
        self.write(self.paths, [1], progress=True)
        self.assertEqual(self.ids(self.resolve()), [1])

    def test_corrupt_current_can_fall_back_to_compatible_source(self):
        path = self.write(self.paths, [1])
        path.write_bytes(b'broken')
        self.write(self.source, [1, 3])
        self.assertEqual(self.ids(self.resolve()), [1, 3])
        self.assertIn('cache-warning', self.output.getvalue())

    def invoke_main(self, *, figures=False):
        args = copy.deepcopy(self.args)
        args.figures_only = figures
        backend = framework.ExecutionConfig('cpu', 'auto', 'sklearn', 'cpu',
                    framework.BackendAvailability(False, False, None, False, False))
        result = dict(as_test=80, ps_test=0.8, rs_test=0.8, f1_test=0.8, fit_final=0.2,
                      n_features=1, runtime=0.001, curve=[0.4, 0.3, 0.2])
        with patch.object(framework, 'parse_args', return_value=args), \
                patch.object(framework, 'resolve_execution_config', return_value=backend), \
                patch.object(framework, 'print_backend_report'), \
                patch.object(framework, 'train_miafex', side_effect=AssertionError('No training')), \
                patch.object(framework, 'extract_miafex_features', side_effect=AssertionError('No extraction')), \
                patch.object(framework, 'run_single', return_value=result) as run, \
                patch.object(framework, 'export_global_excel', return_value=[]), \
                patch.object(framework, 'generate_seven_global_charts', return_value=[]), \
                redirect_stdout(self.output):
            framework.main()
        return run

    def test_cross_exp_noncontiguous_resume_executes_only_missing_ids(self):
        source = self.write(self.source, [1, 3], progress=True)
        before = source.read_bytes(), source.stat().st_mtime_ns
        run = self.invoke_main()
        self.assertEqual([call.args[-1] for call in run.call_args_list], [self.args.seed_base, self.args.seed_base + 2])
        self.assertEqual(self.ids(self.resolve()), [0, 1, 2, 3])
        self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns))
        self.invoke_main().assert_not_called()

    def test_figures_only_imports_without_loading_or_regenerating_features(self):
        source = self.write(self.source, [1, 3], progress=True)
        before = source.read_bytes(), source.stat().st_mtime_ns
        with patch.object(framework, 'load_miafex_feature_data', side_effect=AssertionError('No CSV loading')), \
                patch.object(framework, 'resolve_miafex_csv', side_effect=AssertionError('No artifact preparation')):
            self.invoke_main(figures=True).assert_not_called()
        self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns))
        self.assertTrue((self.root / 'Results/EXP602/RESUMEN_GRAFICAS_EXP602.csv').exists())
        self.assertIn('CACHE IMPORTED', self.output.getvalue())


if __name__ == '__main__':
    unittest.main()
