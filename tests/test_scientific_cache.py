"""Per-combination compatibility and orchestration; no real optimizer execution."""
import argparse
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main_best as f
import scientific_cache as cache


class ScientificCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = f.parse_args([
            '--dataset-name', 'Tiny', '--feature-dataset-root', str(self.root / 'features'),
            '--output-root', str(self.root), '--pipeline-mode', 'feature_selection',
            '--optimizers', 'PSO', 'BRO', '--estimators', 'knn', '--runs', '4',
            '--epochs', '3', '--pop-size', '5', '--parallel', 'no',
            '--exp-id', '605', '--reuse-cache-from-exp-id', '604',
        ])
        self.args.optimizers = f.resolve_optimizers(self.args)
        # Feature-only discovery needs the prepared files before resolving Tiny.
        feature_dir = self.root / 'features' / 'Tiny'
        feature_dir.mkdir(parents=True)
        for split in ('train', 'test'):
            (feature_dir / f'{split}_features.csv').write_text('f0,f1,label\n1,2,0\n3,4,1\n')
        self.scoped = f.resolve_miafex_dataset_args(self.args)['Tiny']
        self.paths = f.make_paths(self.args)
        self.source = f.make_paths(self.args, exp_id=604)
        self.log = io.StringIO()
        self.enterContext(redirect_stdout(self.log))
        self.enterContext(patch.object(f, 'build_optimizer', side_effect=AssertionError('No optimizer')))
        self.enterContext(patch.object(f, 'train_miafex', side_effect=AssertionError('No training')))
        self.enterContext(patch.object(f, 'extract_miafex_features', side_effect=AssertionError('No extraction')))

    def identity(self, method='PSO', args=None, classifier='knn', transfer='vstf_01'):
        return f.build_combination_identity(args or self.scoped, 'Tiny', classifier, method, transfer)

    def test_exp604_cannot_enter_scientific_execution(self):
        args = copy.deepcopy(self.args); args.exp_id = 604
        with patch.object(f, 'parse_args', return_value=args), \
                patch.object(f, 'resolve_execution_config', side_effect=AssertionError('No backend setup')), \
                patch.object(f, 'make_paths', side_effect=AssertionError('No filesystem allocation')):
            with self.assertRaisesRegex(ValueError, 'EXP604 is read-only'):
                f.main()

    def test_common_contract_invalidates_all_and_bro_repair_is_method_scoped(self):
        methods = f.resolve_optimizers(f.parse_args([]))
        before = {method: self.identity(method) for method in methods}
        with patch.dict(f.BINARY_REPRESENTATION_REVISIONS, {'default': 'legacy-transfer-contract'}):
            for method in methods:
                self.assertNotEqual(before[method], self.identity(method), method)
        with patch.object(f, 'SEED_POLICY_REVISION', 'legacy-selector-default'):
            for method in methods:
                self.assertNotEqual(before[method], self.identity(method), method)
        with patch.dict(f.OPTIMIZER_REPAIR_REVISIONS, {'OriginalBRO': 'legacy-continuous-respawn'}):
            for method in methods:
                self.assertEqual(before[method] == self.identity(method), method != 'OriginalBRO', method)

    def row(self, ids=(0, 1, 2, 3)):
        finals = [.2 + run / 100 for run in ids]
        return f.build_label_payload('knn', *[finals for _ in range(7)],
                                     [[.9, .8, final] for final in finals], 3,
                                     completed_run_ids=list(ids))

    def write(self, method='PSO', ids=(0, 1, 2, 3), paths=None):
        identity = self.identity(method)
        f.save_combination(paths or self.source, identity, self.row(ids))
        return cache.combination_files(paths or self.source, identity)

    def resolve(self, *, figures=True, args=None):
        return f.resolve_cached_payload(self.paths, args or self.scoped, 'Tiny', 'knn', figures_only=figures) or {}

    def test_subset_and_superset_identity_and_individual_parameters(self):
        before = self.identity()
        changed = copy.deepcopy(self.scoped)
        changed.optimizers = ['BRO', 'PSO', 'MaCRO-DE-t', 'DE']
        changed.estimators = ['svm', 'knn']
        changed.transfer_functions = ['sstf_01', 'vstf_01']
        changed.dsade_beta_min = .01
        changed.dsade_mahal_q = .4
        changed.exp_id = 999
        changed.miafex_epochs = 999  # prepared content is the feature identity
        self.assertEqual(before, self.identity(args=changed))
        self.assertNotEqual(self.identity('MaCRO-DE-t'), self.identity('MaCRO-DE-t', changed))

    def test_scientific_changes_reject_only_affected_combination(self):
        before = self.identity()
        for key, value in dict(runs=5, epochs=4, pop_size=6, seed_base=9, random_state=9, test_size=.3).items():
            changed = copy.deepcopy(self.scoped)
            setattr(changed, key, value)
            self.assertNotEqual(before, self.identity(args=changed), key)
        self.assertNotEqual(before, self.identity(classifier='svm'))
        self.assertNotEqual(before, self.identity(transfer='sstf_01'))
        path = Path(self.scoped.train_features_csv)
        path.write_text(path.read_text().replace('1,2,0', '9,2,0'))
        self.assertNotEqual(before, self.identity())

    def test_feature_content_can_move_without_changing_identity(self):
        changed = copy.deepcopy(self.scoped)
        for field in ('train_features_csv', 'test_features_csv'):
            old = Path(getattr(changed, field))
            new = old.with_name('moved-' + old.name)
            new.write_bytes(old.read_bytes())
            setattr(changed, field, str(new))
        self.assertEqual(self.identity(), self.identity(args=changed))

    def test_subset_two_of_thirteen_and_source_read_only(self):
        self.scoped.optimizers = f.resolve_optimizers(f.parse_args([]))
        for method in self.scoped.optimizers:
            self.write(method)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(self.source.cache_dir).rglob('*.pkl')}
        self.scoped.optimizers = ['PSO', 'BRO']
        for figures in (True, False):
            self.assertEqual(len(self.resolve(figures=figures)), 2)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        self.assertEqual(len(list(Path(self.paths.cache_dir).rglob('*_results.pkl'))), 2)

    def test_ten_requested_seven_available_and_only_three_scheduled(self):
        methods = ['MaCRO-DE-t', 'DE', 'JADE', 'SHADE', 'PSO', 'GWO', 'WOA', 'HHO', 'BRO', 'DBO']
        self.args.optimizers = [f.resolve_optimizer_name(method) for method in methods]
        self.scoped.optimizers = self.args.optimizers
        for method in methods[:7]:
            self.write(method)
        self.assertEqual(len(self.resolve()), 7)
        run = self.invoke_main(figures=False)
        self.assertEqual(len(run.call_args_list), 12)
        self.assertEqual({call.args[2] for call in run.call_args_list}, set(self.args.optimizers[7:]))
        self.assertEqual(len(self.resolve()), 10)

    def invoke_main(self, *, figures):
        args = copy.deepcopy(self.args)
        args.figures_only = figures
        backend = f.ExecutionConfig('cpu', 'auto', 'sklearn', 'cpu', f.BackendAvailability(False, False, None, False, False))
        result = dict(as_test=80., ps_test=.8, rs_test=.8, f1_test=.8, fit_final=.2, n_features=1,
                      runtime=.01, curve=[.9, .8, .2], convergence=f.convergence_metadata(3, .2))
        with patch.object(f, 'parse_args', return_value=args), \
                patch.object(f, 'resolve_execution_config', return_value=backend), \
                patch.object(f, 'print_backend_report'), \
                patch.object(f, 'export_reporting_outputs', return_value=([], 'summary.csv', [])), \
                patch.object(f, 'run_single', return_value=result) as run:
            f.main()
        return run

    def test_noncontiguous_resume_merges_current_and_source_individually(self):
        self.args.optimizers = self.scoped.optimizers = ['OriginalPSO']
        self.write(ids=(1,), paths=self.paths)
        self.write(ids=(3,))
        run = self.invoke_main(figures=False)
        self.assertEqual([call.args[-1] for call in run.call_args_list], [1234, 1236])
        self.assertEqual(f.completed_run_ids(next(iter(self.resolve().values()))), [0, 1, 2, 3])
        self.invoke_main(figures=False).assert_not_called()

    def test_figures_only_partial_reports_all_available_and_missing(self):
        self.write(ids=(1, 3))
        self.invoke_main(figures=True).assert_not_called()
        self.assertIn('OriginalBRO', self.log.getvalue())
        self.assertIn('CACHE MISS', self.log.getvalue())
        self.assertIn('CACHE MISSING RUNS', self.log.getvalue())
        self.assertEqual(list(Path(self.paths.cache_dir).iterdir()), [])

    def test_binary_repair_implementation_and_seed_guards(self):
        self.scoped.optimizers = ['OriginalPSO', 'OriginalBRO', 'OriginalDE', 'MaCRO-DE-t']
        for method in self.scoped.optimizers:
            self.write(method)
        with patch.dict(f.BINARY_REPRESENTATION_REVISIONS, {'OriginalPSO': 'corrected-v2', 'OriginalBRO': 'corrected-v2'}):
            self.assertEqual(set(self.resolve()), {'ORIGINALDE', 'MACRO-DE-T'})
        with patch.dict(f.OPTIMIZER_REPAIR_REVISIONS, {'OriginalBRO': 'binary-respawn-v2'}):
            self.assertNotIn('ORIGINALBRO', self.resolve())
            self.assertIn('ORIGINALPSO', self.resolve())
        with patch.object(f.MaCRO_DE_t, 'IMPLEMENTATION_REVISION', 'changed'):
            self.assertNotIn('MACRO-DE-T', self.resolve())
            self.assertIn('ORIGINALPSO', self.resolve())
        with patch.object(f, 'SEED_POLICY_REVISION', 'explicit-selector-seed-v2'):
            self.assertEqual(self.resolve(), {})

    def test_unversioned_group_is_not_silently_blessed(self):
        legacy = f.cache_files(self.source, 'Tiny', 'knn', f.build_legacy_group_signature(self.scoped))[0]
        f.save_cache(legacy, {'ORIGINALPSO': self.row()})
        self.assertEqual(self.resolve(), {})

    def test_reference_hash_and_identity_tamper_rejected(self):
        identity = self.identity()
        original = Path(self.source.cache_dir) / 'original.pkl'
        f.save_cache(str(original), {'PSO': self.row()})
        ref = cache.reference_file(self.source, identity)
        entry = {'identity': identity, 'sources': [{'file': original.name, 'label': 'PSO', 'sha256': cache.file_digest(original)}]}
        cache.atomic_json(ref, entry)
        self.assertEqual(len(self.resolve()), 1)
        original.write_bytes(original.read_bytes() + b'changed')
        self.assertEqual(self.resolve(), {})
        self.write()
        final, progress = cache.combination_files(self.source, identity)
        for path in (final, progress):
            entry = f.load_cache(str(path)); entry['identity']['epochs'] += 1
            f.save_cache(str(path), entry)
        self.assertEqual(self.resolve(), {})

    def test_atomic_write_failure_keeps_existing_row(self):
        final, progress = self.write()
        before = final.read_bytes(), progress.read_bytes()
        with patch.object(f.pickle, 'dump', side_effect=RuntimeError('interrupted')), self.assertRaises(RuntimeError):
            self.write(ids=(1,))
        self.assertEqual(before, (final.read_bytes(), progress.read_bytes()))
        self.assertFalse(list(final.parent.glob('*.tmp')))

    def test_no_reuse_resumes_only_current_progress(self):
        self.write()
        self.scoped.reuse_cache = False
        self.assertEqual(self.resolve(figures=False), {})
        self.write(ids=(1, 3), paths=self.paths)
        self.assertEqual(f.completed_run_ids(next(iter(self.resolve(figures=False).values()))), [1, 3])

    def test_figures_only_convergence_validation_is_strict(self):
        row = self.row(); row['CurvesAll'][1] = [.9, .1, .21]
        f.save_combination(self.source, self.identity(), row)
        with self.assertRaisesRegex(ValueError, 'FIGURES_ONLY'):
            self.resolve()

    def test_source_recovers_invalid_current_runs_without_execution(self):
        row = self.row(); row['CurvesAll'][1] = [.9, .1, .21]
        f.save_combination(self.paths, self.identity(), row)
        self.write()
        recovered = self.resolve()
        self.assertEqual(f.cached_convergence_errors(recovered['ORIGINALPSO'], 3), ({}, []))

    def test_row_cannot_override_classifier_or_seed_schedule(self):
        for change in ({'Estimator': 'svm'}, {'CompletedRunIDs': [0, 1, 2, 8]}):
            row = self.row(); row.update(change)
            f.save_combination(self.source, self.identity(), row)
            self.assertEqual(self.resolve(), {})


if __name__ == '__main__':
    unittest.main()
