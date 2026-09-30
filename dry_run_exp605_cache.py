"""Read-only EXP605 cache audit. No optimizer, neural stage, or EXP writes.

Run from the repository root. Only diagnostics/exp605_integration is written.
"""
from contextlib import ExitStack, redirect_stdout
import argparse
import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import main_best as f
import scientific_cache as cache


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostics/exp605_integration'


def snapshot():
    files, directories = {}, {}
    for exp in (604, 605):
        for folder in ('Results', 'Figures'):
            root = ROOT / folder / f'EXP{exp}'
            for path in [root, *root.rglob('*')]:
                relative = str(path.relative_to(ROOT))
                if path.is_file():
                    files[relative] = {'sha256': cache.file_digest(path),
                                       'mtime_ns': path.stat().st_mtime_ns, 'size': path.stat().st_size}
                elif path.is_dir():
                    directories[relative] = path.stat().st_mtime_ns
    return {'files': files, 'directories': directories}


def audit(output_dir=OUT):
    args = f.parse_args([])
    assert (args.exp_id, args.reuse_cache_from_exp_id, args.reuse_cache, args.figures_only) == (605, 604, True, False)
    assert not args.report_only
    assert (args.pipeline_mode, args.train_miafex, args.extract_miafex) == ('feature_selection', 'no', 'no')
    assert (args.runs, args.epochs, args.pop_size) == (20, 200, 30)
    assert args.estimators == ['knn', 'svm'] and len(args.optimizers) == 13
    args.optimizers = f.resolve_optimizers(args)
    before = snapshot()
    manifest = json.loads((ROOT / 'diagnostics/exp604_cache_migration/manifest.json').read_text())
    records = {(r['identity']['dataset'], r['identity']['classifier'], r['identity']['optimizer'],
                r['identity']['transfer_function']): r for r in manifest['records']}
    paths = f.make_paths(args, create=False)
    source = f.make_paths(args, exp_id=604, create=False)
    log = io.StringIO()
    combinations = []
    with ExitStack() as stack:
        for name in ('run_single', 'execute_pending_runs', 'build_optimizer', 'train_miafex',
                     'extract_miafex_features', 'resolve_miafex_csv', 'save_cache', 'save_combination',
                     'load_miafex_feature_data', 'resolve_execution_config'):
            original = getattr(f, name)
            blocked = stack.enter_context(patch.object(f, name, side_effect=AssertionError(f'Forbidden execution: {name}')))
            if name == 'load_miafex_feature_data':
                blocked.__wrapped__ = original  # Hash real source; the loader still cannot execute.
        stack.enter_context(patch.object(cache, 'atomic_json', side_effect=AssertionError('Forbidden cache write')))
        stack.enter_context(redirect_stdout(log))
        for dataset, scoped in f.resolve_miafex_dataset_args(args).items():
            for classifier in args.estimators:
                # figures_only here suppresses cache imports; the requested configuration remains False.
                payload = f.resolve_cached_payload(paths, scoped, dataset, classifier, figures_only=True) or {}
                for method in args.optimizers:
                    for transfer in args.transfer_functions:
                        identity = f.build_combination_identity(scoped, dataset, classifier, method, transfer)
                        record = records.get((dataset, classifier, method, transfer))
                        old = record['identity'] if record else None
                        differences = sorted(k for k in set(identity) | set(old or {})
                                             if identity.get(k) != (old or {}).get(k))
                        if record:
                            # Prove the historical identity and payload still exist, without relabeling them.
                            assert json.loads(cache.reference_file(source, old).read_text()) == record
                            legacy = list(cache.read_candidates(source, old))
                            assert legacy and len(f.completed_run_ids(legacy[0])) == args.runs
                            assert not any(f.cached_convergence_errors(legacy[0], args.epochs))
                        label = f.build_alg_label(method, transfer, classifier, len(args.transfer_functions) > 1, True)
                        row = payload.get(label)
                        reused = f.completed_run_ids(row) if row else []
                        source_rows = list(cache.read_candidates(source, identity))
                        source_reused = sorted({run for item in source_rows for run in f.completed_run_ids(item)})
                        if old is not None and not differences:
                            assert set(source_reused) == set(range(args.runs)), 'Compatible source must be reusable'
                        if differences:
                            assert not source_reused, 'Incompatible legacy results must be rejected'
                        combinations.append({'dataset': dataset, 'classifier': classifier,
                                             'optimizer': method, 'transfer': transfer,
                                             'legacy_available': bool(record), 'changed_identity_fields': differences,
                                             'legacy_identity_digest': cache.identity_digest(old) if old else None,
                                             'requested_identity_digest': cache.identity_digest(identity),
                                             'scientifically_compatible_with_exp604': old is not None and not differences,
                                             'reusable_source_run_ids': source_reused,
                                             'reusable_run_ids': reused,
                                             'pending_run_ids': sorted(set(range(args.runs)) - set(reused))})
    assert snapshot() == before, 'EXP604/EXP605 content, membership or modification times changed'
    exp604 = {kind: {p: value for p, value in entries.items() if '/EXP604' in p}
              for kind, entries in before.items()}
    report = {'configuration': {key: getattr(args, key) for key in
              ('exp_id', 'reuse_cache_from_exp_id', 'reuse_cache', 'figures_only', 'pipeline_mode',
               'train_miafex', 'extract_miafex', 'optimizers', 'estimators', 'runs', 'epochs', 'pop_size')},
              'requested_combinations': len(combinations),
              'reusable_source_combinations': sum(bool(c['reusable_source_run_ids']) for c in combinations),
              'reusable_combinations': sum(bool(c['reusable_run_ids']) for c in combinations),
              'pending_combinations': sum(bool(c['pending_run_ids']) for c in combinations),
              'pending_runs': sum(len(c['pending_run_ids']) for c in combinations),
              'rerun_optimizers': sorted({c['optimizer'] for c in combinations if c['pending_run_ids']}),
              'exp604_files_unchanged': len(exp604['files']),
              'exp604_and_exp605_unchanged': True, 'optimizer_executions': 0, 'neural_executions': 0,
              'combinations': combinations}
    output_dir = Path(output_dir).resolve()
    if (not output_dir.is_relative_to(ROOT / 'diagnostics') or output_dir == ROOT / 'diagnostics'
            or output_dir.is_relative_to(ROOT / 'diagnostics/corrected_binary_final')
            or output_dir.is_relative_to(ROOT / 'diagnostics/exp604_cache_migration')):
        raise ValueError('Dry-run output must be a separate directory under diagnostics/')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'cache_dry_run.json').write_text(json.dumps(report, indent=2) + '\n')
    (output_dir / 'cache_dry_run.log').write_text(log.getvalue())
    (output_dir / 'exp604_preservation.json').write_text(json.dumps(exp604, indent=2) + '\n')
    with (output_dir / 'combination_decisions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'classifier', 'optimizer', 'transfer',
                                                    'scientifically_compatible_with_exp604', 'pending_runs',
                                                    'legacy_identity_digest', 'requested_identity_digest'])
        writer.writeheader()
        for combination in combinations:
            writer.writerow({key: len(combination['pending_run_ids']) if key == 'pending_runs' else combination[key]
                             for key in writer.fieldnames})
    print(json.dumps({k: v for k, v in report.items() if k != 'combinations'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-output', type=Path, default=OUT)
    audit(parser.parse_args().diagnostic_output)
