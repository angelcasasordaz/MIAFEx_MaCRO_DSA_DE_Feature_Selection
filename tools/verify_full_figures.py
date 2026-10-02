"""Cache-only FULL figure verification, including artist values and file hashes.

Run with the project's interpreter: tools/verify_full_figures.py --exp-id 606.
No optimizer, dataset loader or scientific exporter is called.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_best as m
from reporting import core, figures


def fingerprint(path):
    return {'sha256': core.sha256(path), 'mtime_ns': path.stat().st_mtime_ns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', type=int, default=606)
    options = parser.parse_args()
    args = m.parse_args(['--exp-id', str(options.exp_id), '--reuse-cache-from-exp-id', 'none'])
    root = Path(args.output_root).resolve()
    protected = {str(path.relative_to(root)): fingerprint(path)
                 for exp in (605, options.exp_id)
                 for path in (root / 'Results' / f'EXP{exp:03d}').rglob('*') if path.is_file()}
    report = core.load_completed_cache(args)
    identities = json.dumps(report.identities, sort_keys=True)
    science_names = ('build_combination_identity', 'build_optimizer', 'run_single', 'build_label_payload',
                     'export_global_excel', 'export_statistical_excel', 'export_friedman_analysis',
                     'build_friedman_fitness_matrix')
    science = {name: hashlib.sha256(inspect.getsource(getattr(m, name)).encode()).hexdigest()
               for name in science_names}
    plot = m.generate_plot_dataframe(report.results, report.args)
    expected = plot.copy()
    selected, expected_curves = [], {}
    for dataset, rows in report.results.items():
        for label, row in rows.items():
            index = int(np.argmin(row['FitRuns']))
            match = (expected.Archivo == dataset) & (expected.Configuracion == label)
            for column, key, scale in (
                ('AS_test', 'AccRuns', 100), ('PS_test', 'PSRuns', 1), ('RS_test', 'RSRuns', 1),
                ('F1_test', 'F1Runs', 1), ('N_Features_Selected', 'FeatRuns', 1), ('Runtime', 'TimeRuns', 1),
            ):
                expected.loc[match, column] = row[key][index] / scale
                np.testing.assert_array_equal(plot.loc[match, column], expected.loc[match, column])
            method = m.parse_result_label(label, report.args)['method']
            expected_curves[dataset, row['Estimator'], method] = row['CurvesAll'][index]
            selected.append(dict(dataset=dataset, classifier=row['Estimator'], optimizer=method,
                                 index=index, run_id=int(row.get('CompletedRunIDs', range(len(row['FitRuns'])))[index]),
                                 fitness=float(row['FitRuns'][index]), accuracy=float(row['AccRuns'][index] / 100),
                                 mean_accuracy=float(row['AccMean'] / 100), f1=float(row['F1Runs'][index]),
                                 precision=float(row['PSRuns'][index]), recall=float(row['RSRuns'][index]),
                                 features=float(row['FeatRuns'][index]), runtime=float(row['TimeRuns'][index])))
    for classifier in report.classifiers:
        for metric in report.metrics:
            actual = figures.metric_matrix(report, classifier, metric)
            wanted = np.asarray([[row[metric.run_key][int(np.argmin(row['FitRuns']))] / metric.scale
                                  for dataset in report.datasets
                                  for row in [report.indexed[dataset, classifier, algorithm]]]
                                 for algorithm in report.algorithms])
            np.testing.assert_array_equal(actual, wanted)
        summary = figures.summary_figure(report, classifier)
        for axis, metric in zip(summary.axes, report.metrics):
            np.testing.assert_allclose([bar.get_height() for bar in axis.patches],
                                       figures.metric_matrix(report, classifier, metric).mean(axis=1),
                                       rtol=0, atol=0)
        m.plt.close(summary)
    checked = []
    save = m._save_chart
    def verified_save(fig, out_dir, filename):
        if filename.startswith('06_heatmap_accuracy_'):
            classifier = filename.rsplit('_', 1)[1].removesuffix('.png')
            table, order, _, _ = m.prepare_plot_groups(expected[expected.Estimador == classifier], report.args.optimizers)
            matrix = table.groupby(['GrupoGrafica', 'Archivo']).AS_test.mean().unstack().reindex(
                index=order, columns=sorted(report.datasets)).to_numpy()
            np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), matrix)
            checked.append(filename)
        if filename == '09_resultados_clasificador_metrica_todos_datasets.png' or filename.startswith('01_resultados_clasificador_'):
            table = expected
            if filename.startswith('01_'):
                classifier = filename.rsplit('_', 1)[1].removesuffix('.png')
                dataset = filename[len('01_resultados_clasificador_'):].rsplit('_', 1)[0]
                table = expected[(expected.Archivo == dataset) & (expected.Estimador == classifier)]
            table, order, _, _ = m.prepare_plot_groups(table, report.args.optimizers)
            classifiers = [classifier for classifier in m.SUPPORTED_ESTIMATORS if classifier in set(table.Estimador)]
            grouped = table.groupby(['Estimador', 'GrupoGrafica'])[['AS_test', 'PS_test', 'RS_test', 'F1_test']].mean()
            for ci, classifier in enumerate(classifiers):
                for mi, column in enumerate(('AS_test', 'PS_test', 'RS_test', 'F1_test')):
                    np.testing.assert_allclose([bar.get_height() for bar in fig.axes[ci * 4 + mi].patches],
                                               [grouped.loc[classifier, algorithm][column] for algorithm in order],
                                               rtol=0, atol=0)
            checked.append(filename)
        if filename.startswith('05_convergence_'):
            classifier = (filename.rsplit('_', 1)[1].removesuffix('.png') if '_por_dataset_' not in filename else 'knn')
            for ax in fig.axes:
                for line in ax.lines:
                    np.testing.assert_array_equal(line.get_ydata(), expected_curves[ax.get_title(), classifier, line.get_label()])
            checked.append(filename)
        save(fig, out_dir, filename)
    with patch.object(m, 'PLOT_RUN_AGGREGATION', 'best'), patch.object(m, '_save_chart', side_effect=verified_save):
        result = core.run_full_figures(args)
    for name, state in protected.items():
        assert fingerprint(root / name) == state, name
    after = core.load_completed_cache(args)
    assert json.dumps(after.identities, sort_keys=True) == identities
    assert json.dumps(after.sources, sort_keys=True) == json.dumps(report.sources, sort_keys=True)
    baseline = Path('/tmp/miafex-exp606-before.json')
    if options.exp_id == 606 and baseline.is_file():
        audit = json.loads(baseline.read_text())
        assert audit['science'] == science
        assert audit['identities'] == report.identities
        for path, state in audit['protected'].items():
            assert fingerprint(root / path) == state, path
    result.update(aggregation='best', combinations_verified=len(selected), artist_checks=checked,
                  protected_files_verified=len(protected), scientific_sources=science,
                  scientific_tables_and_caches_unchanged=True, selected_runs=selected)
    output = root / 'diagnostics' / f'exp{options.exp_id}_plotting' / 'verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Verification: {output}; {len(selected)} aligned runs, {len(checked)} artist checks, '
          f'{len(protected)} protected files unchanged.', flush=True)


if __name__ == '__main__':
    main()
