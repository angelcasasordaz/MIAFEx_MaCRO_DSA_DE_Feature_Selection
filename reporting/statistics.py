"""Matched-block Friedman and paired Wilcoxon/Holm analysis of cached runs."""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from reporting.core import framework


def analyze(matrix, algorithms, *, higher_is_better=True):
    """Use the same complete blocks for all tests; never impute observations.

    Exact signed ranks require nonzero, untied differences (at most 50 pairs).
    Zeros/ties use the explicitly reported normal approximation with Wilcox zero
    removal. All-zero differences have W=0, p=1, matching the framework policy.
    Fewer than two complete blocks produce descriptive ranks but no inference.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(algorithms) or not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError('Expected a blocks-by-unique-algorithms matrix')
    complete = np.isfinite(x).all(axis=1)
    omitted = np.flatnonzero(~complete).tolist()
    x = x[complete]
    n, k = x.shape
    ranks = rankdata(-x if higher_is_better else x, axis=1, method='average')
    mean_ranks = ranks.mean(axis=0) if n else np.full(k, np.nan)
    friedman = {'statistic': None, 'pvalue': None, 'blocks': n, 'algorithms': k}
    if n < 2 or k < 3:
        friedman['status'] = 'insufficient observations: need at least 2 complete blocks and 3 algorithms'
    elif np.all(x == x[:, :1]):
        friedman['status'] = 'undefined: all algorithms tied in every block (zero tie correction)'
    else:
        test = friedmanchisquare(*x.T)
        friedman.update(statistic=float(test.statistic), pvalue=float(test.pvalue), status='computed; chi-square approximation')
    rows = []
    for i, j in combinations(range(k), 2):
        d = x[:, i] - x[:, j]
        zeros = int(np.count_nonzero(d == 0))
        nonzero = d[d != 0]
        ties = len(nonzero) != len(np.unique(np.abs(nonzero)))
        if n < 2:
            w, p, method = np.nan, np.nan, 'not tested: fewer than 2 complete blocks'
        elif not len(nonzero):
            w, p, method = 0., 1., 'all differences zero: degenerate W=0, p=1'
        else:
            method = 'exact' if not zeros and not ties and n <= 50 else 'approx'
            test = wilcoxon(d, alternative='two-sided', zero_method='wilcox', method=method, correction=False)
            w, p = float(test.statistic), float(test.pvalue)
        rows.append([algorithms[i], algorithms[j], w, p, method, n, zeros, ties])
    pairs = pd.DataFrame(rows, columns=['Algorithm_A', 'Algorithm_B', 'Wilcoxon_statistic', 'Raw_p',
                                       'Method', 'Matched_blocks', 'Zero_differences', 'Tied_absolute_differences'])
    raw = pairs.Raw_p.to_numpy(dtype=float)
    valid = np.isfinite(raw)
    if valid.any() and not valid.all():
        raise ValueError('Cannot apply Holm to an incomplete intended all-pairs family')
    adjusted = np.full(len(raw), np.nan)
    if valid.any():
        adjusted[valid] = framework()._holm_adjusted_pvalues(raw[valid])
    pairs['Holm_adjusted_p'] = adjusted
    pairs['Significant_0.05'] = [bool(p < .05) if np.isfinite(p) else None for p in adjusted]
    pvalues = np.eye(k)
    pvalues[:] = np.nan
    np.fill_diagonal(pvalues, 1.)
    for row in pairs.itertuples():
        i, j = algorithms.index(row.Algorithm_A), algorithms.index(row.Algorithm_B)
        pvalues[i, j] = pvalues[j, i] = row.Holm_adjusted_p
    summary = pd.DataFrame({'Algorithm': algorithms,
                            'Mean': x.mean(axis=0) if n else np.full(k, np.nan),
                            'Std': x.std(axis=0, ddof=1) if n > 1 else np.full(k, np.nan),
                            'Median': np.median(x, axis=0) if n else np.full(k, np.nan),
                            'Mean_Rank': mean_ranks})
    ranked = np.argsort(mean_ranks, kind='stable')
    return dict(x=x, ranks=ranks, mean_ranks=mean_ranks, ranked=ranked, pairs=pairs, matrix=pvalues,
                summary=summary.iloc[ranked].reset_index(drop=True), friedman=friedman,
                omitted_blocks=omitted, complete_mask=complete, family_size=k*(k-1)//2)


def matched_block_matrix(indexed, datasets, classifiers, algorithms, metric, *, expected_runs=None):
    """Run mean within a completed cell, then identical dataset/classifier blocks.

    Missing, nonfinite or explicitly incomplete cells stay NaN; never flatten
    raw runs across datasets/classifiers or change the intended algorithm family.
    """
    blocks = list(product(datasets, classifiers))
    x = np.full((len(blocks), len(algorithms)), np.nan)
    for i, (dataset, classifier) in enumerate(blocks):
        for j, algorithm in enumerate(algorithms):
            row = indexed.get((dataset, classifier, algorithm), {})
            values = np.asarray(row.get(metric.run_key, []), dtype=float)
            completed = row.get('CompletedRuns', len(values)) if values.ndim else 0
            ids = row.get('CompletedRunIDs', list(range(values.size)))
            valid_ids = (isinstance(ids, (list, tuple, np.ndarray))
                         and list(ids) == list(range(values.size)))
            if (values.ndim == 1 and values.size and np.isfinite(values).all()
                    and completed == values.size and valid_ids
                    and (expected_runs is None or values.size == expected_runs)):
                x[i, j] = values.mean() / metric.scale
    return blocks, x


def selected_metric(report):
    from reporting.figures import metric_token
    token = getattr(report.args, 'statistical_metric', framework().STATISTICAL_METRIC)
    if token not in ('accuracy', 'precision', 'recall', 'f1'):
        raise ValueError(f'Unsupported STATISTICAL_METRIC: {token}')
    selected = next((metric for metric in report.metrics if metric_token(metric) == token), None)
    if selected is None:
        raise ValueError(f'Statistical metric {token!r} has no completed cached observations')
    return selected


def export(report, figures, results):
    from reporting.figures import statistical_figures, save_png, reference_method, palette_name, figure_language
    import matplotlib.pyplot as plt
    from full_plot_style import STYLE
    figures, results = Path(figures), Path(results)
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    metric = selected_metric(report)
    algorithms = report.algorithms
    blocks, matrix = matched_block_matrix(report.indexed, report.datasets, report.classifiers, algorithms, metric,
                                          expected_runs=getattr(report.args, 'runs', None))
    analysis = analyze(matrix, algorithms, higher_is_better=metric.best_mode == 'max')
    summary, pairs = analysis['summary'].copy(), analysis['pairs']
    f = analysis['friedman']
    summary['Metric'] = metric.name
    summary['Complete_Blocks'] = len(analysis['x'])
    summary['Omitted_Blocks'] = len(analysis['omitted_blocks'])
    summary['Friedman_statistic'] = f['statistic']
    summary['Friedman_p'] = f['pvalue']
    summary['Friedman_significant_0.05'] = bool(f['pvalue'] < .05) if f['pvalue'] is not None else None
    summary['Alpha'] = .05
    summary['Pairwise_family_size'] = analysis['family_size']
    for name, frame in [('statistical_summary.csv', summary), ('pairwise_wilcoxon_holm.csv', pairs)]:
        frame.to_csv(results / name, index=False, float_format='%.17g')
        roundtrip_expected = frame.astype(object).where(frame.notna(), np.nan)
        pd.testing.assert_frame_equal(pd.read_csv(results / name), roundtrip_expected, check_dtype=False,
                                      check_exact=False, rtol=1e-14, atol=1e-15)
    block_frame = pd.DataFrame(matrix, columns=algorithms,
                               index=pd.MultiIndex.from_tuples(blocks, names=['Dataset', 'Classifier']))
    block_frame.to_csv(results / 'matched_block_means.csv', float_format='%.17g')
    omitted = [{'Dataset': blocks[i][0], 'Classifier': blocks[i][1],
                'Missing_or_incomplete_algorithms': ', '.join(a for a, value in zip(algorithms, matrix[i]) if not np.isfinite(value))}
               for i in analysis['omitted_blocks']]
    pd.DataFrame(omitted, columns=['Dataset', 'Classifier', 'Missing_or_incomplete_algorithms']).to_csv(
        results / 'omitted_blocks.csv', index=False)
    omitted_keys = [blocks[i] for i in analysis['omitted_blocks']]
    text = [f'{report.exp_tag} MIAFEx — matched-block {metric.name} analysis',
            f'Cache identity: {report.signature}; source caches: {list(report.sources)}',
            f'Hierarchy: arithmetic run mean within each dataset/classifier/algorithm; units: {metric.unit}.',
            f'Statistical metric: {metric.name}; reference method: {reference_method(report)}; alpha=0.05.',
            f'Complete blocks: {len(analysis["x"])}; algorithms: {len(algorithms)}; omitted blocks: {omitted_keys}.',
            f'Complete dataset/classifier keys: {[block for block, complete in zip(blocks, analysis["complete_mask"]) if complete]}.',
            'Mean, Median and sample Std (ddof=1) describe block means, not within-block run variability.',
            'Std is unavailable with fewer than two blocks. Ranks use average ties; rank 1 is best.',
            f'Friedman: {f}',
            'Paired Wilcoxon: two-sided; zero_method=wilcox; no continuity correction.',
            'Exact distribution for nonzero untied differences up to 50 blocks; otherwise normal approximation.',
            'Every pair records method, block count, zero differences and tied absolute differences.',
            f'Holm family: all {len(pairs)} unordered algorithm pairs; adjusted p < 0.05.',
            'All pairwise comparisons form a prespecified reporting family on the same complete blocks, including when the omnibus is nonsignificant.',
            'The displayed mean ranks, Friedman result, raw/adjusted pairwise p-values and significance are saved in the statistical CSVs.',
            'All-zero differences use degenerate W=0,p=1. Insufficient blocks produce no test result.',
            'Classifiers on one dataset share data. Wilcoxon inference assumes symmetric paired differences.',
            'Non-significance does not establish equivalence; ranks do not measure effect magnitude.',
            'Optimization calls: 0; all observations read from completed caches.']
    (results / 'statistical_report.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    skipped = []
    if len(analysis['x']):
        with plt.rc_context(STYLE):
            for stem, fig in statistical_figures(analysis, algorithms, metric.name,
                    reference=reference_method(report), palette_name=palette_name(report), language=figure_language(report)):
                save_png(fig, figures / f'{stem}.png')
    else:
        skipped.append({'output': 'Statistical PNGs', 'reason': 'No complete matched blocks'})
    if f['statistic'] is None:
        skipped.append({'output': 'Matched-block Friedman test', 'reason': f['status']})
    if not np.isfinite(pairs.Holm_adjusted_p.to_numpy(dtype=float)).any():
        skipped.append({'output': 'Reference-comparison PNG',
                        'reason': 'No estimable algorithm comparisons; need at least two algorithms and two complete blocks'})
    return skipped
