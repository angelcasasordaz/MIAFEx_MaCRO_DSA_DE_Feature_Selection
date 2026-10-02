"""Publication figures derived from actual cached datasets, classifiers and algorithms."""
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import t

from reporting.core import report_stage, framework


STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11,
         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
         'figure.dpi': 100, 'savefig.dpi': 600}
# A reusable ordered palette, independent of experiment IDs and algorithm names.
COLORS = ('#6E0D1B', '#009E73', '#E69F00', '#CC79A7', '#56B4E9', '#D55E00',
          '#6A3D9A', '#8DAA00', '#F0C808', '#A65628', '#4D4D4D', '#999999')


def palette(algorithms):
    extra = max(0, len(algorithms) - len(COLORS))
    colors = list(COLORS) + [plt.get_cmap('hsv')((i + .5) / extra) for i in range(extra)]
    return dict(zip(algorithms, colors))


def style_axes(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color='#D9D9D9', linestyle='--', linewidth=.65)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#777777')


def metric_values(df, metric, classifier, datasets, algorithms):
    """Preserve observed dataset order; absent/duplicate cells fail explicitly."""
    sub = df[df.Estimator == classifier]
    if sub.duplicated(['Dataset', 'Optimizer']).any():
        raise ValueError(f'Ambiguous metric observations for {classifier}')
    return np.stack([sub[sub.Optimizer == opt].set_index('Dataset').loc[list(datasets), metric].to_numpy(float)
                     for opt in algorithms])


def save_png(fig, path):
    try:
        target = Path(path).with_suffix('.png')
        with report_stage(f'Save {target.name} (600 dpi PNG)'):
            fig.savefig(target, format='png', dpi=600, bbox_inches='tight', facecolor='white')
        if not target.is_file():
            raise ValueError(f'Missing generated figure: {target}')
    finally:
        plt.close(fig)


def metric_matrix(report, classifier, metric):
    """Figure-only representative observations; selection is shared across metrics."""
    def value(row):
        index = framework()._plot_run_index(row)
        values = np.asarray(row[metric.run_key], dtype=float)
        if values.ndim != 1 or (index is not None and len(values) != len(row['FitRuns'])):
            raise ValueError(f'{metric.run_key} must align with FitRuns')
        return (np.mean(values) if index is None else values[index]) / metric.scale
    return np.asarray([[value(report.indexed[ds, classifier, opt])
                        for ds in report.datasets] for opt in report.algorithms])


def run_observations(report, classifier, metric):
    """Every real run across datasets, regardless of representative aggregation."""
    return np.asarray([np.concatenate([
        np.asarray(report.indexed[ds, classifier, opt][metric.run_key], dtype=float) / metric.scale
        for ds in report.datasets]) for opt in report.algorithms])


def dataset_mean_ci(values):
    """Student-t 95% intervals across dataset representatives, never flattened runs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Expected finite algorithm-by-dataset observations')
    n = values.shape[1]
    return values.mean(axis=1), (t.ppf(.975, n - 1) * values.std(axis=1, ddof=1) / np.sqrt(n)
                                 if n > 1 else None)


def panel_grid(count, *, width=5, height=4, polar=False):
    if count < 1:
        raise ValueError('A figure needs at least one observed panel')
    columns = min(3, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                             squeeze=False, layout='constrained',
                             subplot_kw={'polar': True} if polar else None)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def algorithm_ticks(ax, algorithms, *, horizontal=False):
    if horizontal:
        ax.set_yticks(range(len(algorithms)), algorithms)
    else:
        ax.set_xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')


def summary_figure(report, classifier):
    colors = palette(report.algorithms)
    fig, axes = panel_grid(len(report.metrics), width=max(5, len(report.algorithms) * .55))
    for ax, metric in zip(axes, report.metrics):
        values = metric_matrix(report, classifier, metric)
        ax.bar(range(len(report.algorithms)), values.mean(axis=1), color=list(colors.values()))
        algorithm_ticks(ax, report.algorithms)
        ax.set_ylabel(f'{metric.name} ({metric.unit})')
        ax.set_title(f'{classifier.upper()} — mean across {len(report.datasets)} datasets')
        style_axes(ax)
    return fig


def radar_values(report, classifier):
    """Classification metrics plus the established per-dataset feature efficiency."""
    available = {metric.run_key: metric for metric in report.metrics}
    metrics = [available[key] for key in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs') if key in available]
    labels = [metric.name for metric in metrics]
    values = [metric_matrix(report, classifier, metric) for metric in metrics]
    if 'FeatRuns' in available:
        features = metric_matrix(report, classifier, available['FeatRuns'])
        values.append(1 - features / np.maximum(features.max(axis=0), 1.0))
        labels.append('Feature\nefficiency')
    if not values:
        return labels, np.empty((len(report.algorithms), len(report.datasets), 0))
    return labels, np.stack(values, axis=-1)


def radar_figure(report, classifier, labels, values):
    colors = palette(report.algorithms)
    fig, axes = panel_grid(len(report.datasets), width=5, height=4.8, polar=True)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.r_[angles, angles[0]]
    for di, (ax, dataset) in enumerate(zip(axes, report.datasets)):
        for ai, algorithm in enumerate(report.algorithms):
            observed = values[ai, di]
            ax.plot(angles, np.r_[observed, observed[0]], color=colors[algorithm], label=algorithm,
                    linestyle=('-', '--', ':', '-.')[ai % 4], linewidth=1.5)
        ax.set_xticks(angles[:-1], labels, fontsize=9)
        ax.set_ylim(min(0., float(values.min())), max(1., float(values.max())))
        ax.set_title(f'{dataset} / {classifier.upper()}', pad=24)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=min(6, len(names)), frameon=False)
    return fig


def heatmap_figure(report, classifier, metric):
    values = metric_matrix(report, classifier, metric)
    fig, ax = plt.subplots(figsize=(max(5, len(report.datasets) * 1.15), max(3, len(report.algorithms) * .45)), layout='constrained')
    im = ax.imshow(values, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(report.datasets)), report.datasets, rotation=35, ha='right')
    algorithm_ticks(ax, report.algorithms, horizontal=True)
    ax.set_title(f'{classifier.upper()} — {metric.name} ({metric.unit})')
    fig.colorbar(im, ax=ax, label=f'Cached {framework().PLOT_RUN_AGGREGATION} run')
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f'{values[i,j]:.4g}', ha='center', va='center', fontsize=8,
                color='white' if im.norm(values[i,j]) > .6 else 'black')
    return fig


def precision_figure(values, algorithms, classifier):
    means, intervals = dataset_mean_ci(values)
    colors = palette(algorithms)
    fig, ax = plt.subplots(figsize=(max(7, max(map(len, algorithms)) * .12), max(3, len(algorithms)*.45)), layout='constrained')
    for i, algorithm in enumerate(algorithms):
        ax.errorbar(means[i], i, xerr=None if intervals is None else intervals[i], fmt='o',
                    color=colors[algorithm], markersize=6, capsize=4)
        ax.annotate(f'{means[i]:.4f}', (means[i], i), xytext=(8, 7), textcoords='offset points', fontsize=9)
    algorithm_ticks(ax, algorithms, horizontal=True)
    ax.invert_yaxis()
    ax.set_xlabel('Average precision (test)' + (' ± 95% CI' if intervals is not None else ' (CI unavailable)'))
    ax.set_title(classifier.upper())
    style_axes(ax, True)
    return fig


def observations(ax, values, algorithms, *, means=False):
    colors = palette(algorithms)
    for i, algorithm in enumerate(algorithms):
        ax.scatter(i + np.linspace(-.13, .13, values.shape[1]), values[i], s=28,
                   color=colors[algorithm], edgecolor='white', linewidth=.5, zorder=4)
        if means:
            ax.scatter(i, values[i].mean(), marker='D', s=65, color=colors[algorithm], edgecolor='black', zorder=5)
    algorithm_ticks(ax, algorithms)
    style_axes(ax)


def boxplot_figure(values, algorithms, classifier):
    fig, ax = plt.subplots(figsize=(max(6, len(algorithms)*.8), 5), layout='constrained')
    boxes = ax.boxplot(values.T, positions=np.arange(len(algorithms)), patch_artist=True,
                       showfliers=False, medianprops={'color': 'black', 'linewidth': 1.5})
    for box, color in zip(boxes['boxes'], palette(algorithms).values()):
        box.set_facecolor(color); box.set_alpha(.35)
    observations(ax, values, algorithms)
    ax.set_ylabel('Accuracy (test): all cached runs')
    ax.set_title(classifier.upper())
    return fig


def violin_figure(values, algorithms, classifier, metric_name='Recall'):
    """Draw densities only for nonconstant samples, retaining every observation."""
    fig, ax = plt.subplots(figsize=(max(6, len(algorithms)*.8), 5), layout='constrained')
    for i, color in enumerate(palette(algorithms).values()):
        if len(values[i]) > 1 and np.ptp(values[i]) > 0:
            parts = ax.violinplot([values[i]], positions=[i], widths=.78, showextrema=False)
            parts['bodies'][0].set_facecolor(color)
            parts['bodies'][0].set_alpha(.28)
        ax.hlines(np.median(values[i]), i-.3, i+.3, colors='black', linestyles='--', linewidth=1.3)
    observations(ax, values, algorithms, means=True)
    ax.set_ylabel(f'{metric_name} (test): all cached runs')
    ax.set_title(classifier.upper())
    ax.legend(handles=[Line2D([], [], marker='D', color='none', markerfacecolor='#777777', label='Mean'),
                       Line2D([], [], color='black', linestyle='--', label='Median'),
                       Line2D([], [], marker='o', color='none', markerfacecolor='#777777', label='Dataset/run')], frameon=False)
    return fig


def convergence_figure(report, classifier, datasets):
    fig, axes = panel_grid(len(datasets))
    colors = palette(report.algorithms)
    for ax, ds in zip(axes, datasets):
        for i, algorithm in enumerate(report.algorithms):
            row = report.indexed[ds, classifier, algorithm]
            index = framework()._plot_run_index(row)
            curves = row.get('CurvesAll', [])
            validated = [framework().validate_convergence_curve(c, getattr(report.args, 'epochs', len(c)), fit)
                         for c, fit in zip(curves, row['FitRuns'])]
            if index is None and not validated:
                curve = np.asarray(row['Curve'], dtype=float)
            elif len(validated) != len(row['FitRuns']):
                raise ValueError('Convergence requires individual curves aligned with FitRuns')
            else:
                curve = np.mean(np.stack(validated), axis=0) if index is None else validated[index]
            ax.plot(np.arange(len(curve)), curve, label=algorithm, color=colors[algorithm],
                    linestyle=('-', '--', ':', '-.')[i % 4])
        ax.set_title(f'{ds} / {classifier.upper()}')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Fitness')
        style_axes(ax)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=min(6, len(names)), frameon=False)
    return fig


def tradeoff_figure(features, runtime, algorithms, classifier):
    fig, ax = plt.subplots(figsize=(max(7, len(algorithms)*.85), 5), layout='constrained')
    twin = ax.twinx()
    colors = list(palette(algorithms).values())
    for axis, values, shift, hatch, alpha in ((ax, features.mean(axis=1), -.2, None, 1),
                                             (twin, runtime.mean(axis=1), .2, '///', .45)):
        axis.bar(np.arange(len(algorithms))+shift, values, width=.36, color=colors, hatch=hatch, alpha=alpha)
        axis.set_ylim(0, max(1., float(values.max())*1.2))
    algorithm_ticks(ax, algorithms)
    ax.set_ylabel('Average selected features'); twin.set_ylabel('Average runtime (s)')
    ax.set_title(classifier.upper())
    style_axes(ax); twin.spines['top'].set_visible(False)
    ax.legend(handles=[Patch(facecolor='#777777', label='Selected features'),
                       Patch(facecolor='#777777', hatch='///', alpha=.45, label='Runtime')], frameon=False)
    return fig


def publication_figures(report, skipped):
    """Eight figure types, applied uniformly to every actual classifier."""
    available = {metric.run_key: metric for metric in report.metrics}
    for ci, classifier in enumerate(report.classifiers, 1):
        suffix = f'c{ci}'
        yield f'generic_summary_{suffix}', summary_figure(report, classifier)
        labels, radar = radar_values(report, classifier)
        if len(labels) >= 3:
            yield f'generic_radar_{suffix}', radar_figure(report, classifier, labels, radar)
        else:
            skipped.append({'output': f'Radar/{classifier}', 'reason': 'Fewer than three available classification/feature-efficiency axes'})
        for mi, metric in enumerate(report.metrics, 1):
            yield f'generic_heatmap_{suffix}_m{mi}', heatmap_figure(report, classifier, metric)
        for key, stem, generate_figure in (('PSRuns', 'precision', precision_figure),
                                           ('AccRuns', 'accuracy_boxplot', boxplot_figure),
                                           ('RSRuns', 'recall_violin', violin_figure)):
            if key not in available:
                skipped.append({'output': f'{stem}/{classifier}', 'reason': f'{key} unavailable'})
                continue
            values = (metric_matrix(report, classifier, available[key]) if key == 'PSRuns'
                      else run_observations(report, classifier, available[key]))
            if key == 'PSRuns' and values.shape[1] < 2:
                skipped.append({'output': f'Precision CI/{classifier}', 'reason': 'At least two datasets required; only observed representatives are plotted'})
            if key == 'RSRuns':
                for i, algorithm in enumerate(report.algorithms):
                    if values.shape[1] < 2 or np.ptp(values[i]) == 0:
                        skipped.append({'output': f'Violin density/{classifier}/{algorithm}',
                                        'reason': 'Insufficient or constant run observations; observations, mean and median retained'})
            yield f'generic_{stem}_{suffix}', generate_figure(values, report.algorithms, classifier)
        if 'AccRuns' in available:
            yield f'07_violin_accuracy_{classifier}', violin_figure(
                run_observations(report, classifier, available['AccRuns']),
                report.algorithms, classifier, 'Accuracy')
        complete = []
        for ds in report.datasets:
            missing = [a for a in report.algorithms
                       if not np.asarray(report.indexed[ds, classifier, a].get('Curve', [])).size
                       or (framework().PLOT_RUN_AGGREGATION != 'mean'
                           and len(report.indexed[ds, classifier, a].get('CurvesAll', []))
                           != len(report.indexed[ds, classifier, a].get('FitRuns', [])))]
            if missing:
                skipped.append({'output': f'Convergence {ds}/{classifier}', 'reason': f'No aligned convergence history for {missing}'})
            else:
                complete.append(ds)
        if complete:
            yield f'generic_convergence_{suffix}', convergence_figure(report, classifier, complete)
        if {'FeatRuns', 'TimeRuns'} <= available.keys():
            yield f'generic_features_runtime_{suffix}', tradeoff_figure(
                metric_matrix(report, classifier, available['FeatRuns']),
                metric_matrix(report, classifier, available['TimeRuns']), report.algorithms, classifier)
        else:
            skipped.append({'output': f'Features/runtime/{classifier}', 'reason': 'Requires both FeatRuns and TimeRuns'})


def generate(report, destination):
    skipped = []
    with plt.rc_context(STYLE):
        for stem, fig in publication_figures(report, skipped):
            save_png(fig, Path(destination) / f'{stem}.png')
    return skipped


def statistical_figures(analysis, algorithms, metric):
    x, ranked, ranks = analysis['x'], analysis['ranked'], analysis['mean_ranks']
    labels = [algorithms[i] for i in ranked]
    k, n = len(algorithms), len(x)
    fig, ax = plt.subplots(figsize=(max(6, max(map(len, algorithms)) * .12), max(3, k * .45)), layout='constrained')
    ax.barh(range(k), ranks[ranked], color='#777777')
    ax.set_yticks(range(k), labels); ax.invert_yaxis()
    ax.set_xlabel('Average rank (1 = best)')
    style_axes(ax, True)
    yield 'generic_average_rank', fig

    reference = algorithms[0]
    pairs = analysis['pairs']
    comparisons = pairs[(pairs.Algorithm_A == reference) & np.isfinite(pairs.Holm_adjusted_p)]
    if len(comparisons):
        fig, ax = plt.subplots(figsize=(8, max(3, len(comparisons)*.5)), layout='constrained')
        values = comparisons.Holm_adjusted_p.to_numpy()
        # Keep zero p-values visible without assigning a made-up positive value.
        ax.scatter(values, np.arange(len(values)), color='#333333')
        for i, value in enumerate(values):
            ax.annotate(f'{value:.5g}', (value, i), xytext=(5, 5), textcoords='offset points')
        ax.axvline(.05, linestyle='--', color='#777777')
        ax.set_yticks(range(len(values)), comparisons.Algorithm_B); ax.invert_yaxis()
        ax.set_xlim(-.02, 1.08)
        ax.set_xlabel(f'Holm-adjusted p; {len(pairs)}-pair family')
        ax.set_title(f'{reference} versus other algorithms (configured reference)')
        style_axes(ax, True)
        yield 'generic_reference_comparisons', fig

    fig, ax = plt.subplots(figsize=(max(5, k * .6), max(4, k * .5)), layout='constrained')
    im = ax.imshow(np.ma.masked_invalid(analysis['matrix']), vmin=0, vmax=1, cmap='Greys_r')
    ax.set_xticks(range(k), algorithms, rotation=45, ha='right'); ax.set_yticks(range(k), algorithms)
    for i, j in np.ndindex((k, k)):
        p = analysis['matrix'][i, j]
        ax.text(j, i, f'{p:.3g}' if np.isfinite(p) else 'N/A', ha='center', va='center', fontsize=8,
                color='white' if p < .5 else 'black')
    fig.colorbar(im, ax=ax, label='Holm-adjusted p (all algorithm pairs)')
    yield 'generic_holm_heatmap', fig

    fig, ax = plt.subplots(figsize=(max(6, k * .65), 4), layout='constrained')
    ax.boxplot(x[:, ranked], positions=np.arange(k), showfliers=False)
    for pos, i in enumerate(ranked):
        ax.scatter(pos + np.linspace(-.18, .18, n), x[:, i], s=15, color='#555555')
    ax.set_xticks(range(k), labels, rotation=45, ha='right')
    ax.set_ylabel(f'{metric}: cached run mean per matched block')
    style_axes(ax)
    yield 'generic_block_distribution', fig
