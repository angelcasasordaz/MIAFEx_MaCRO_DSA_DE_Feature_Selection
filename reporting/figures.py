"""Shared MIAFEx publication figures adapted from the final Code Smell style.

Representatives use the existing PLOT_RUN_AGGREGATION; distributions retain
all real runs. No cache/science writes, model fitting, or curve reconstruction.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.stats import t

from reporting.core import framework, report_stage, CompletedReport
from reporting.paper_tables import METRICS
from figure_layout import main_figure_names, filename_component, full_figure_directories, full_figure_path, METRIC_TOKENS
from figure_text import localize_figure, visible_text
from plot_labels import plot_display_label, primary_algorithm, is_primary
import full_plot_style as base_style

STYLE = base_style.STYLE
COLORS = base_style.PALETTES['dark_classic']
palette = base_style.palette
style_axes = base_style.style_axes


def report_from_results(args, results):
    """Adapt supplied results for plotting without accessing scientific caches."""
    m = framework()
    indexed, algorithms, classifiers = {}, [], []
    for dataset, rows in results.items():
        for label, row in rows.items():
            parsed = m.parse_result_label(label, args)
            classifier = str(row.get('Estimator') or parsed['estimator']).lower()
            algorithm = m.optimizer_acronym(parsed['method'])
            if len(args.transfer_functions) > 1:
                algorithm += '_' + parsed['transfer_function'].upper()
            key = dataset, classifier, algorithm
            if key in indexed:
                raise ValueError(f'Ambiguous plotting observations: {key}')
            indexed[key] = row
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            if classifier not in classifiers:
                classifiers.append(classifier)
    preferred = [m.optimizer_acronym(method) + (f'_{tf.upper()}' if len(args.transfer_functions) > 1 else '')
                 for method in args.optimizers for tf in args.transfer_functions]
    algorithms = list(dict.fromkeys([a for a in preferred if a in algorithms] + algorithms))
    classifiers = list(dict.fromkeys([c for c in args.estimators if c in classifiers] + classifiers))
    metrics = [metric for metric in METRICS if indexed and all(metric.run_key in row for row in indexed.values())]
    return CompletedReport(argparse.Namespace(**vars(args)), results, indexed, list(results),
                           classifiers, algorithms, metrics, 'in-memory', {})


def metric_token(metric):
    return {'AccRuns': 'accuracy', 'PSRuns': 'precision', 'RSRuns': 'recall', 'F1Runs': 'f1',
            'FitRuns': 'fitness', 'FeatRuns': 'features', 'TimeRuns': 'runtime'}[metric.run_key]


def metric_matrix(report, classifier, metric):
    """ONE fitness-selected positional run shared by every representative metric."""
    def value(row):
        index = framework()._plot_run_index(row)
        values = np.asarray(row[metric.run_key], dtype=float)
        if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
                or (index is not None and len(values) != len(row['FitRuns']))):
            raise ValueError(f'{metric.run_key} must be finite and align with FitRuns')
        return float(np.mean(values) if index is None else values[index]) / metric.scale
    return np.asarray([[value(report.indexed[ds, classifier, opt])
                        for ds in report.datasets] for opt in report.algorithms])


def run_observations(report, classifier, metric):
    """All actual cached runs, independently of representative-run selection."""
    return np.asarray([np.concatenate([np.asarray(report.indexed[ds, classifier, opt][metric.run_key], dtype=float)
                                       / metric.scale for ds in report.datasets]) for opt in report.algorithms])


def metric_values(df, metric, classifier, datasets, algorithms):
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


def palette_name(report):
    return getattr(report.args, 'plot_color_palette', framework().PLOT_COLOR_PALETTE)


def reference_method(report):
    return primary_algorithm(report.algorithms, report.args)


def figure_language(report):
    language = getattr(report.args, 'figure_language', framework().FIGURE_LANGUAGE)
    if language not in ('en', 'es'):
        raise ValueError(f'Unsupported FIGURE_LANGUAGE: {language}')
    return language


def base_classifier(report):
    configured = str(getattr(report.args, 'plot_global_estimator', framework().PLOT_GLOBAL_ESTIMATOR)).lower()
    if configured not in report.classifiers:
        if hasattr(report.args, 'plot_global_estimator'):
            raise ValueError(f'Publication classifier {configured!r} has no selected results; available: {report.classifiers}')
        return report.classifiers[0]
    return configured


def base_metric_token(report):
    metric = str(getattr(report.args, 'plot_global_metric', framework().PLOT_GLOBAL_METRIC)).lower()
    if metric not in METRIC_TOKENS:
        raise ValueError(f'Unsupported PLOT_GLOBAL_METRIC: {metric}')
    return metric


def base_figure_names(report):
    return main_figure_names(base_classifier(report), base_metric_token(report))


def dataset_mean_ci(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Expected finite algorithm-by-dataset observations')
    n = values.shape[1]
    return values.mean(axis=1), (t.ppf(.975, n-1)*values.std(axis=1, ddof=1)/np.sqrt(n) if n > 1 else None)


def panel_grid(count, *, width=5, height=4, polar=False):
    if count < 1:
        raise ValueError('A figure needs at least one observed panel')
    rows, columns = base_style.grid_shape(count)
    fig, axes = plt.subplots(rows, columns, figsize=(width*columns, height*rows), squeeze=False,
                             subplot_kw={'polar': True} if polar else None)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def algorithm_ticks(ax, algorithms, *, horizontal=False):
    labels = [plot_display_label(a, algorithms) for a in algorithms]
    if horizontal:
        ax.set_yticks(range(len(algorithms)), labels)
    else:
        ax.set_xticks(range(len(algorithms)), labels, rotation=45, ha='right')


def common_legend(fig, algorithms, colors, reference, *, primary_linewidth=2.5):
    handles = [Line2D([], [], color=colors[a], label=plot_display_label(a, algorithms),
                       **base_style.line_style(a), markersize=4,
                       linewidth=primary_linewidth if is_primary(a, reference) else 1.3) for a in algorithms]
    rows = math.ceil(len(handles)/min(6, len(handles)))
    legend = fig.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)), framealpha=.95)
    for label in legend.get_texts():
        label.set_gid('result-identity')
    fig.tight_layout(rect=(0, min(.2, .035*rows+.025), 1, 1))


def summary_figure(report, classifier=None):
    """Classifier × metric grid averaging the selected dataset representatives."""
    colors, reference = palette(report.algorithms, palette_name(report)), reference_method(report)
    available = {m.run_key: m for m in report.metrics}
    metrics = ([available[k] for k in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs') if k in available]
               if classifier is None else report.metrics)
    classifiers = report.classifiers if classifier is None else [classifier]
    columns = min(4, len(metrics))
    rows_per_classifier = math.ceil(len(metrics)/columns)
    rows = rows_per_classifier*len(classifiers)
    fig, axes = plt.subplots(rows, columns, figsize=(max(4.2, .35*len(report.algorithms))*columns, 3*rows+1.2), squeeze=False)
    for ci, cls in enumerate(classifiers):
        for mi, metric in enumerate(metrics):
            ax = axes[ci*rows_per_classifier+mi//columns, mi % columns]
            values = metric_matrix(report, cls, metric).mean(axis=1)
            base_style.metric_bars(ax, values, report.algorithms, colors, reference)
            algorithm_ticks(ax, report.algorithms); ax.tick_params(labelsize=8)
            ax.set_ylim(0, 1.12 if metric.unit == '0–1' else max(1., float(values.max())*1.3))
            ax.set_ylabel(cls.upper() if classifier is None else f'{metric.name} ({metric.unit})')
            if ci == 0 or classifier is not None:
                header = {'AccRuns': 0, 'PSRuns': 1, 'RSRuns': 2, 'F1Runs': 3}.get(metric.run_key, mi)
                base_style.metric_header(ax, metric.name, header)
            style_axes(ax)
        for mi in range(len(metrics), rows_per_classifier*columns):
            axes[ci*rows_per_classifier+mi//columns, mi % columns].set_visible(False)
    common_legend(fig, report.algorithms, colors, reference)
    return fig


def radar_values(report, classifier):
    """Preserve MIAFEx's absolute selected/original input feature ratio."""
    available = {m.run_key: m for m in report.metrics}
    metrics = [available[k] for k in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs') if k in available]
    labels, values = [m.name for m in metrics], [metric_matrix(report, classifier, m) for m in metrics]
    if 'FeatRuns' in available:
        features = metric_matrix(report, classifier, available['FeatRuns'])
        if getattr(report.args, 'dataset_source', None) == 'miafex':
            try:
                counts = framework().plot_original_feature_counts(report.args, report.datasets)
            except FileNotFoundError:
                counts = None  # Omit unavailable denominator, never invent original dimensions.
            if counts:
                values.append(np.stack([framework().selected_feature_ratio(features[:, di], counts[dataset])
                                        for di, dataset in enumerate(report.datasets)], axis=1))
                labels.append('Selected Feature Ratio')
        else:
            values.append(1-features/np.maximum(features.max(axis=0), 1.))
            labels.append('Feature\nefficiency')
    return labels, (np.stack(values, axis=-1) if values else np.empty((len(report.algorithms), len(report.datasets), 0)))


def curve_draw_order(algorithms, reference=None):
    reference = reference if reference is not None else primary_algorithm(algorithms)
    return sorted(range(len(algorithms)), key=lambda i: is_primary(algorithms[i], reference))


def radar_figure(report, classifier, labels, values):
    colors, reference = palette(report.algorithms, palette_name(report)), reference_method(report)
    fig, axes = panel_grid(len(report.datasets), width=5.2, height=4.8, polar=True)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    closed = np.r_[angles, angles[0]]
    for di, (ax, dataset) in enumerate(zip(axes, report.datasets)):
        for ai in curve_draw_order(report.algorithms, reference):
            algorithm, observed = report.algorithms[ai], values[ai, di]
            selected = is_primary(algorithm, reference)
            ax.plot(closed, np.r_[observed, observed[0]], color=colors[algorithm], label=plot_display_label(algorithm),
                    **base_style.line_style(algorithm), markersize=4, linewidth=2.5 if selected else 1.2,
                    zorder=4 if selected else 2)
            if selected:
                ax.fill(closed, np.r_[observed, observed[0]], color=colors[algorithm], alpha=.05)
        ax.set_xticks(angles, labels, fontsize=8)
        for label in ax.get_xticklabels():
            label.set_gid('figure-visible')
        ax.set_ylim(min(0., float(values.min())), max(1., float(values.max())))
        ax.set_title(f'{dataset} / {classifier.upper()}', pad=18)
    common_legend(fig, report.algorithms, colors, reference)
    return fig


def heatmap_figure(report, classifier, metric):
    values = metric_matrix(report, classifier, metric)
    fig, ax = plt.subplots(figsize=base_style.figure_size('heatmap', len(report.algorithms), len(report.datasets)))
    im = ax.imshow(values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(report.datasets)), report.datasets, rotation=35, ha='right')
    algorithm_ticks(ax, report.algorithms, horizontal=True)
    ax.set_xlabel('Dataset'); ax.set_ylabel('Metaheuristics')
    ax.set_title(f'{classifier.upper()} — {metric.name} ({metric.unit})')
    fig.colorbar(im, ax=ax, label=f'Cached {framework().PLOT_RUN_AGGREGATION} run', shrink=.85)
    for i, algorithm in enumerate(report.algorithms):
        if is_primary(algorithm, reference_method(report)):
            ax.add_patch(Rectangle((-.5, i-.5), len(report.datasets), 1, fill=False, edgecolor='black', linewidth=2.3))
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f'{values[i,j]:.4f}', ha='center', va='center', fontsize=8,
                color='white' if values[i,j] > .8 else 'black')
    fig.tight_layout()
    return fig


def precision_figure(values, algorithms, classifier, *, palette_name=None, reference=None):
    means, intervals = dataset_mean_ci(values)
    colors = palette(algorithms, palette_name)
    fig, ax = plt.subplots(figsize=(8, max(3, len(algorithms)*.45)))
    for i, algorithm in enumerate(algorithms):
        ax.errorbar(means[i], i, xerr=None if intervals is None else intervals[i], fmt='o',
                    color=colors[algorithm], markersize=8 if is_primary(algorithm, reference) else 6, capsize=4)
        ax.annotate(f'{means[i]:.4f}', (means[i], i), xytext=(8, 7), textcoords='offset points', fontsize=8)
    algorithm_ticks(ax, algorithms, horizontal=True); ax.invert_yaxis()
    ax.set_xlabel('Average precision (test)' + (' ± 95% CI' if intervals is not None else ' (CI unavailable)'))
    ax.set_title(classifier.upper()); style_axes(ax, True); fig.tight_layout()
    return fig


def mean_value_labels(ax, values):
    for i, sample in enumerate(values):
        mean = float(np.mean(sample))
        above = mean >= float(np.median(sample))
        label = ax.annotate(f'{mean:.3f}', (i, mean), xytext=(0, 9 if above else -9), textcoords='offset points',
                            ha='center', va='bottom' if above else 'top', fontsize=8, fontweight='bold', zorder=6,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.5))
        label.set_in_layout(False)
    ax.margins(y=.12)


def draw_boxplot(ax, values, algorithms, *, palette_name=None, reference=None, show_mean_labels=True):
    # Original observations define quartiles/median/mean. Min/max whiskers show
    # the entire observed range without raw observation or outlier points.
    boxes = ax.boxplot(values.T, positions=np.arange(len(algorithms)), patch_artist=True, widths=.55,
                       whis=(0, 100), showfliers=False, showmeans=True,
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       meanprops={'marker': 'D', 'markersize': 6, 'markerfacecolor': 'black', 'markeredgecolor': 'white'})
    for box, algorithm, color in zip(boxes['boxes'], algorithms, palette(algorithms, palette_name).values()):
        box.set_facecolor(color); box.set_alpha(.7)
        base_style.highlight_patch(box, algorithm, reference)
    algorithm_ticks(ax, algorithms); style_axes(ax)
    if show_mean_labels:
        mean_value_labels(ax, values)
    else:
        ax.margins(y=.12)
    return boxes


def distribution_legend(ax):
    ax.legend(handles=[Line2D([], [], marker='D', color='none', markerfacecolor='black', label='Mean'),
                       Line2D([], [], color='black', linestyle='--', label='Median')], loc='lower right')


def boxplot_figure(values, algorithms, classifier, *, show_points=False, metric_name='Accuracy', palette_name=None, reference=None):
    fig, ax = plt.subplots(figsize=base_style.figure_size('boxplot', len(algorithms)))
    draw_boxplot(ax, values, algorithms, palette_name=palette_name, reference=reference)
    ax.set_ylabel(f'{metric_name} (test): cached runs across datasets')
    ax.set_title(classifier.upper()); distribution_legend(ax); fig.tight_layout()
    return fig


def dataset_boxplot_figure(report, classifier, metric=None):
    metric = metric or next(m for m in report.metrics if m.run_key == 'AccRuns')
    fig, axes = panel_grid(len(report.datasets), width=max(5.8, .5*len(report.algorithms)), height=4.6)
    for ax, dataset in zip(axes, report.datasets):
        values = np.asarray([report.indexed[dataset, classifier, a][metric.run_key] for a in report.algorithms], dtype=float)/metric.scale
        draw_boxplot(ax, values, report.algorithms, palette_name=palette_name(report), reference=reference_method(report),
                     show_mean_labels=False)
        ax.set_ylabel(f'{metric.name} (test)'); ax.set_title(f'{dataset} / {classifier.upper()}'); ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def violin_figure(values, algorithms, classifier, metric_name='Recall', *, show_points=False, palette_name=None, reference=None):
    """Real density, median, mean marker and number; no jitter/observation artists."""
    fig, ax = plt.subplots(figsize=base_style.figure_size('violin', len(algorithms)))
    for i, (algorithm, color) in enumerate(palette(algorithms, palette_name).items()):
        sample = np.asarray(values[i], dtype=float)
        if len(sample) > 1 and np.ptp(sample) > 0:
            parts = ax.violinplot([sample], positions=[i], widths=.78, showextrema=False)
            body = parts['bodies'][0]
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.65)
            base_style.highlight_patch(body, algorithm, reference)
        ax.hlines(np.median(sample), i-.3, i+.3, colors='black', linestyles='--', linewidth=1.3)
        ax.plot(i, np.mean(sample), marker='D', markersize=7, color='black', markeredgecolor='white', linestyle='none', zorder=5)
    algorithm_ticks(ax, algorithms); style_axes(ax); mean_value_labels(ax, values)
    ax.set_ylabel(f'{metric_name} (test): cached runs across datasets')
    ax.set_title(classifier.upper()); distribution_legend(ax); fig.tight_layout()
    return fig


def stored_curve(report, dataset, classifier, algorithm):
    """Select/average validated stored histories only, never reconstruct missing ones."""
    row = report.indexed.get((dataset, classifier, algorithm), {})
    index = framework()._plot_run_index(row)
    histories, fitness = row.get('CurvesAll'), np.asarray(row.get('FitRuns', []), dtype=float)
    epochs = getattr(report.args, 'epochs', None)
    if histories is not None:
        if len(histories) != len(fitness) or not len(histories):
            raise ValueError('Individual histories must align with FitRuns')
        valid = [framework().validate_convergence_curve(c, epochs or len(c), fit) for c, fit in zip(histories, fitness)]
        evidence = row.get('ConvergenceRuns', {})
        if not isinstance(evidence, dict):
            raise ValueError('Invalid convergence evidence')
        ids = list(row.get('CompletedRunIDs', range(len(valid))))
        for run_id, metadata in evidence.items():
            if run_id not in ids:
                raise ValueError('Convergence evidence contains an unknown run')
            curve = valid[ids.index(run_id)]
            framework().validate_convergence_metadata(metadata, curve, epochs or len(curve))
        return np.mean(np.stack(valid), axis=0) if index is None else valid[index]
    if index is not None:
        raise ValueError('No individual stored history for the fitness-selected run')
    curve = row.get('Curve', [])
    if not len(curve):
        raise ValueError('No stored curves')
    return framework().validate_convergence_curve(curve, epochs or len(curve), float(fitness.mean()) if fitness.size else None)


def final_stage_inset(ax, curves, algorithms, colors, language='en', reference=None):
    """Final 25%; low final-fitness band alone defines y limits (all curves drawn)."""
    if not curves or min(len(c) for c in curves) < 3:
        return None
    length = min(len(c) for c in curves)
    start = min(int(.75*length), length-2)
    finals = np.asarray([c[-1] for c in curves], dtype=float)
    cutoff = min(float(np.median(finals)), float(finals.min()+.1*np.ptp(finals)))
    tail = np.concatenate([c[start:] for c in curves if c[-1] <= cutoff])
    padding = max(float(np.ptp(tail))*.12, float(np.max(np.abs(tail)))*.001, 1e-6)
    inset = ax.inset_axes((.56, .16, .39, .30)); inset.set_in_layout(False)
    for i in curve_draw_order(algorithms, reference):
        algorithm, curve = algorithms[i], curves[i]
        line, = inset.plot(np.arange(start+1, len(curve)+1), curve[start:], color=colors[algorithm],
                   **base_style.line_style(algorithm), markersize=3, markevery=max(1, (len(curve)-start)//5),
                   linewidth=base_style.convergence_linewidth(algorithm, reference, inset=True),
                   zorder=4 if is_primary(algorithm, reference) else 2, label=plot_display_label(algorithm))
        if base_style.method_key(algorithm) == 'MACRO-DE-T':
            base_style.highlight_macro_t_convergence(line)
    inset.set_xlim(start+1, max(len(c) for c in curves)); inset.set_ylim(float(tail.min())-padding, float(tail.max())+padding)
    inset.set_title(visible_text('Final stage', language), fontsize=8, fontweight='normal', pad=3)
    inset.tick_params(labelsize=6); inset.grid(alpha=.2)
    inset.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3)); inset.yaxis.get_offset_text().set_fontsize(6)
    return inset


def convergence_figure(report, classifier, datasets, *, skipped=None):
    fig, axes = panel_grid(len(datasets), width=5.8, height=4.4)
    colors, reference = palette(report.algorithms, palette_name(report)), reference_method(report)
    for ax, dataset in zip(axes, datasets):
        curves, missing = {}, []
        for i in curve_draw_order(report.algorithms, reference):
            algorithm = report.algorithms[i]
            try:
                curve = stored_curve(report, dataset, classifier, algorithm)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                missing.append(algorithm)
                if skipped is not None:
                    skipped.append({'output': f'Convergence {dataset}/{classifier}/{algorithm}', 'reason': str(exc)})
                continue
            curves[algorithm] = curve
            line, = ax.plot(np.arange(1, len(curve)+1), curve, color=colors[algorithm], label=plot_display_label(algorithm),
                    **base_style.line_style(algorithm), markersize=4, markevery=max(1, len(curve)//12),
                    linewidth=base_style.convergence_linewidth(algorithm, reference), zorder=4 if is_primary(algorithm, reference) else 2)
            if base_style.method_key(algorithm) == 'MACRO-DE-T':
                base_style.highlight_macro_t_convergence(line)
        ax.set_title(f'{dataset} / {classifier.upper()}'); ax.set_xlabel('Iteration'); ax.set_ylabel('Fitness'); style_axes(ax)
        inset = (final_stage_inset(ax, [curves[a] for a in report.algorithms], report.algorithms, colors,
                                  figure_language(report), reference) if not missing else None)
        if inset is None and skipped is not None:
            skipped.append({'output': f'Convergence inset {dataset}/{classifier}',
                            'reason': f'Missing/invalid stored curves: {missing}' if missing else 'Fewer than three stored iterations'})
        if not curves:
            ax.text(.5, .5, 'No stored curves', transform=ax.transAxes, ha='center')
    common_legend(fig, report.algorithms, colors, reference,
                  primary_linewidth=base_style.convergence_linewidth(reference, reference))
    return fig


def tradeoff_value_labels(axis, bars):
    """Compact values above each bar, in that bar's own axis coordinates."""
    for bar in bars:
        value = bar.get_height()
        label = f'{value:.2f}'.rstrip('0').rstrip('.')
        axis.annotate(label, (bar.get_x()+bar.get_width()/2, value),
                      xytext=(0, 3), textcoords='offset points', rotation=90,
                      ha='center', va='bottom', fontsize=7, zorder=8, clip_on=False)


def draw_tradeoff_axes(ax, features, runtime, algorithms, *, palette_name=None, reference=None):
    twin, colors = ax.twinx(), palette(algorithms, palette_name)
    for axis, values, shift, hatch, alpha in ((ax, features, -.2, None, .85), (twin, runtime, .2, '///', .45)):
        bars = axis.bar(np.arange(len(algorithms))+shift, values, width=.36,
                        color=[colors[a] for a in algorithms], hatch=hatch, alpha=alpha)
        for bar, algorithm in zip(bars, algorithms):
            base_style.highlight_patch(bar, algorithm, reference)
        axis.set_ylim(0, max(1., float(np.max(values))*1.25))
        tradeoff_value_labels(axis, bars)
    algorithm_ticks(ax, algorithms); ax.set_ylabel('Selected features'); twin.set_ylabel('Runtime (s)'); style_axes(ax)
    ax.legend(handles=[Patch(facecolor='#777777', label='Selected features'),
                       Patch(facecolor='#777777', hatch='///', alpha=.45, label='Runtime')],
              fontsize=8, loc='upper center', ncol=2)
    return twin


def tradeoff_figure(features, runtime, algorithms, classifier, *, palette_name=None, reference=None):
    fig, ax = plt.subplots(figsize=base_style.figure_size('features_runtime', len(algorithms)))
    twin = draw_tradeoff_axes(ax, features.mean(axis=1), runtime.mean(axis=1), algorithms, palette_name=palette_name, reference=reference)
    ax.set_ylabel('Average selected features'); twin.set_ylabel('Average runtime (s)')
    ax.set_title(classifier.upper()); fig.tight_layout()
    return fig


def dataset_tradeoff_figure(report, classifier):
    features = metric_matrix(report, classifier, next(m for m in report.metrics if m.run_key == 'FeatRuns'))
    runtime = metric_matrix(report, classifier, next(m for m in report.metrics if m.run_key == 'TimeRuns'))
    fig, axes = panel_grid(len(report.datasets), width=max(5.8, .5*len(report.algorithms)), height=4.6)
    for di, (ax, dataset) in enumerate(zip(axes, report.datasets)):
        draw_tradeoff_axes(ax, features[:, di], runtime[:, di], report.algorithms, palette_name=palette_name(report), reference=reference_method(report))
        ax.set_title(f'{dataset} / {classifier.upper()}'); ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def _base_publication_figures(report, skipped):
    names, classifier = base_figure_names(report), base_classifier(report)
    available = {m.run_key: m for m in report.metrics}
    metric = next((m for m in report.metrics if metric_token(m) == base_metric_token(report)), None)
    if any(k in available for k in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs')):
        yield Path(names[0]).stem, summary_figure(report)
    else:
        skipped.append({'output': names[0], 'reason': 'No classification metrics'})
    labels, values = radar_values(report, classifier)
    if len(labels) >= 3:
        yield Path(names[1]).stem, radar_figure(report, classifier, labels, values)
    else:
        skipped.append({'output': names[1], 'reason': 'Fewer than three available radar axes'})
    if {'FeatRuns', 'TimeRuns'} <= available.keys():
        yield Path(names[2]).stem, dataset_tradeoff_figure(report, classifier)
    else:
        skipped.append({'output': names[2], 'reason': 'Requires FeatRuns and TimeRuns'})
    if metric:
        yield Path(names[3]).stem, dataset_boxplot_figure(report, classifier, metric)
    else:
        skipped.append({'output': names[3], 'reason': f'{base_metric_token(report)} unavailable'})
    yield Path(names[4]).stem, convergence_figure(report, classifier, report.datasets, skipped=skipped)
    if metric:
        yield Path(names[5]).stem, heatmap_figure(report, classifier, metric)
        observed = run_observations(report, classifier, metric)
        kwargs = dict(palette_name=palette_name(report), reference=reference_method(report), metric_name=metric.name)
        yield Path(names[6]).stem, violin_figure(observed, report.algorithms, classifier, **kwargs)
        yield Path(names[7]).stem, boxplot_figure(observed, report.algorithms, classifier, **kwargs)
    else:
        for name in names[5:8]:
            skipped.append({'output': name, 'reason': f'{base_metric_token(report)} unavailable'})
    if {'FeatRuns', 'TimeRuns'} <= available.keys():
        yield Path(names[8]).stem, tradeoff_figure(metric_matrix(report, classifier, available['FeatRuns']),
            metric_matrix(report, classifier, available['TimeRuns']), report.algorithms, classifier,
            palette_name=palette_name(report), reference=reference_method(report))
    else:
        skipped.append({'output': names[8], 'reason': 'Requires FeatRuns and TimeRuns'})


def localized_figures(report, generator):
    protected = [*report.datasets, *report.algorithms, *(plot_display_label(a) for a in report.algorithms),
                 *(c.upper() for c in report.classifiers)]
    for stem, fig in generator:
        localize_figure(fig, figure_language(report), protected=protected)
        yield stem, fig


def base_publication_figures(report, skipped):
    yield from localized_figures(report, _base_publication_figures(report, skipped))


def publication_figures(report, skipped):
    """Additional non-heatmap diagnostics placed under individual/."""
    available = {m.run_key: m for m in report.metrics}
    for classifier in report.classifiers:
        suffix = filename_component(classifier)
        yield f'generic_summary_{suffix}', summary_figure(report, classifier)
        labels, values = radar_values(report, classifier)
        if len(labels) >= 3:
            yield f'generic_radar_{suffix}', radar_figure(report, classifier, labels, values)
        if 'PSRuns' in available:
            yield f'generic_precision_{suffix}', precision_figure(metric_matrix(report, classifier, available['PSRuns']),
                report.algorithms, classifier, palette_name=palette_name(report), reference=reference_method(report))
        yield f'generic_convergence_{suffix}', convergence_figure(report, classifier, report.datasets, skipped=skipped)
        if {'FeatRuns', 'TimeRuns'} <= available.keys():
            yield f'generic_features_runtime_{suffix}', dataset_tradeoff_figure(report, classifier)


def per_dataset_figures(report, skipped):
    for dataset in report.datasets:
        single = replace(report, datasets=[dataset])
        for classifier in report.classifiers:
            suffix = f'{filename_component(dataset)}_{filename_component(classifier)}'
            labels, values = radar_values(single, classifier)
            if len(labels) >= 3:
                yield f'radar_{suffix}', radar_figure(single, classifier, labels, values)
            yield f'convergence_{suffix}', convergence_figure(single, classifier, [dataset], skipped=skipped)
            if {'FeatRuns', 'TimeRuns'} <= {m.run_key for m in report.metrics}:
                yield f'features_runtime_{suffix}', dataset_tradeoff_figure(single, classifier)


def generate(report, destination, *, generated=None, include_individual=True):
    destination = Path(destination)
    full_figure_directories(destination)
    skipped = []
    with plt.rc_context(STYLE):
        generators = [base_publication_figures(report, skipped)]
        if include_individual:
            generators.extend([localized_figures(report, publication_figures(report, skipped)),
                               localized_figures(report, per_dataset_figures(report, skipped))])
        for generator in generators:
            for stem, fig in generator:
                target = full_figure_path(destination, f'{stem}.png')
                save_png(fig, target)
                if generated is not None:
                    generated.append(str(target.relative_to(destination)))
    return skipped


def statistical_figures(analysis, algorithms, metric, *, reference=None, palette_name=None, language='en'):
    x, ranked, ranks = analysis['x'], analysis['ranked'], analysis['mean_ranks']
    labels, k = [algorithms[i] for i in ranked], len(algorithms)
    reference = reference if reference is not None else primary_algorithm(algorithms)
    colors = palette(algorithms, palette_name)
    def finish(fig):
        return localize_figure(fig, language, protected=[*algorithms, *(plot_display_label(a) for a in algorithms)])
    fig, ax = plt.subplots(figsize=(8, max(3, k*.45)), layout='constrained')
    bars = ax.barh(range(k), ranks[ranked], color=[colors[a] for a in labels])
    for bar, label in zip(bars, labels):
        base_style.highlight_patch(bar, label, reference)
        ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f' {bar.get_width():.3f}', va='center', fontsize=8)
    ax.set_yticks(range(k), [plot_display_label(a) for a in labels]); ax.invert_yaxis()
    ax.set_xlabel('Average rank (1 = best)'); style_axes(ax, True)
    yield 'generic_average_rank', finish(fig)
    pairs = analysis['pairs']
    comparisons = pairs[((pairs.Algorithm_A == reference) | (pairs.Algorithm_B == reference)) & np.isfinite(pairs.Holm_adjusted_p)]
    if len(comparisons):
        fig, ax = plt.subplots(figsize=(8, max(3, len(comparisons)*.5)), layout='constrained')
        values = comparisons.Holm_adjusted_p.to_numpy()
        others = [row.Algorithm_B if row.Algorithm_A == reference else row.Algorithm_A for row in comparisons.itertuples()]
        ax.scatter(values, np.arange(len(values)), color=colors[reference])
        for i, value in enumerate(values):
            ax.annotate(f'{value:.5g}', (value, i), xytext=(5, 5), textcoords='offset points')
        ax.axvline(.05, linestyle='--', color='#777777')
        ax.set_yticks(range(len(values)), [plot_display_label(a) for a in others]); ax.invert_yaxis()
        ax.set_xlim(-.02, 1.08); ax.set_xlabel(f'Holm-adjusted p; {len(pairs)}-pair family')
        ax.set_title(f"{plot_display_label(reference)} — {visible_text('Reference comparisons', language)}"); style_axes(ax, True)
        yield 'generic_reference_comparisons', finish(fig)
    fig, ax = plt.subplots(figsize=(max(5, k*.6), max(4, k*.5)), layout='constrained')
    im = ax.imshow(np.ma.masked_invalid(analysis['matrix']), vmin=0, vmax=1, cmap='Greys_r')
    labels_all = [plot_display_label(a) for a in algorithms]
    ax.set_xticks(range(k), labels_all, rotation=45, ha='right'); ax.set_yticks(range(k), labels_all)
    for i, j in np.ndindex((k, k)):
        p = analysis['matrix'][i, j]
        ax.text(j, i, (f'{p:.3g}' + (' *' if p < .05 and i != j else '')) if np.isfinite(p) else 'N/A',
                ha='center', va='center', fontsize=8, color='white' if p < .5 else 'black')
    fig.colorbar(im, ax=ax, label='Holm-adjusted p (all algorithm pairs)')
    yield 'generic_holm_heatmap', finish(fig)
    fig = boxplot_figure(x[:, ranked].T, labels, '', metric_name=metric, palette_name=palette_name, reference=reference)
    fig.axes[0].set_ylabel(f'{metric}: cached run mean per matched block')
    yield 'generic_block_distribution', finish(fig)
