"""Shared publication primitives; colors/styles never depend on selected order."""
import colorsys
import hashlib
import math

import matplotlib.colors as mcolors
import numpy as np
from plot_labels import method_key, plot_display_label, is_primary


STYLE_ID = "miafex-numbered-publication-v1"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 100, "savefig.dpi": 600,
    "grid.color": "#b0b0b0", "grid.linestyle": "-", "grid.linewidth": .8,
}
PALETTES = {
    "dark_classic": ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00",
                     "#6A3D9A", "#8DAA00", "#4D4D4D", "#A65628", "#999999", "#F0C808"),
    "deep": ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
             "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"),
    "muted": ("#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C",
              "#DC7EC0", "#797979", "#B8A33B", "#5A9EB5"),
    "tab20_dark": ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"),
    "dark2": ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#b89c00", "#a6761d", "#666666"),
    "paired": ("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
               "#4b86a3", "#638c39", "#a03e52", "#b78b28", "#766196", "#806744"),
    "accent": ("#43854e", "#81609b", "#ba7431", "#aaa72c", "#386cb0", "#b83183", "#bf5b17", "#666666"),
}
# Fixed registry: subsetting/reordering never reassigns an algorithm's color.
METHODS = ("MACRO-DE-T-V2", "DE", "JADE", "SHADE", "PSO", "WOA", "HHO", "GWO",
           "RUN", "DBO", "FOX", "BRO", "DSADE", "MACRO-DE", "MACRO-DE-T", "FLA")
# Proposed-method burgundy from the Code Smell reference palette.
EXTENSION_COLORS = {"dark_classic": {"MACRO-DE-T": "#7F0F1B"}}
HEADER_STYLES = (("#d8e8f3", "#b8d3e6"), ("#d2efee", "#abd9d7"),
                 ("#f7efd8", "#ead9ad"), ("#f9d5d9", "#edaeb8"))


def method_index(name):
    key = method_key(name)
    return METHODS.index(key) if key in METHODS else len(METHODS) + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def palette(algorithms, name=None):
    if name is None:
        from reporting.core import framework
        name = framework().PLOT_COLOR_PALETTE
    if name not in PALETTES:
        raise ValueError(f"Unsupported PLOT_COLOR_PALETTE: {name}")
    pool = PALETTES[name]
    colors, used = {}, set()
    for algorithm in sorted(algorithms, key=lambda a: (method_index(a), str(a))):
        index = method_index(algorithm)
        # Extend with deterministic, moderately saturated mid/dark colors. No
        # cycling through a short pool or dependence on the selected count.
        hue = (index * .618033988749895 + .11) % 1
        color = EXTENSION_COLORS.get(name, {}).get(method_key(algorithm)) or (
            pool[index] if index < len(pool) else
            mcolors.to_hex(colorsys.hls_to_rgb(hue, .34 + .05*(index % 3), .58)))
        # Distinct active identities (including transfer variants) must not
        # share an exact color. Resolve collisions in a stable identity order.
        attempt = 0
        while mcolors.to_hex(color) in used:
            digest = hashlib.sha256(f'{name}:{algorithm}:{attempt}'.encode()).hexdigest()
            hue = int(digest[:8], 16) / 2**32
            color = mcolors.to_hex(colorsys.hls_to_rgb(hue, .36, .58))
            attempt += 1
        colors[algorithm] = color
        used.add(mcolors.to_hex(color))
    return {algorithm: colors[algorithm] for algorithm in algorithms}


def convergence_linewidth(method, reference, *, inset=False):
    if is_primary(method, reference):
        return 3.5 if inset else 4.0
    return 1.1 if inset else 1.3


def highlight_macro_t_convergence(line):
    """Emphasize only a MaCRO-DE-t convergence artist, including inset curves."""
    line.set(color=EXTENSION_COLORS["dark_classic"]["MACRO-DE-T"],
             linestyle="-", linewidth=3.5, marker="D", markersize=7, zorder=10)


def line_style(name):
    index = method_index(name)
    return {"linestyle": ("-", "--", ":", "-.")[index % 4],
            "marker": ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p")[index % 12]}


def grid_shape(count):
    columns = min(3, max(1, math.ceil(math.sqrt(max(1, count)))))
    return math.ceil(max(1, count) / columns), columns


def figure_size(kind, algorithms, datasets=0):
    if kind == "heatmap":
        return max(10, .9*datasets+4), max(5, .45*algorithms+2)
    if kind in {"violin", "boxplot", "features_runtime"}:
        return max(10, .85*algorithms+3), 6
    raise ValueError(f"Unknown figure geometry: {kind}")


def style_axes(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", alpha=.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.tick_params(colors="black")


def highlight_patch(patch, method, reference=None, linewidth=2.4):
    if is_primary(method, reference):
        patch.set_edgecolor("black")
        patch.set_linewidth(linewidth)


def metric_header(ax, label, index):
    face, edge = HEADER_STYLES[index % len(HEADER_STYLES)]
    ax.set_title(label, fontsize=12, fontweight="bold", pad=12,
                 bbox=dict(boxstyle="round,pad=0.22", facecolor=face, edgecolor=edge))
    ax.title.set_gid('figure-visible')


def metric_bars(ax, values, algorithms, colors, reference=None):
    bars = ax.bar(np.arange(len(algorithms)), values, width=.68, color=[colors[a] for a in algorithms])
    offset = max(float(np.max(np.abs(values)))*.012, .006)
    for bar, method, value in zip(bars, algorithms, values):
        highlight_patch(bar, method, reference)
        ax.text(bar.get_x()+bar.get_width()/2, value+offset, f"{value:.3f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90)
    if np.isfinite(values).any():
        ax.axhline(np.mean(values), color="#a44b4b", linestyle="--", linewidth=.9, alpha=.8)
    return bars


display_label = plot_display_label
