"""Presentation-only destinations for new FULL experiment/report figures."""
from pathlib import Path
import re
from urllib.parse import quote


ROOT = "root"
INDIVIDUAL = "individual"
STATISTICS = "statistics"

# Main distribution views belong in the numbered publication set. Inferential
# rank/Holm figures are separate; extra generic/detail views stay individual.
MAIN_ESTIMATOR = "knn"
MAIN_METRIC = "accuracy"
METRIC_TOKENS = ("accuracy", "f1", "precision", "recall")


def filename_component(value):
    """Readable, reversible tokens; dataset/classifier names cannot escape folders."""
    return quote(str(value), safe="._-")


def main_figure_names(classifier=MAIN_ESTIMATOR, metric=MAIN_METRIC):
    if metric not in METRIC_TOKENS:
        raise ValueError(f'Unsupported publication metric: {metric}')
    classifier = filename_component(classifier)
    return (
        "01_resultados_clasificador_todos_datasets.png",
        f"02_radar_por_dataset_{classifier}.png",
        f"03_features_runtime_por_dataset_{classifier}.png",
        f"04_boxplot_{metric}_por_dataset_{classifier}.png",
        f"05_convergence_por_dataset_{classifier}.png",
        f"06_heatmap_{metric}_{classifier}.png",
        f"07_violin_{metric}_{classifier}.png",
        f"08_global_{metric}_distribution.png",
        "09_global_features_runtime_tradeoff.png",
    )
STATISTICAL_STEMS = frozenset({
    "generic_average_rank", "generic_reference_comparisons", "generic_holm_heatmap",
    "generic_block_distribution",
})


def full_figure_category(filename):
    stem = Path(filename).stem
    if Path(filename).name in main_figure_names(MAIN_ESTIMATOR) or (
            stem.startswith(("02_radar_por_dataset_", "03_features_runtime_por_dataset_",
                             "05_convergence_por_dataset_"))) or re.fullmatch(
            r'(?:04_boxplot_(?:accuracy|f1|precision|recall)_por_dataset_.+|'
            r'0[67]_(?:heatmap|violin)_(?:accuracy|f1|precision|recall)_.+|'
            r'08_global_(?:accuracy|f1|precision|recall)_distribution)', stem):
        return ROOT
    if stem in STATISTICAL_STEMS:
        return STATISTICS
    return INDIVIDUAL


def full_figure_directories(destination):
    root = Path(destination)
    directories = {ROOT: root, INDIVIDUAL: root / INDIVIDUAL, STATISTICS: root / STATISTICS}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def full_figure_path(destination, filename):
    category = full_figure_category(filename)
    return Path(destination) / ("" if category == ROOT else category) / filename
