# MIAFEx + Metaheuristic Feature Selection Framework
# Standard library imports
import argparse
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
import inspect
from typing import Dict, List

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from mafese import Data, MhaSelector, get_dataset
from mafese.utils.mealpy_util import FeatureSelectionProblem
from mafese.utils.estimator import get_general_estimator
from mealpy.swarm_based.DMOA import OriginalDMOA
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Local project imports
from dbo_optimizer import DBOOptimizer
from dsade_optimizer import DSADE
from dsade_awad_optimizer import DSADE_AWAD
from macro_de_optimizer import MaCRO_DE
from algorithm_acronym_list import (
    list_available_optimizers,
    optimizer_acronym,
    resolve_optimizer_name,
)

try:
    from train_miafex import train_miafex
    from extract_miafex_features import extract_miafex_features
    MIAFEX_IMPORT_ERROR = None
except Exception as exc:
    train_miafex = None
    extract_miafex_features = None
    MIAFEX_IMPORT_ERROR = exc

def available_memory_bytes() -> int | None:
    """Available RAM (not total RAM), or None if the OS cannot report it."""
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return max(0, int(line.split()[1]) * 1024)
    except (OSError, ValueError):
        pass
    try:
        return max(0, int(importlib.import_module("psutil").virtual_memory().available))
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages >= 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def automatic_worker_count() -> int:
    """Cap workers by usable CPUs and available RAM; always allow one worker."""
    cpus = os.cpu_count() or 1
    try:
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    cpu_limit = max(1, int(cpus * AUTO_WORKER_CPU_FRACTION))
    available = available_memory_bytes()
    if available is None:
        return cpu_limit
    ram_limit = max(1, (available - AUTO_RAM_RESERVE_BYTES) // AUTO_WORKER_RAM_BYTES)
    return min(cpu_limit, ram_limit)


# User-editable configuration: the controlled test used by PyCharm's Run action.
# Dataset and pipeline
DATASET_SOURCE = "miafex"  # Options: "miafex", "mafese"
PIPELINE_MODE = "feature_selection"  # Existing CSVs only; also supports "extract" and "full".
MIAFEX_DATASETS = None
# ["Brain_MRI"]
# None: all valid discovered datasets.
MAFESE_DATASET_SUITE = "test14"

# MIAFEx artifacts and neural-network settings
MIAFEX_DATASET_ROOT = "datasets"
MIAFEX_CHECKPOINT_ROOT = "checkpoints/miafex"
FEATURE_DATASET_ROOT = "datasets_features/miafex"
MIAFEX_TRAIN = "auto"  # Options: "auto", "yes", "no"
MIAFEX_EXTRACT = "auto"  # Options: "auto", "yes", "no"
MIAFEX_EPOCHS = 10  # Neural-network training epochs.
MIAFEX_BATCH_SIZE = 8
MIAFEX_LEARNING_RATE = 1e-5

# Feature selection: supported optimizers/classifiers remain available via config/CLI.
OPTIMIZERS = [
    "DE",
    "JADE",
    "SHADE",
    # "PSO",
    # "GWO",
    # "WOA",
    # "HHO",
    # "BRO",
    # "DBO",
    # "FLA",
    "MaCRO-DE",
]
ESTIMATORS = ["knn", "svm"]
TRANSFER_FUNCTIONS = ["vstf_01"]
RUNS = 20
FS_EPOCHS = 150  # Metaheuristic feature-selection iterations.
POP_SIZE = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42
SEED_BASE = 1234
DSADE_BETA_MIN = 0.2
DSADE_BETA_MAX = 0.8
DSADE_PCR = 0.2
DSADE_MAHAL_Q = 0.68

# Experiment and cache reuse
EXP_ID = 602
REUSE_CACHE = True
REUSE_CACHE_FROM_EXP_ID = 602  # None: current EXP only; another ID: read-only fallback.
FIGURES_ONLY = False

# Workers: native BLAS/OpenMP thread settings are deliberately unchanged.
PARALLEL = True
AUTO_WORKER_CPU_FRACTION = 2 / 3
AUTO_WORKER_RAM_BYTES = 192 * 1024**2
AUTO_RAM_RESERVE_BYTES = 512 * 1024**2
N_WORKERS = automatic_worker_count()  # Also capped by pending runs at execution time.
PROGRESS_INTERVAL_SECONDS = 60.0

# Backend and output
DEFAULT_COMPUTE_MODE = "torch-gpu"
DEFAULT_ML_BACKEND = "auto"
OUTPUT_ROOT = "."

# Supported values and compatibility aliases (edit the configuration above).
COMPUTE_MODES = ("cpu", "torch-gpu", "gpu-full")
ML_BACKENDS = ("auto", "sklearn", "cuml")
DEFAULT_OPTIMIZERS = OPTIMIZERS
DEFAULT_ESTIMATORS = ESTIMATORS
DEFAULT_TRANSFER_FUNCTIONS = TRANSFER_FUNCTIONS

TEST_datasets_clasific_14 = [
    "BreastCancer",
    "BreastEW",
    "Glass",
    "HeartEW",
    "Ionosphere",
    "Lymphography",
    "Sonar",
    "SpectEW",
    "Tic-tac-toe",
    "Wine",
    "WaveformEW",
    "Zoo",
]
SUPPORTED_ESTIMATORS = ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"]
SUPPORTED_TRANSFER_FUNCTIONS = [
    "vstf_01",
    "vstf_02",
    "vstf_03",
    "vstf_04",
    "sstf_01",
    "sstf_02",
    "sstf_03",
    "sstf_04",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# Helper types and validation functions
@dataclass
class Paths:
    exp_tag: str
    fig_dir: str
    res_dir: str
    cache_dir: str


@dataclass(frozen=True)
class BackendAvailability:
    torch_installed: bool
    torch_cuda_available: bool
    gpu_device_name: str | None
    cupy_available: bool
    cuml_available: bool
    torch_error: str | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    compute_mode: str
    requested_ml_backend: str
    selected_ml_backend: str
    miafex_device: str
    availability: BackendAvailability


def _module_available(module_name: str) -> bool:
    """Check an optional dependency without importing or initializing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def detect_backend_availability() -> BackendAvailability:
    """Detect GPU capabilities while keeping CuPy and cuML imports lazy."""
    torch_installed = _module_available("torch")
    torch_cuda_available = False
    gpu_device_name = None
    torch_error = None

    if torch_installed:
        try:
            torch = importlib.import_module("torch")
            torch_cuda_available = bool(torch.cuda.is_available())
            if torch_cuda_available:
                gpu_device_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception as exc:
            torch_error = f"{type(exc).__name__}: {exc}"

    return BackendAvailability(
        torch_installed=torch_installed,
        torch_cuda_available=torch_cuda_available,
        gpu_device_name=gpu_device_name,
        cupy_available=_module_available("cupy"),
        cuml_available=_module_available("cuml"),
        torch_error=torch_error,
    )


def resolve_execution_config(args: argparse.Namespace) -> ExecutionConfig:
    availability = detect_backend_availability()
    miafex_device = "cpu"
    if args.compute_mode in {"torch-gpu", "gpu-full"} and availability.torch_cuda_available:
        miafex_device = "cuda"

    # Keep the established sklearn classifiers until explicit cuML adapters exist.
    selected_ml_backend = "sklearn" if args.ml_backend == "auto" else args.ml_backend
    return ExecutionConfig(
        compute_mode=args.compute_mode,
        requested_ml_backend=args.ml_backend,
        selected_ml_backend=selected_ml_backend,
        miafex_device=miafex_device,
        availability=availability,
    )


def print_backend_report(config: ExecutionConfig) -> None:
    availability = config.availability
    print("=" * 60)
    print(" Execution backend")
    print("=" * 60)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"{'Python':<24}: {python_version}")
    for label, distribution_name in (
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("pandas", "pandas"),
        ("scikit-learn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("MAFESE", "mafese"),
        ("MEALPY", "mealpy"),
        ("Permetrics", "permetrics"),
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("timm", "timm"),
    ):
        print(f"{label:<24}: {_package_version(distribution_name)}")
    print(f"{'PyTorch CUDA available':<24}: {'yes' if availability.torch_cuda_available else 'no'}")
    print(f"{'GPU device':<24}: {availability.gpu_device_name or 'none detected'}")
    print(f"{'CuPy available':<24}: {'yes' if availability.cupy_available else 'no'}")
    print(f"{'cuML available':<24}: {'yes' if availability.cuml_available else 'no'}")
    print(f"{'Selected compute mode':<24}: {config.compute_mode}")
    print(f"{'ML backend request':<24}: {config.requested_ml_backend}")
    print(f"{'Selected ML backend':<24}: {config.selected_ml_backend}")
    print(f"{'MIAFEx torch device':<24}: {config.miafex_device}")
    print(f"{'Feature selection':<24}: CPU (MEALPY/MAFESE unchanged)")
    print(f"{'Classifier implementation':<24}: sklearn (unchanged)")

    if availability.torch_error:
        print(f"[backend] PyTorch CUDA detection failed: {availability.torch_error}")
    if config.compute_mode in {"torch-gpu", "gpu-full"} and not availability.torch_cuda_available:
        print("[backend] CUDA was requested but is unavailable; using the project's existing CPU fallback.")
    if config.compute_mode == "gpu-full":
        if not availability.cupy_available:
            print("[backend] gpu-full: CuPy is not installed; optional CuPy support is disabled.")
        if not availability.cuml_available:
            print("[backend] gpu-full: cuML is not installed; optional cuML support is disabled.")
        print("[backend] gpu-full is capability-only for now; optimizer and classifier logic is unchanged.")
    if config.selected_ml_backend == "cuml":
        print("[backend] cuML was selected as a future backend; classifiers remain sklearn until adapters are implemented.")
    print("=" * 60)


def validate_execution_config(config: ExecutionConfig) -> None:
    if config.selected_ml_backend != "cuml":
        return
    if config.compute_mode != "gpu-full":
        raise ValueError("--ml-backend cuml requires --compute-mode gpu-full.")
    if not config.availability.cuml_available:
        raise RuntimeError(
            "--ml-backend cuml was requested, but cuML is not installed. "
            "Select auto/sklearn or install a CUDA-compatible cuML build after checking the driver/CUDA version."
        )
    try:
        importlib.import_module("cuml")
    except Exception as exc:
        raise RuntimeError(
            "--ml-backend cuml was requested and the module was found, but it could not be imported. "
            "Check that the cuML build matches the NVIDIA driver and CUDA environment."
        ) from exc


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIAFEx feature datasets + MAFESE/MEALPY feature selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("General")
    general.add_argument("--exp-id", type=int, default=EXP_ID, help="ID numerico del experimento")
    general.add_argument("--output-root", default=OUTPUT_ROOT, help="Raiz para Figures/Results")
    general.add_argument("--pipeline-mode", default=PIPELINE_MODE, choices=["extract", "feature_selection", "full"],
                         help="extract: prepare/reuse features only; feature_selection: existing CSVs only; full: both stages")

    dataset = parser.add_argument_group("Dataset")
    dataset.add_argument("--dataset-source", default=DATASET_SOURCE, choices=["mafese", "miafex"], help="Origen de datasets")
    dataset.add_argument("--dataset-suite", default=MAFESE_DATASET_SUITE, choices=["test14"], help="Suite de datasets")
    selection = dataset.add_mutually_exclusive_group()
    selection.add_argument("--dataset-name", default=None, help="Seleccionar un unico dataset MIAFEx")
    selection.add_argument("--miafex-datasets", nargs="+", default=MIAFEX_DATASETS,
                           help="Seleccionar datasets MIAFEx; None usa todos los descubiertos")
    dataset.add_argument("--features-csv", default=None,
                         help="Ubicacion legacy: usa train_features.csv y test_features.csv junto a esta ruta; nunca divide un CSV unico")
    dataset.add_argument("--train-features-csv", default=None, help="Override del CSV de entrenamiento MIAFEx para un unico dataset")
    dataset.add_argument("--test-features-csv", default=None, help="Override del CSV de prueba MIAFEx para un unico dataset")
    dataset.add_argument("--list-miafex-datasets", action="store_true", help="Listar datasets de imagenes validos bajo --miafex-dataset-root")
    dataset.add_argument("--test-size", type=float, default=TEST_SIZE, help="Holdout ratio")
    dataset.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Semilla de split")

    miafex = parser.add_argument_group("MIAFEx")
    miafex.add_argument("--dataset-root", default=None, help="Raiz del dataset MIAFEx con subdirectorios train/ y test/")
    miafex.add_argument("--miafex-dataset-root", default=MIAFEX_DATASET_ROOT, help="Raiz que contiene los datasets de imagenes")
    miafex.add_argument("--miafex-checkpoint-root", default=MIAFEX_CHECKPOINT_ROOT, help="Raiz de checkpoints; un subdirectorio por dataset")
    miafex.add_argument("--feature-dataset-root", default=FEATURE_DATASET_ROOT, help="Raiz de features reutilizables; un subdirectorio por dataset")
    miafex.add_argument("--train-miafex", default=MIAFEX_TRAIN, choices=["auto", "yes", "no"], help="auto: reutilizar checkpoint existente; yes: entrenar; no: no entrenar")
    miafex.add_argument("--extract-miafex", default=MIAFEX_EXTRACT, choices=["auto", "yes", "no"], help="auto: reutilizar CSV existente; yes: extraer; no: exigir CSV existente")
    miafex.add_argument("--miafex-output", default=None, help="Override compatible del directorio de checkpoint para un unico dataset")
    miafex.add_argument("--miafex-epochs", type=int, default=MIAFEX_EPOCHS, help="Epocas de entrenamiento de la red neuronal MIAFEx")
    miafex.add_argument("--miafex-batch-size", type=int, default=MIAFEX_BATCH_SIZE, help="Batch size para MIAFEx")
    miafex.add_argument("--miafex-learning-rate", type=float, default=MIAFEX_LEARNING_RATE, help="Learning rate para MIAFEx")

    feature_selection = parser.add_argument_group("Feature Selection")
    feature_selection.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS), help="Lista de optimizadores")
    feature_selection.add_argument("--list-optimizers", action="store_true", help="Listar optimizadores MEALPY/custom disponibles")
    feature_selection.add_argument("--estimators", nargs="+", default=list(ESTIMATORS), help="Lista de clasificadores")
    feature_selection.add_argument("--transfer-functions", nargs="+", default=list(TRANSFER_FUNCTIONS), help="Lista de transfer functions")
    feature_selection.add_argument("--runs", type=int, default=RUNS, help="Ejecuciones independientes por combinacion")
    feature_selection.add_argument("--epochs", "--fs-epochs", dest="epochs", type=int, default=FS_EPOCHS, help="Iteraciones de seleccion de features (metaheuristica), no epocas MIAFEx")
    feature_selection.add_argument("--pop-size", type=int, default=POP_SIZE, help="Tamano de poblacion")

    execution = parser.add_argument_group("Execution")
    execution.add_argument(
        "--compute-mode",
        default=DEFAULT_COMPUTE_MODE,
        choices=COMPUTE_MODES,
        help="Backend mode: cpu, torch-gpu (recommended), or gpu-full (optional capability detection)",
    )
    execution.add_argument(
        "--ml-backend",
        default=DEFAULT_ML_BACKEND,
        choices=ML_BACKENDS,
        help="ML backend selector; auto preserves sklearn, cuml is a future integration hook",
    )
    execution.add_argument(
        "--show-backends",
        action="store_true",
        help="Print Python/GPU/backend availability and exit without running an experiment",
    )
    execution.add_argument("--seed-base", type=int, default=SEED_BASE, help="Semilla base por run")
    execution.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=REUSE_CACHE, help="Usar cache si existe")
    execution.add_argument("--reuse-cache-from-exp-id", type=lambda value: None if value.lower() == "none" else int(value),
                           default=REUSE_CACHE_FROM_EXP_ID,
                           help="EXP fuente de solo lectura si no hay cache actual compatible; 'none' desactiva importacion")
    execution.add_argument("--figures-only", action="store_true", default=FIGURES_ONLY, help="Regenerar solo graficas desde cache existente")
    execution.add_argument("--parallel", default="yes" if PARALLEL else "no", choices=["yes", "no"], help="Ejecutar runs en paralelo: yes/no")
    execution.add_argument("--n-workers", type=int, default=N_WORKERS, help="Maximo de procesos; se limita automaticamente a los runs pendientes")

    macro_dsade = parser.add_argument_group("MaCRO-DE / DSADE")
    macro_dsade.add_argument("--dsade-beta-min", type=float, default=DSADE_BETA_MIN)
    macro_dsade.add_argument("--dsade-beta-max", type=float, default=DSADE_BETA_MAX)
    macro_dsade.add_argument("--dsade-pcr", type=float, default=DSADE_PCR)
    macro_dsade.add_argument("--dsade-mahal-q", type=float, default=DSADE_MAHAL_Q)
    return parser.parse_args(argv)

def resolve_optimizers(args: argparse.Namespace) -> List[str]:
    return list(dict.fromkeys(resolve_optimizer_name(name) for name in args.optimizers))

def validate_selection_options(args: argparse.Namespace) -> None:
    invalid_estimators = [e for e in args.estimators if e not in SUPPORTED_ESTIMATORS]
    if invalid_estimators:
        raise ValueError(
            f"Clasificadores no soportados: {invalid_estimators}. "
            f"Validos: {', '.join(SUPPORTED_ESTIMATORS)}"
        )
    invalid_tf = [tf for tf in args.transfer_functions if tf not in SUPPORTED_TRANSFER_FUNCTIONS]
    if invalid_tf:
        raise ValueError(
            f"Transfer functions no soportadas: {invalid_tf}. "
            f"Validas: {', '.join(SUPPORTED_TRANSFER_FUNCTIONS)}"
        )

def make_paths(args: argparse.Namespace, *, exp_id: int | None = None, create: bool = True) -> Paths:
    exp_tag = f"EXP{args.exp_id if exp_id is None else exp_id:03d}"
    fig_dir = os.path.join(args.output_root, "Figures", exp_tag)
    res_dir = os.path.join(args.output_root, "Results", exp_tag)
    cache_dir = os.path.join(res_dir, "cache")
    if create:
        for p in (fig_dir, res_dir, cache_dir):
            os.makedirs(p, exist_ok=True)
    return Paths(exp_tag=exp_tag, fig_dir=fig_dir, res_dir=res_dir, cache_dir=cache_dir)

def resolve_mafese_dataset_names(args: argparse.Namespace) -> List[str]:
    if args.dataset_suite == "test14":
        return list(TEST_datasets_clasific_14)
    raise ValueError(f"Suite de datasets no soportada: {args.dataset_suite}")


def valid_miafex_dataset(dataset_root: str) -> bool:
    """Require matching, populated ImageFolder classes in train and test."""
    class_names = []
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
    for split in ("train", "test"):
        folder = os.path.join(dataset_root, split)
        if not os.path.isdir(folder):
            return False
        classes = {item.name: item.path for item in os.scandir(folder)
                   if item.is_dir() and not item.name.startswith(".")}
        if not classes or any(
            not any(filename.lower().endswith(image_extensions)
                    for _, _, files in os.walk(path) for filename in files)
            for path in classes.values()
        ):
            return False
        class_names.append(set(classes))
    return class_names[0] == class_names[1]


def discover_miafex_datasets(base_dir=MIAFEX_DATASET_ROOT) -> Dict[str, str]:
    if not os.path.isdir(base_dir):
        return {}

    discovered = {}
    for item in os.scandir(base_dir):
        if not item.is_dir():
            continue
        dataset_root = item.path
        if valid_miafex_dataset(dataset_root):
            discovered[item.name] = dataset_root
    return dict(sorted(discovered.items(), key=lambda row: row[0].lower()))


def print_miafex_datasets(datasets: Dict[str, str], base_dir=MIAFEX_DATASET_ROOT) -> None:
    if not datasets:
        print(f"No MIAFEx image datasets found in: {base_dir}")
        print()
        print("Expected structure:")
        print(os.path.join(base_dir, "DatasetName", "train", "<class>"))
        print(os.path.join(base_dir, "DatasetName", "test", "<class>"))
        return
    print("Available MIAFEx datasets:")
    for idx, name in enumerate(datasets, start=1):
        print(f"{idx}. {name}")


def resolve_miafex_dataset_root(args: argparse.Namespace) -> None:
    if args.dataset_root:
        return

    discovered = discover_miafex_datasets(args.miafex_dataset_root)
    if args.dataset_name in discovered:
        args.dataset_root = discovered[args.dataset_name]
        return

    available = ", ".join(discovered.keys()) if discovered else "none"
    raise ValueError(
        f"Dataset MIAFEx '{args.dataset_name}' no encontrado en {args.miafex_dataset_root}. "
        f"Datasets disponibles: {available}"
    )


def resolve_miafex_dataset_args(args: argparse.Namespace) -> Dict[str, argparse.Namespace]:
    """Select datasets and scope the existing single-dataset options per dataset."""
    discovered = discover_miafex_datasets(args.miafex_dataset_root)
    available = dict(discovered)
    if args.pipeline_mode == "feature_selection" or args.figures_only:
        # Feature-only runs can work even when the original images are offline.
        if os.path.isdir(args.feature_dataset_root):
            for item in os.scandir(args.feature_dataset_root):
                if item.is_dir() and all(os.path.isfile(os.path.join(item.path, f"{split}_features.csv"))
                                        for split in ("train", "test")):
                    available.setdefault(item.name, os.path.join(args.miafex_dataset_root, item.name))

    selected = [args.dataset_name] if args.dataset_name else args.miafex_datasets
    if selected is None:
        selected = ([os.path.basename(os.path.normpath(args.dataset_root))] if args.dataset_root
                    else sorted(available, key=str.lower))
    if not selected:
        raise ValueError(f"No MIAFEx datasets selected or discovered under {args.miafex_dataset_root}.")
    if any(not isinstance(name, str) or name in {".", ".."} or not name
           or "/" in name or "\\" in name for name in selected):
        raise ValueError("MIAFEx dataset selections must be directory names, not paths.")
    selected = list(dict.fromkeys(selected))
    if len(selected) != 1 and any((args.dataset_root, args.features_csv, args.train_features_csv,
                                   args.test_features_csv, args.miafex_output)):
        raise ValueError("Dataset, feature-CSV and checkpoint directory overrides require a single selected dataset.")

    resolved = {}
    for name in selected:
        scoped = argparse.Namespace(**vars(args))
        scoped.dataset_name = name
        if not scoped.dataset_root:
            if name in available:
                scoped.dataset_root = available[name]
            elif not (scoped.features_csv or (scoped.train_features_csv and scoped.test_features_csv)) and not args.figures_only:
                raise ValueError(f"Unknown MIAFEx dataset '{name}'. Available: {', '.join(sorted(available)) or 'none'}")
            else:
                scoped.dataset_root = os.path.join(args.miafex_dataset_root, name)
        scoped.miafex_output = args.miafex_output or os.path.join(args.miafex_checkpoint_root, name)
        feature_dir = (os.path.dirname(args.features_csv) or ".") if args.features_csv else os.path.join(args.feature_dataset_root, name)
        scoped.train_features_csv = args.train_features_csv or os.path.join(feature_dir, "train_features.csv")
        scoped.test_features_csv = args.test_features_csv or os.path.join(feature_dir, "test_features.csv")
        if os.path.realpath(scoped.train_features_csv) == os.path.realpath(scoped.test_features_csv):
            raise ValueError("Train and test feature CSVs must have different paths.")
        scoped.features_csv = args.features_csv or scoped.train_features_csv
        resolved[name] = scoped
    return resolved


def read_miafex_csv(csv_path: str):
    if not csv_path:
        raise ValueError("--features-csv es requerido cuando --dataset-source=miafex")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV de features no encontrado: {os.path.abspath(csv_path)}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV de features vacio: {os.path.abspath(csv_path)}") from exc

    if df.empty:
        raise ValueError(f"CSV de features vacio: {os.path.abspath(csv_path)}")
    if df.shape[1] < 2:
        raise ValueError("El CSV de MIAFEx debe contener al menos una columna de features y una etiqueta.")

    label_col = "label" if "label" in df.columns else df.columns[-1]
    y_series = df[label_col]
    X_df = df.drop(columns=[label_col])

    try:
        X = X_df.to_numpy(dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Las columnas de features del CSV de MIAFEx deben ser numericas.") from exc

    if not np.isfinite(X).all() or y_series.isna().any():
        raise ValueError(f"Non-finite features or missing labels in: {csv_path}")
    return X_df, y_series


def load_miafex_csv(csv_path: str):
    """Read an individual CSV; paired partitions use load_miafex_feature_data."""
    X_df, y_series = read_miafex_csv(csv_path)
    y_values = y_series.to_numpy()
    if pd.api.types.is_numeric_dtype(y_series):
        y = y_values
    else:
        y = LabelEncoder().fit_transform(y_values.astype(str))

    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("El CSV de MIAFEx debe contener al menos 2 clases.")

    return X_df.to_numpy(dtype=np.float64), y


def miafex_feature_paths(args: argparse.Namespace) -> Dict[str, str]:
    return {"train": args.train_features_csv, "test": args.test_features_csv}


def load_miafex_feature_data(csv_paths: Dict[str, str]) -> Data:
    """Load the prepared partitions directly; never concatenate or resplit them."""
    X_train, y_train = read_miafex_csv(csv_paths["train"])
    X_test, y_test = read_miafex_csv(csv_paths["test"])
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns must match in both names and order.")
    if pd.api.types.is_numeric_dtype(y_train) != pd.api.types.is_numeric_dtype(y_test):
        raise ValueError("Train/test labels must use the same type and class mapping.")
    if pd.api.types.is_numeric_dtype(y_train):
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise ValueError("Train/test labels must be finite.")
        if not set(np.unique(y_test)).issubset(np.unique(y_train)):
            raise ValueError("Test labels contain classes absent from the training partition.")
    else:
        encoder = LabelEncoder().fit(y_train.astype(str))
        y_train = encoder.transform(y_train.astype(str))
        try:
            y_test = encoder.transform(y_test.astype(str))
        except ValueError as exc:
            raise ValueError("Test labels contain classes absent from the training partition.") from exc
    if np.unique(y_train).size < 2:
        raise ValueError("The training feature CSV must contain at least two classes.")
    data = Data()
    data.set_train_test(
        X_train=X_train.to_numpy(dtype=np.float64), y_train=y_train,
        X_test=X_test.to_numpy(dtype=np.float64), y_test=y_test,
    )
    return data


def resolve_miafex_csv(args: argparse.Namespace) -> Dict[str, str]:
    """Prepare/reuse one dataset using the existing training and extraction code."""
    csv_paths = miafex_feature_paths(args)
    missing = [path for path in csv_paths.values() if not os.path.isfile(path)]
    if args.pipeline_mode == "feature_selection":
        if missing:
            raise FileNotFoundError(
                f"Prepared train/test feature CSVs missing: {', '.join(missing)}. "
                "Generate it with --pipeline-mode extract or full first."
            )
        print(f"[features] {args.dataset_name}: reusing {csv_paths} (feature_selection only)")
        return csv_paths

    checkpoint_path = os.path.join(args.miafex_output, "miafex_checkpoint.pth")
    run_training = args.train_miafex == "yes" or (args.train_miafex == "auto" and not os.path.isfile(checkpoint_path))
    run_extraction = args.extract_miafex == "yes" or (args.extract_miafex == "auto" and bool(missing))
    if not run_extraction and missing:
        raise FileNotFoundError(f"Prepared feature CSVs missing with --extract-miafex no: {', '.join(missing)}")
    if run_extraction and not run_training and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing with --train-miafex no: {os.path.abspath(checkpoint_path)}")
    if run_training or run_extraction:
        if MIAFEX_IMPORT_ERROR is not None:
            raise ImportError("MIAFEx training/extraction dependencies could not be imported.") from MIAFEX_IMPORT_ERROR
        if not valid_miafex_dataset(args.dataset_root):
            raise ValueError(f"Invalid MIAFEx train/test class folders: {args.dataset_root}")

    if run_training:
        print(f"[checkpoint] {args.dataset_name}: training for {args.miafex_epochs} neural-network epochs")
        checkpoint_path = train_miafex(
            train_root=os.path.join(args.dataset_root, "train"),
            output_dir=args.miafex_output,
            num_classes=None,
            num_epochs=args.miafex_epochs,
            batch_size=args.miafex_batch_size,
            learning_rate=args.miafex_learning_rate,
            device=args.miafex_device,
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"MIAFEx training did not produce a checkpoint: {checkpoint_path}")
    elif os.path.isfile(checkpoint_path):
        print(f"[checkpoint] {args.dataset_name}: reusing {checkpoint_path}")

    if run_extraction:
        # Stage each split separately: the existing extractor writes fixed names.
        # Publish only after both extractions and their schema checks succeed.
        with ExitStack() as stack:
            staged = {}
            for split, csv_path in csv_paths.items():
                output_dir = os.path.dirname(csv_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix=f".extract-{split}-", dir=output_dir))
                print(f"[features] {args.dataset_name}: extracting {split}/ images to {csv_path}")
                staged[split] = extract_miafex_features(
                    data_dir=os.path.join(args.dataset_root, split),
                    checkpoint_path=checkpoint_path,
                    output_dir=temporary,
                    batch_size=args.miafex_batch_size,
                    device=args.miafex_device,
                    run_ml_baselines=False,
                )
            load_miafex_feature_data(staged)
            mappings = []
            for path in staged.values():
                mapping_path = os.path.join(os.path.dirname(path), "class_to_idx.json")
                if os.path.isfile(mapping_path):
                    with open(mapping_path, encoding="utf-8") as stream:
                        mappings.append(json.load(stream))
            if mappings and (len(mappings) != 2 or mappings[0] != mappings[1]):
                raise ValueError("Train/test extraction class mappings do not match.")
            for split, path in staged.items():
                for original, filename in (("miafex_features.npy", f"{split}_features.npy"),
                                           ("class_to_idx.json", f"{split}_class_to_idx.json")):
                    artifact = os.path.join(os.path.dirname(path), original)
                    if os.path.isfile(artifact):
                        os.replace(artifact, os.path.join(os.path.dirname(csv_paths[split]), filename))
                os.replace(path, csv_paths[split])
    else:
        print(f"[features] {args.dataset_name}: reusing {csv_paths}")

    return csv_paths


class SafeOriginalDMOA(OriginalDMOA):
    """OriginalDMOA with numerically safe updates for binary feature-selection spaces."""

    def evolve(self, epoch):
        cf = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / (mean_cost + self.EPSILON))

        for idx in range(0, self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[alpha].solution + phi * (self.pop[alpha].solution - self.pop[k].solution)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        sm = np.zeros(self.pop_size)
        for idx in range(0, self.pop_size):
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[idx].solution + phi * (self.pop[idx].solution - self.pop[k].solution)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            current_fit = self.pop[idx].target.fitness
            trial_fit = agent.target.fitness
            denom = max(abs(trial_fit), abs(current_fit), self.EPSILON)
            sm[idx] = (trial_fit - current_fit) / denom
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        for idx in range(0, self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.generate_agent()
                self.C[idx] = 0

        new_tau = np.mean(sm)
        for idx in range(0, self.pop_size):
            m = np.full(self.problem.n_dims, sm[idx], dtype=float)
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > self.tau:
                new_pos = self.pop[idx].solution - cf * phi * self.generator.random() * (self.pop[idx].solution - m)
            else:
                new_pos = self.pop[idx].solution + cf * phi * self.generator.random() * (self.pop[idx].solution - m)
            self.tau = new_tau
            new_pos = self.correct_solution(new_pos)
            self.pop[idx] = self.generate_agent(new_pos)


def build_optimizer(name: str, args: argparse.Namespace):
    resolved_name = resolve_optimizer_name(name)
    resolved_upper = resolved_name.upper()
    if resolved_upper in {"DSA-DE", "DSADE"}:
        return DSADE(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if resolved_upper in {"DSADE_AWAD", "DSADE-AWAD"}:
        return DSADE_AWAD(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if resolved_upper in {"MACRO-DE", "MACRO_DE"}:
        return MaCRO_DE(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if resolved_upper == "DBO":
        return DBOOptimizer(epoch=args.epochs, pop_size=args.pop_size)
    if resolved_upper == "ORIGINALDMOA":
        return SafeOriginalDMOA(epoch=args.epochs, pop_size=args.pop_size)
    return resolved_name

def _legacy_cache_settings(args: argparse.Namespace) -> dict:
    """Original signature fields, retained to locate validated pre-migration caches."""
    payload = {
        "dataset_source": args.dataset_source,
        "dataset_name": args.dataset_name,
        "features_csv": args.features_csv,
        "dataset_root": args.dataset_root,
        "train_miafex": args.train_miafex,
        "extract_miafex": args.extract_miafex,
        "miafex_output": args.miafex_output,
        "miafex_epochs": int(args.miafex_epochs),
        "miafex_batch_size": int(args.miafex_batch_size),
        "miafex_learning_rate": float(args.miafex_learning_rate),
        "optimizers": list(args.optimizers),
        "transfer_functions": list(args.transfer_functions),
        "runs": int(args.runs),
        "epochs": int(args.epochs),
        "pop_size": int(args.pop_size),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "seed_base": int(args.seed_base),
        "obj_name": "AS",
        "fitness_mode": "minimize_metric_loss_plus_feature_ratio_v1",
        "dsade_beta_min": float(args.dsade_beta_min),
        "dsade_beta_max": float(args.dsade_beta_max),
        "dsade_pcr": float(args.dsade_pcr),
        "dsade_mahal_q": float(args.dsade_mahal_q),
    }
    if args.dataset_source == "miafex":
        # Do not resume results produced by the obsolete single-CSV resplit flow.
        payload["miafex_partition_mode"] = "prepared_train_test_v1"
        payload["train_features_csv"] = args.train_features_csv
        payload["test_features_csv"] = args.test_features_csv
    return payload


def _hash_cache_settings(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def build_cache_signature(args: argparse.Namespace) -> str:
    payload = _legacy_cache_settings(args)
    # Scheduling, output locations and stage switches do not identify the science.
    # Keep RUNS and all scientific settings, including the prepared CSV paths.
    for key in ("train_miafex", "extract_miafex", "miafex_output", "dataset_root", "features_csv"):
        payload.pop(key)
    if args.dataset_source == "mafese":
        payload["dataset_suite"] = args.dataset_suite
    return _hash_cache_settings(payload)


def legacy_cache_signatures(args: argparse.Namespace) -> List[str]:
    # These stage switches were execution-only even in the old format. Keep
    # lookup bounded to hashes we can prove compatible, never arbitrary pickles.
    payload = _legacy_cache_settings(args)
    signatures = []
    for train in dict.fromkeys((args.train_miafex, "auto", "yes", "no")):
        for extract in dict.fromkeys((args.extract_miafex, "auto", "yes", "no")):
            signatures.append(_hash_cache_settings(dict(payload, train_miafex=train, extract_miafex=extract)))
    return signatures

def build_alg_label(method: str, transfer_function: str, classifier: str, show_tf: bool, show_cls: bool) -> str:
    parts = [method.upper()]
    if show_tf:
        parts.append(str(transfer_function).upper())
    if show_cls:
        parts.append(classifier.upper())
    return "_".join(parts)

def muted_color_palette(n: int) -> np.ndarray:
    cmap = plt.get_cmap("turbo", max(n, 1))
    colors = cmap(np.arange(max(n, 1)))[:, :3]
    colors = 0.8 * colors + 0.2
    return np.clip(colors, 0.0, 1.0)


class RobustClassificationFeatureSelectionProblem(FeatureSelectionProblem):
    """Classification objective that tolerates validation folds missing classes."""

    def __init__(self, bounds=None, minmax=None, data=None, estimator=None, metric_class=None,
                 obj_name=None, obj_paras=None, fit_weights=(0.9, 0.1), fit_sign=None, **kwargs):
        super().__init__(
            bounds=bounds,
            minmax="min",
            data=data,
            estimator=estimator,
            metric_class=metric_class,
            obj_name=obj_name,
            obj_paras=obj_paras,
            fit_weights=fit_weights,
            fit_sign=1,
            **kwargs,
        )

    def obj_func(self, solution):
        x = self.decode_solution(solution)["my_var"]
        cols = np.flatnonzero(x)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        obj = self._score(self.data.y_test, y_valid_pred)
        feature_ratio = np.sum(x) / self.n_dims
        fitness = self.fit_weights[0] * (1.0 - obj) + self.fit_weights[1] * feature_ratio
        return [fitness, obj, np.sum(x)]

    def _score(self, y_true, y_pred) -> float:
        metric = str(self.obj_name).upper()
        average = (self.obj_paras or {}).get("average", "macro")
        labels = np.unique(np.concatenate((np.asarray(self.data.y_train), np.asarray(y_true), np.asarray(y_pred))))

        if metric == "AS":
            return float(accuracy_score(y_true, y_pred))
        if metric == "PS":
            return float(precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0))
        if metric == "RS":
            return float(recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0))
        if metric == "F1S":
            return float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0))

        evaluator = self.metric_class(y_true, y_pred)
        try:
            return float(evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name])
        except ValueError as err:
            if "Invalid y_pred" not in str(err):
                raise
            paras = dict(self.obj_paras or {})
            paras["labels"] = labels
            return float(evaluator.get_metric_by_name(self.obj_name, paras=paras)[self.obj_name])


def run_single(data: Data, estimator: str, optimizer_name: str, tf: str, args: argparse.Namespace, seed: int):
    logging.disable(logging.INFO)
    np.random.seed(seed)
    optimizer = build_optimizer(optimizer_name, args)
    selector_kwargs = dict(
        problem="classification",
        estimator=estimator,
        optimizer=optimizer,
        optimizer_paras=({"epoch": args.epochs, "pop_size": args.pop_size} if isinstance(optimizer, str) else None),
        obj_name="AS",
    )
    init_params = inspect.signature(MhaSelector.__init__).parameters
    if "transfer_func" in init_params:
        selector_kwargs["transfer_func"] = tf

    selector = MhaSelector(**selector_kwargs)

    t0 = time.time()
    fit_params = inspect.signature(selector.fit).parameters
    fit_kwargs = {}
    if "transfer_func" in fit_params:
        fit_kwargs["transfer_func"] = tf
    if "verbose" in fit_params:
        fit_kwargs["verbose"] = False
    if "fs_problem" in fit_params:
        fit_kwargs["fs_problem"] = RobustClassificationFeatureSelectionProblem
    # MAFESE's internal fitness validation uses only these training rows.
    # The prepared test partition is supplied only to final evaluation below.
    selector.fit(data.X_train, data.y_train, **fit_kwargs)
    runtime = time.time() - t0

    fit_curve = np.array(selector.optimizer.history.list_global_best_fit, dtype=float)
    fit_final = float(fit_curve[-1]) if fit_curve.size else np.nan

    selected = selector.transform(data.X_train)
    n_features = int(selected.shape[1])

    try:
        metrics = selector.evaluate(estimator=selector.estimator, data=data, metrics=["AS", "PS", "RS", "F1S"])
        as_test = float(metrics.get("AS_test", np.nan))
        ps_test = float(metrics.get("PS_test", np.nan))
        rs_test = float(metrics.get("RS_test", np.nan))
        f1_test = float(metrics.get("F1S_test", np.nan))
    except ValueError as err:
        # Permetrics can fail when y_pred contains labels absent in y_test.
        if "Invalid y_pred" not in str(err):
            raise
        X_train_sel = selector.transform(data.X_train)
        X_test_sel = selector.transform(data.X_test)
        if isinstance(selector.estimator, str):
            est = get_general_estimator("classification", selector.estimator)
        else:
            est = clone(selector.estimator)
        est.fit(X_train_sel, data.y_train)
        y_pred = est.predict(X_test_sel)
        labels = np.unique(np.concatenate((np.asarray(data.y_test), np.asarray(y_pred))))
        as_test = float(accuracy_score(data.y_test, y_pred))
        ps_test = float(precision_score(data.y_test, y_pred, labels=labels, average="macro", zero_division=0))
        rs_test = float(recall_score(data.y_test, y_pred, labels=labels, average="macro", zero_division=0))
        f1_test = float(f1_score(data.y_test, y_pred, labels=labels, average="macro", zero_division=0))

    return {
        "as_test": 100.0 * as_test,
        "ps_test": ps_test,
        "rs_test": rs_test,
        "f1_test": f1_test,
        "fit_final": fit_final,
        "n_features": n_features,
        "runtime": runtime,
        "curve": fit_curve,
    }


def run_single_parallel_task(task: dict):
    data_split = task["data_split"]
    data = Data()
    data.set_train_test(
        X_train=data_split["X_train"],
        y_train=data_split["y_train"],
        X_test=data_split["X_test"],
        y_test=data_split["y_test"],
    )
    out = run_single(
        data,
        task["estimator"],
        task["method"],
        task["tf"],
        task["args"],
        task["seed"],
    )
    return task["run"], out


def execute_pending_runs(
    data: Data,
    estimator: str,
    method: str,
    tf: str,
    args: argparse.Namespace,
    pending_runs: List[int],
    on_run_complete=None,
    dataset_name: str = "",
    completed_runs: int = 0,
    started_at: float | None = None,
):
    if args.parallel != "yes" or len(pending_runs) <= 1:
        completed = []
        for run in pending_runs:
            item = (run, run_single(data, estimator, method, tf, args, args.seed_base + run))
            if on_run_complete is not None:
                on_run_complete(*item)
            completed.append(item)
        return completed

    data_split = {
        "X_train": data.X_train,
        "y_train": data.y_train,
        "X_test": data.X_test,
        "y_test": data.y_test,
    }
    max_workers = min(args.n_workers, len(pending_runs))
    tasks = [
        {
            "run": run,
            "data_split": data_split,
            "estimator": estimator,
            "method": method,
            "tf": tf,
            "args": args,
            "seed": args.seed_base + run,
        }
        for run in pending_runs
    ]
    completed = []
    started_at = time.monotonic() if started_at is None else started_at
    next_heartbeat = time.monotonic() + PROGRESS_INTERVAL_SECONDS
    progress_label = build_alg_label(
        optimizer_display_label(method), tf, estimator, len(args.transfer_functions) > 1, True,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_parallel_task, task) for task in tasks}
        while futures:
            ready, _ = wait(
                futures, timeout=max(0.0, next_heartbeat - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )
            failure = None
            for future in ready:
                futures.remove(future)
                try:
                    item = future.result()
                except Exception as exc:
                    failure = exc
                    continue
                if on_run_complete is not None:
                    on_run_complete(*item)
                completed.append(item)
            # Save successful completions in this batch before propagating a worker failure.
            if failure is not None:
                raise failure
            now = time.monotonic()
            if futures and now >= next_heartbeat:
                # ProcessPoolExecutor marks prefetched work as running too; cap by worker slots.
                active = min(max_workers, len(futures))
                print(
                    f"[{time.strftime('%H:%M:%S')}] PROGRESS | {dataset_name} | {progress_label} | "
                    f"completed={completed_runs + len(completed)}/{args.runs} | active={active} | "
                    f"pending={len(futures) - active} | elapsed={format_elapsed(now - started_at)} | status=running",
                    flush=True,
                )
                next_heartbeat = now + PROGRESS_INTERVAL_SECONDS
    return sorted(completed, key=lambda item: item[0])


def format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def format_run_ids(run_ids: List[int]) -> str:
    """Display zero-based cache IDs as compact, one-based run ranges."""
    ranges = []
    for run in sorted(run_ids):
        if ranges and run == ranges[-1][1] + 1:
            ranges[-1][1] = run
        else:
            ranges.append([run, run])
    return ",".join(str(start + 1) if start == end else f"{start + 1}..{end + 1}"
                    for start, end in ranges) or "none"


def completed_run_ids(payload: dict) -> List[int]:
    """Legacy checkpoints contain a contiguous prefix; new ones identify each row."""
    count = len(payload.get("AccRuns", []))
    run_ids = list(payload.get("CompletedRunIDs", range(count)))
    if (len(run_ids) != count or any(not isinstance(run, (int, np.integer)) or run < 0 for run in run_ids)
            or len(set(run_ids)) != len(run_ids)):
        raise ValueError("Invalid checkpoint: CompletedRunIDs must uniquely identify every result row.")
    return run_ids


def pad_mean_curves(curves: List[np.ndarray], target_len: int) -> np.ndarray:
    if not curves:
        return np.array([])
    mat = np.full((len(curves), target_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        c = np.asarray(curve, dtype=float).ravel()
        ln = min(target_len, c.size)
        mat[i, :ln] = c[:ln]
    return np.nanmean(mat, axis=0)

def build_label_payload(
    estimator: str,
    acc_runs: List[float],
    ps_runs: List[float],
    rs_runs: List[float],
    f1_runs: List[float],
    fit_runs: List[float],
    feat_runs: List[float],
    time_runs: List[float],
    curves: List[np.ndarray],
    epochs: int,
    completed_run_ids: List[int] | None = None,
):
    if completed_run_ids is not None:
        # Completion order must not change the scientific aggregation or exported run order.
        order = np.argsort(completed_run_ids)
        acc_runs, ps_runs, rs_runs, f1_runs, fit_runs, feat_runs, time_runs, curves = (
            [values[i] for i in order]
            for values in (acc_runs, ps_runs, rs_runs, f1_runs, fit_runs, feat_runs, time_runs, curves)
        )
    curve_mean = pad_mean_curves(curves, epochs)
    payload = {
        "Estimator": estimator,
        "AccMean": float(np.nanmean(acc_runs)),
        "F1Mean": float(np.nanmean(f1_runs)),
        "PSMean": float(np.nanmean(ps_runs)),
        "RSMean": float(np.nanmean(rs_runs)),
        "FitMean": float(np.nanmean(fit_runs)),
        "FeatMean": float(np.nanmean(feat_runs)),
        "TimeMean": float(np.nanmean(time_runs)),
        "AccBest": float(np.nanmax(acc_runs)),
        "AccRuns": np.array(acc_runs, dtype=float),
        "F1Runs": np.array(f1_runs, dtype=float),
        "PSRuns": np.array(ps_runs, dtype=float),
        "RSRuns": np.array(rs_runs, dtype=float),
        "FitRuns": np.array(fit_runs, dtype=float),
        "FeatRuns": np.array(feat_runs, dtype=float),
        "TimeRuns": np.array(time_runs, dtype=float),
        "Curve": curve_mean,
        "CurvesAll": curves,
        "CompletedRuns": len(acc_runs),
    }
    if completed_run_ids is not None:
        payload["CompletedRunIDs"] = sorted(completed_run_ids)
    return payload

def save_cache(path: str, payload: dict):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=os.path.dirname(os.path.abspath(path)),
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = stream.name
            pickle.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

def load_cache(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_cache_safe(path: str, label: str):
    if not os.path.exists(path):
        return None
    try:
        return load_cache(path)
    except Exception as exc:
        print(f"[cache-warning] No se pudo cargar {label} '{path}': {exc}")
        return None


def cache_files(paths: Paths, dataset_name: str, estimator: str, signature: str) -> tuple[str, str]:
    prefix = os.path.join(paths.cache_dir, f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{signature}")
    return f"{prefix}_results.pkl", f"{prefix}_progress.pkl"


def resolve_cached_payload(paths: Paths, args: argparse.Namespace, dataset_name: str,
                           estimator: str, signature: str, *, figures_only: bool = False):
    """Current EXP wins; import a compatible fallback into current EXP atomically.

    Source EXP is never created or written. None disables the source lookup;
    the current EXP ID as source performs no duplicate lookup. --no-reuse-cache
    retains the established current-progress resume but disables final/source
    reuse. Figures-only explicitly reads caches regardless of that switch.
    """
    destination = cache_files(paths, dataset_name, estimator, signature)
    signatures = list(dict.fromkeys([signature, *legacy_cache_signatures(args)]))
    use_final = args.reuse_cache or figures_only

    def read_best(location):
        for candidate_sig in signatures:
            final, progress = cache_files(location, dataset_name, estimator, candidate_sig)
            payloads = [load_cache_safe(final, "cache final")] if use_final else []
            payloads.append(load_cache_safe(progress, "checkpoint parcial"))
            payloads = [payload for payload in payloads if isinstance(payload, dict)]
            if payloads:
                return max(payloads, key=payload_completed_runs), candidate_sig
        return None, None

    def publish(payload):
        # Preserve label payloads, CompletedRunIDs and legacy contiguous prefixes.
        # Progress first also makes an interrupted import resumable.
        save_cache(destination[1], payload)
        save_cache(destination[0], payload)

    payload, found_sig = read_best(paths)
    if payload is not None:
        if found_sig != signature:
            publish(payload)
        print(f"CACHE HIT CURRENT | {paths.exp_tag} | {dataset_name} / {estimator}"
              + (" | migrated legacy signature" if found_sig != signature else ""), flush=True)
        return payload

    source_id = args.reuse_cache_from_exp_id
    if use_final and source_id is not None and source_id != args.exp_id:
        source = make_paths(args, exp_id=source_id, create=False)
        payload, found_sig = read_best(source)
        if payload is not None:
            publish(payload)
            print(f"CACHE IMPORTED | {source.exp_tag} -> {paths.exp_tag} | {dataset_name} / {estimator}"
                  + (" | migrated legacy signature" if found_sig != signature else ""), flush=True)
            return payload
        prefix = f"{source.exp_tag}_{dataset_name}_{estimator.lower()}_"
        if os.path.isdir(source.cache_dir):
            with os.scandir(source.cache_dir) as entries:
                has_source = any(entry.is_file() and entry.name.startswith(prefix)
                                 and entry.name.endswith(("_results.pkl", "_progress.pkl")) for entry in entries)
            if has_source:
                print(f"CACHE SOURCE INCOMPATIBLE | {source.exp_tag} | {dataset_name} / {estimator} | "
                      "no readable cache with a compatible scientific signature", flush=True)
    print(f"CACHE MISS | {paths.exp_tag} | {dataset_name} / {estimator}", flush=True)
    return None


def load_results_from_cache(paths: Paths, args: argparse.Namespace, dataset_names: List[str], cache_sig: str | Dict[str, str]) -> Dict[str, Dict]:
    results_struct = {}
    missing = []
    dataset_args = resolve_miafex_dataset_args(args) if args.dataset_source == "miafex" else {}
    for dataset_name in dataset_names:
        results_struct[dataset_name] = {}
        dataset_cache_sig = cache_sig[dataset_name] if isinstance(cache_sig, dict) else cache_sig
        for estimator in args.estimators:
            payload = resolve_cached_payload(
                paths, dataset_args.get(dataset_name, args), dataset_name, estimator, dataset_cache_sig,
                figures_only=True,
            )
            if payload is None:
                missing.append(f"{dataset_name}/{estimator}")
                continue
            results_struct[dataset_name].update(payload)

    if missing:
        raise FileNotFoundError(
            "No se encontraron caches para: "
            + ", ".join(missing)
            + ". Ejecuta el experimento completo o revisa que los parametros coincidan con el cache existente."
        )
    return results_struct

def payload_completed_runs(payload: dict) -> int:
    total = 0
    for row in payload.values():
        if not isinstance(row, dict):
            continue
        total += (len(row["CompletedRunIDs"]) if "CompletedRunIDs" in row
                  else int(row.get("CompletedRuns", len(row.get("AccRuns", [])))))
    return total


def parse_result_label(label: str, args: argparse.Namespace) -> dict:
    label_upper = str(label).upper()
    optimizer_tokens = []
    for opt in args.optimizers:
        optimizer_tokens.append(str(opt))
        optimizer_tokens.append(optimizer_acronym(opt))
    ordered_opts = sorted(
        list(dict.fromkeys(optimizer_tokens + ["DSA-DE", "DSADE", "DSADE_AWAD", "DSADE-AWAD", "MaCRO-DE", "MACRO-DE", "DBO"])),
        key=len,
        reverse=True,
    )
    method = next(
        (
            opt
            for opt in ordered_opts
            if label_upper == opt.upper() or label_upper.startswith(f"{opt.upper()}_")
        ),
        str(label),
    )
    rest = label_upper[len(method):].lstrip("_") if method != str(label) else ""

    estimator = ""
    for est in sorted([str(e) for e in args.estimators], key=len, reverse=True):
        est_upper = est.upper()
        if rest == est_upper:
            estimator = est.lower()
            rest = ""
            break
        suffix = f"_{est_upper}"
        if rest.endswith(suffix):
            estimator = est.lower()
            rest = rest[: -len(suffix)]
            break

    transfer_function = ""
    for tf in sorted(SUPPORTED_TRANSFER_FUNCTIONS, key=len, reverse=True):
        tf_upper = tf.upper()
        if rest == tf_upper or rest.startswith(f"{tf_upper}_") or f"_{tf_upper}" in rest:
            transfer_function = tf.lower()
            break

    return {"method": method, "transfer_function": transfer_function, "estimator": estimator}


def optimizer_display_label(name: str) -> str:
    return optimizer_acronym(name)

def optimizer_order_key(name: str) -> tuple:
    label = optimizer_display_label(name).upper()
    if label == "MACRO-DE":
        return (0, "")
    if label == "DSA-DE":
        return (1, "")
    if label in {"DSADE-AWAD", "DSADE_AWAD"}:
        return (2, "")
    return (3, label)

def is_dsade_method(name: str) -> bool:
    return str(name).upper() in {"MACRO-DE", "DSA-DE", "DSADE", "DSADE_AWAD", "DSADE-AWAD"}

def is_exact_dsade_method(name: str) -> bool:
    return str(name).upper() in {"DSA-DE", "DSADE"}

def prepare_plot_groups(df: pd.DataFrame, opt_order: List[str]) -> tuple[pd.DataFrame, List[str], Dict[str, str], Dict[str, str]]:
    if df.empty:
        return df.copy(), [], {}, {}

    plot_df = df.copy()
    if "FuncionTransferencia" not in plot_df.columns:
        plot_df["FuncionTransferencia"] = ""
    plot_df["FuncionTransferencia"] = plot_df["FuncionTransferencia"].fillna("").astype(str).str.lower()

    tf_counts = plot_df[plot_df["FuncionTransferencia"] != ""].groupby("Optimizador")["FuncionTransferencia"].nunique()
    variant_methods = set(tf_counts[tf_counts > 1].index)

    def make_group(row):
        opt = str(row["Optimizador"])
        tf = str(row["FuncionTransferencia"]).lower()
        return f"{opt}_{tf.upper()}" if opt in variant_methods and tf else opt

    plot_df["GrupoGrafica"] = plot_df.apply(make_group, axis=1)
    group_meta = (
        plot_df[["GrupoGrafica", "Optimizador", "FuncionTransferencia"]]
        .drop_duplicates()
        .set_index("GrupoGrafica")
        .to_dict("index")
    )

    method_order = list(dict.fromkeys([str(o) for o in opt_order] + [str(meta["Optimizador"]) for meta in group_meta.values()]))
    method_order = sorted([opt for opt in method_order if opt in {meta["Optimizador"] for meta in group_meta.values()}], key=optimizer_order_key)

    opts = []
    for opt in method_order:
        opt_groups = sorted(
            [g for g, meta in group_meta.items() if meta["Optimizador"] == opt],
            key=lambda g: (str(group_meta[g]["FuncionTransferencia"]), g),
        )
        opts.extend(opt_groups)
    opts.extend(sorted((g for g in group_meta if g not in set(opts)), key=lambda g: optimizer_order_key(group_meta[g]["Optimizador"])))

    colors = muted_color_palette(len(opts))
    color_map = {}
    label_map = {}
    for i, group in enumerate(opts):
        meta = group_meta[group]
        method = meta["Optimizador"]
        tf = meta["FuncionTransferencia"]
        color_map[group] = colors[i]
        base_label = optimizer_display_label(method)
        label_map[group] = f"{base_label} {tf.upper()}" if tf and method in variant_methods else base_label

    return plot_df, opts, color_map, label_map

def export_global_excel(results_struct: Dict[str, Dict], dataset_names: List[str], out_path: str):
    all_labels = sorted(set().union(*[set(v.keys()) for v in results_struct.values()])) if results_struct else []
    if not all_labels:
        return []
    idx = pd.Index(dataset_names, name="Dataset")
    acc = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    ps = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    rs = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    f1 = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    fit = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    feat = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    tim = pd.DataFrame(np.nan, index=idx, columns=all_labels)

    for ds, alg_data in results_struct.items():
        for lbl, row in alg_data.items():
            acc.loc[ds, lbl] = row.get("AccMean", np.nan)
            ps.loc[ds, lbl] = row.get("PSMean", np.nan)
            rs.loc[ds, lbl] = row.get("RSMean", np.nan)
            f1.loc[ds, lbl] = row.get("F1Mean", np.nan)
            fit.loc[ds, lbl] = row.get("FitMean", np.nan)
            feat.loc[ds, lbl] = row.get("FeatMean", np.nan)
            tim.loc[ds, lbl] = row.get("TimeMean", np.nan)

    try:
        with pd.ExcelWriter(out_path) as writer:
            acc.to_excel(writer, sheet_name="Accuracy")
            ps.to_excel(writer, sheet_name="Precision")
            rs.to_excel(writer, sheet_name="Recall")
            f1.to_excel(writer, sheet_name="F1Score")
            fit.to_excel(writer, sheet_name="Fitness")
            feat.to_excel(writer, sheet_name="Features")
            tim.to_excel(writer, sheet_name="Time")
        return [out_path]
    except ModuleNotFoundError:
        base = os.path.splitext(out_path)[0]
        paths = [
            f"{base}_Accuracy.csv",
            f"{base}_Precision.csv",
            f"{base}_Recall.csv",
            f"{base}_F1Score.csv",
            f"{base}_Fitness.csv",
            f"{base}_Features.csv",
            f"{base}_Time.csv",
        ]
        acc.to_csv(paths[0])
        ps.to_csv(paths[1])
        rs.to_csv(paths[2])
        f1.to_csv(paths[3])
        fit.to_csv(paths[4])
        feat.to_csv(paths[5])
        tim.to_csv(paths[6])
        return paths

def generate_summary_dataframe(results_struct: Dict[str, Dict], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset_name, alg_data in results_struct.items():
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            method = parsed["method"]
            estimator = parsed["estimator"] or None
            estimator = estimator or (row.get("Estimator") if isinstance(row, dict) else None) or (
                args.estimators[0] if len(args.estimators) == 1 else ""
            )
            rows.append(
                {
                    "Archivo": dataset_name,
                    "Estimador": estimator,
                    "Optimizador": method,
                    "FuncionTransferencia": parsed["transfer_function"],
                    "Configuracion": label,
                    "F1_test": float(row.get("F1Mean", np.nan)),
                    "AS_test": float(row.get("AccMean", np.nan)) / 100.0,
                    "PS_test": float(row.get("PSMean", np.nan)),
                    "RS_test": float(row.get("RSMean", np.nan)),
                    "N_Features_Selected": float(row.get("FeatMean", np.nan)),
                    "Runtime": float(row.get("TimeMean", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _plot_legend_patches(opts: List[str], color_map: Dict[str, str], label_map: Dict[str, str]) -> List[mpatches.Patch]:
    return [mpatches.Patch(color=color_map.get(o, "#888"), label=label_map.get(o, o)) for o in opts]


def _force_white_background(fig):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for ax in fig.get_axes():
        ax.set_facecolor("white")


def _save_chart(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    _force_white_background(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_classifier_metric_grid_chart(df: pd.DataFrame, out_dir: str, opt_order: List[str]):
    if df.empty:
        return None

    plot_df = df.copy()
    plot_df["Estimador"] = plot_df["Estimador"].astype(str).str.lower()
    plot_df, opts, color_map, label_map = prepare_plot_groups(plot_df, opt_order)
    if not opts:
        return None
    method_by_group = plot_df.drop_duplicates("GrupoGrafica").set_index("GrupoGrafica")["Optimizador"].to_dict()

    metric_cols = ["AS_test", "PS_test", "RS_test", "F1_test"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_header_styles = [
        ("#d8e8f3", "#b8d3e6"),
        ("#d2efee", "#abd9d7"),
        ("#f7efd8", "#ead9ad"),
        ("#f9d5d9", "#edaeb8"),
    ]

    present_estimators = [str(e).lower() for e in plot_df["Estimador"].dropna().unique()]
    required_estimators = [e for e in DEFAULT_ESTIMATORS if e in SUPPORTED_ESTIMATORS]
    estimators = [e for e in SUPPORTED_ESTIMATORS if e in set(required_estimators + present_estimators)]
    estimators += sorted(e for e in present_estimators if e not in set(estimators))
    if not estimators:
        return None

    grouped = plot_df.groupby(["Estimador", "GrupoGrafica"])[metric_cols].mean()
    n_rows = len(estimators)
    n_cols = len(metric_cols)
    fig_w = max(16.0, 4.2 * n_cols)
    fig_h = max(4.5, 2.75 * n_rows + 2.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, facecolor="#f7f9fc")
    x = np.arange(len(opts))
    colors = [color_map.get(opt, "#888888") for opt in opts]
    xlabels = [label_map.get(opt, opt) for opt in opts]

    for r, estimator in enumerate(estimators):
        for c, (metric, metric_label) in enumerate(zip(metric_cols, metric_labels)):
            ax = axes[r, c]
            ax.set_facecolor("#f3f6fa")
            vals = [
                float(grouped.loc[(estimator, opt), metric])
                if (estimator, opt) in grouped.index
                else np.nan
                for opt in opts
            ]
            edges = ["black" if is_dsade_method(method_by_group.get(opt)) else "none" for opt in opts]
            widths = [1.8 if is_dsade_method(method_by_group.get(opt)) else 0.0 for opt in opts]
            bars = ax.bar(x, vals, color=colors, edgecolor=edges, linewidth=widths, width=0.68)

            mean_val = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            if np.isfinite(mean_val):
                ax.axhline(mean_val, color="#d76c6c", linestyle="--", linewidth=0.9, alpha=0.8)

            for bar, value in zip(bars, vals):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.006,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                    color="#333333",
                )

            if not np.isfinite(vals).any():
                ax.text(
                    0.5,
                    0.5,
                    "Sin datos",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#777777",
                )

            ax.set_ylim(0.0, 1.10)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.24, linewidth=0.8)
            ax.set_axisbelow(True)

            if c == 0:
                ax.set_ylabel(estimator.upper(), fontsize=12, fontweight="bold", color="#19365f")
            if r == 0:
                face, edge = metric_header_styles[c]
                ax.set_title(
                    metric_label,
                    fontsize=12,
                    fontweight="bold",
                    color="#19365f",
                    pad=12,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=face, edgecolor=edge),
                )

    legend = _plot_legend_patches(opts, color_map, label_map)
    if any(is_exact_dsade_method(method_by_group.get(opt)) for opt in opts):
        legend.append(mpatches.Patch(facecolor="#333333", edgecolor="black", label="DSA-DE: borde negro"))
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6), fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    filename = "09_resultados_clasificador_metrica_todos_datasets.png"
    _save_chart(fig, out_dir, filename)
    return filename


def build_run_level_dataframe(results_struct: Dict[str, Dict], args: argparse.Namespace, estimator_filter: str = "knn") -> pd.DataFrame:
    rows = []
    for dataset_name, alg_data in results_struct.items():
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            estimator = parsed["estimator"] or row.get("Estimator", "")
            if str(estimator).lower() != estimator_filter.lower():
                continue
            runs_by_metric = {
                "AS_test": np.asarray(row.get("AccRuns", []), dtype=float) / 100.0,
                "F1_test": np.asarray(row.get("F1Runs", []), dtype=float),
                "PS_test": np.asarray(row.get("PSRuns", []), dtype=float),
                "RS_test": np.asarray(row.get("RSRuns", []), dtype=float),
                "N_Features_Selected": np.asarray(row.get("FeatRuns", []), dtype=float),
                "Runtime": np.asarray(row.get("TimeRuns", []), dtype=float),
            }
            n_runs = max((values.size for values in runs_by_metric.values()), default=0)
            for run_idx in range(n_runs):
                out = {
                    "Archivo": dataset_name,
                    "Estimador": estimator_filter.lower(),
                    "Optimizador": parsed["method"],
                    "FuncionTransferencia": parsed["transfer_function"],
                    "Configuracion": label,
                    "Run": run_idx + 1,
                }
                for metric, values in runs_by_metric.items():
                    out[metric] = float(values[run_idx]) if run_idx < values.size else np.nan
                rows.append(out)
    return pd.DataFrame(rows)


def build_curve_dataframe(results_struct: Dict[str, Dict], args: argparse.Namespace, estimator_filter: str = "svm") -> pd.DataFrame:
    rows = []
    for dataset_name, alg_data in results_struct.items():
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            estimator = parsed["estimator"] or row.get("Estimator", "")
            if str(estimator).lower() != estimator_filter.lower():
                continue
            rows.append(
                {
                    "Archivo": dataset_name,
                    "Estimador": estimator_filter.lower(),
                    "Optimizador": parsed["method"],
                    "FuncionTransferencia": parsed["transfer_function"],
                    "Configuracion": label,
                    "Curve": np.asarray(row.get("Curve", []), dtype=float),
                }
            )
    return pd.DataFrame(rows)


def _grid_shape(n_items: int) -> tuple[int, int]:
    n_cols = min(4, max(1, int(np.ceil(np.sqrt(max(1, n_items))))))
    n_rows = int(np.ceil(max(1, n_items) / n_cols))
    return n_rows, n_cols


def generate_seven_global_charts(
    df: pd.DataFrame,
    results_struct: Dict[str, Dict],
    out_dir: str,
    opt_order: List[str],
    args: argparse.Namespace,
    estimator_filter: str = "svm", # Change here for knn
):
    if df.empty:
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    chart1 = generate_classifier_metric_grid_chart(df, out_dir, opt_order)
    if chart1:
        new_chart1 = "01_resultados_clasificador_todos_datasets.png"
        os.replace(os.path.join(out_dir, chart1), os.path.join(out_dir, new_chart1))
        saved.append(new_chart1)

    knn_df = df[df["Estimador"].astype(str).str.lower() == estimator_filter.lower()].copy()
    if knn_df.empty:
        return saved
    plot_df, opts, color_map, label_map = prepare_plot_groups(knn_df, opt_order)
    if not opts:
        return saved
    method_by_group = plot_df.drop_duplicates("GrupoGrafica").set_index("GrupoGrafica")["Optimizador"].to_dict()
    datasets = sorted(plot_df["Archivo"].dropna().unique())
    n_rows, n_cols = _grid_shape(len(datasets))

    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "Feat.\nEfficiency"]
    angles = [n / 5.0 * 2 * np.pi for n in range(5)]
    angles += angles[:1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.8 * n_rows),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = plot_df[plot_df["Archivo"] == dataset]
        medias = sub.groupby("GrupoGrafica")[["AS_test", "PS_test", "RS_test", "F1_test", "N_Features_Selected"]].mean()
        max_feat = max(float(medias["N_Features_Selected"].max()), 1.0)
        for opt in opts:
            if opt not in medias.index:
                continue
            row = medias.loc[opt]
            vals = [row["AS_test"], row["PS_test"], row["RS_test"], row["F1_test"], 1 - row["N_Features_Selected"] / max_feat]
            vals += vals[:1]
            is_dsade = is_dsade_method(method_by_group.get(opt))
            is_macro = method_by_group.get(opt) == "MaCRO-DE"

            ax.plot(
                angles,
                vals,
                color=color_map.get(opt, "#888"),
                linewidth=4.0 if is_macro else (2.4 if is_dsade else 1.1),
                linestyle="-" if is_macro else ("-" if is_dsade else "--"),
                zorder=10 if is_macro else 2
            )

            ax.fill(
                angles,
                vals,
                color=color_map.get(opt, "#888"),
                alpha=0.20 if is_macro else (0.12 if is_dsade else 0.04)
            )
            # ax.plot(angles, vals, color=color_map.get(opt, "#888"), linewidth=2.4 if is_dsade else 1.1, linestyle="-" if is_dsade else "--")
            # ax.fill(angles, vals, color=color_map.get(opt, "#888"), alpha=0.12 if is_dsade else 0.04)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(dataset, fontsize=11, fontweight="bold", pad=14)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.legend(handles=_plot_legend_patches(opts, color_map, label_map), loc="lower center", ncol=min(len(opts), 6), fontsize=9)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    _save_chart(fig, out_dir, "02_radar_por_dataset_knn.png")
    saved.append("02_radar_por_dataset_knn.png")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax1 = axes[idx // n_cols, idx % n_cols]
        ax2 = ax1.twinx()
        sub = plot_df[plot_df["Archivo"] == dataset].groupby("GrupoGrafica")[["N_Features_Selected", "Runtime"]].mean()
        x = np.arange(len(opts))
        feat_vals = [sub.loc[o, "N_Features_Selected"] if o in sub.index else np.nan for o in opts]
        rt_vals = [sub.loc[o, "Runtime"] if o in sub.index else np.nan for o in opts]
        colors = [color_map.get(o, "#888") for o in opts]
        ax1.bar(x - 0.18, feat_vals, 0.36, color=colors, alpha=0.85)
        ax2.bar(x + 0.18, rt_vals, 0.36, color=colors, alpha=0.40, hatch="///")
        ax1.set_xticks(x)
        ax1.set_xticklabels([label_map.get(o, o) for o in opts], rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("Features", fontsize=9)
        ax2.set_ylabel("Runtime (s)", fontsize=9)
        ax1.set_title(dataset, fontsize=11, fontweight="bold")
        ax1.grid(axis="y", alpha=0.25)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    _save_chart(fig, out_dir, "03_features_runtime_por_dataset_knn.png")
    saved.append("03_features_runtime_por_dataset_knn.png")

    run_df = build_run_level_dataframe(results_struct, args, estimator_filter)
    run_source = run_df if not run_df.empty else plot_df
    run_plot_df, run_opts, run_color_map, run_label_map = prepare_plot_groups(run_source, opt_order)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = run_plot_df[run_plot_df["Archivo"] == dataset]
        data_box = [sub[sub["GrupoGrafica"] == opt]["AS_test"].dropna().values for opt in run_opts]
        bp = ax.boxplot(data_box, patch_artist=True, widths=0.55, showmeans=True)
        for patch, opt in zip(bp["boxes"], run_opts):
            patch.set_facecolor(run_color_map.get(opt, "#888"))
            patch.set_alpha(0.60)
        ax.set_xticks(range(1, len(run_opts) + 1))
        ax.set_xticklabels([run_label_map.get(o, o) for o in run_opts], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Accuracy (test)", fontsize=9)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    _save_chart(fig, out_dir, "04_boxplot_accuracy_por_dataset_knn.png")
    saved.append("04_boxplot_accuracy_por_dataset_knn.png")

    curve_df = build_curve_dataframe(results_struct, args, estimator_filter)
    if curve_df.empty:
        curve_plot_df = pd.DataFrame()
        curve_opts, curve_color_map, curve_label_map = opts, color_map, label_map
    else:
        curve_plot_df, curve_opts, curve_color_map, curve_label_map = prepare_plot_groups(curve_df, opt_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.4 * n_rows), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = curve_plot_df[curve_plot_df["Archivo"] == dataset] if not curve_plot_df.empty else pd.DataFrame()
        plotted = False
        for opt in curve_opts:
            rows_opt = sub[sub["GrupoGrafica"] == opt] if not sub.empty else pd.DataFrame()
            if rows_opt.empty:
                continue
            curve = np.asarray(rows_opt.iloc[0]["Curve"], dtype=float)
            if curve.size == 0:
                continue
            is_dsade = is_dsade_method(rows_opt.iloc[0]["Optimizador"])
            is_macro = str(rows_opt.iloc[0]["Optimizador"]).upper() == "MACRO-DE"
            ax.plot(curve, color=curve_color_map.get(opt, "#888"), linewidth=2.4 if is_macro else (2.4 if is_dsade else 1.4), linestyle="-")
            #ax.plot(curve, color=curve_color_map.get(opt, "#888"), linewidth=2.4 if is_dsade else 1.4, linestyle="-" if is_dsade else "--")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "Sin curvas", transform=ax.transAxes, ha="center", va="center", color="#777")
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Fitness", fontsize=9)
        ax.grid(alpha=0.25)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.legend(handles=_plot_legend_patches(curve_opts, curve_color_map, curve_label_map), loc="lower center", ncol=min(len(curve_opts), 6), fontsize=9)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    _save_chart(fig, out_dir, "05_convergence_por_dataset_knn.png")
    saved.append("05_convergence_por_dataset_knn.png")

    pivot = plot_df.groupby(["GrupoGrafica", "Archivo"])["F1_test"].mean().unstack()
    mat = pivot.reindex(index=opts, columns=datasets).values
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(datasets) + 4), max(5, 0.45 * len(opts) + 2)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1-Score (test)", shrink=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right")
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels([label_map.get(o, o) for o in opts])
    # for tick, opt in zip(ax.get_yticklabels(), opts):
    #     if str(method_by_group.get(opt)).upper() == "MACRO-DE":
    #         tick.set_color("red")
    #         tick.set_fontweight("bold")
    macro_idx = next(
        (i for i, opt in enumerate(opts)
         if str(method_by_group.get(opt)).upper() == "MACRO-DE"),
        None,
    )

    if macro_idx is not None:
        rect = plt.Rectangle((-0.5, macro_idx - 0.5), len(datasets),1, fill=False, edgecolor="black", linewidth=2.5, zorder=100)
        ax.add_patch(rect)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Metaheuristics")
    for i in range(len(opts)):
        for j in range(len(datasets)):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="white" if value > 0.80 else "#222", fontsize=8)
    fig.tight_layout()
    _save_chart(fig, out_dir, "06_heatmap_f1_knn.png")
    saved.append("06_heatmap_f1_knn.png")

    data_violin = [run_plot_df[run_plot_df["GrupoGrafica"] == opt]["RS_test"].dropna().values for opt in run_opts]
    fig, ax = plt.subplots(figsize=(max(12, 0.85 * len(run_opts) + 5), 6.5))
    parts = ax.violinplot(data_violin, showmeans=False, showmedians=False, widths=0.78)
    for body, opt in zip(parts["bodies"], run_opts):
        body.set_facecolor(run_color_map.get(opt, "#888"))
        body.set_edgecolor(run_color_map.get(opt, "#888"))
        body.set_alpha(0.22)
    for i, (opt, values) in enumerate(zip(run_opts, data_violin), start=1):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(np.full(values.size, i) + jitter, values, color=run_color_map.get(opt, "#888"), edgecolor="white", linewidth=0.5, s=35, zorder=3)
        mean_val = float(np.nanmean(values))
        median_val = float(np.nanmedian(values))
        ax.scatter(i, mean_val, marker="D", color="black", edgecolor="white", linewidth=1.2, s=140, zorder=4)
        ax.hlines(median_val, i - 0.25, i + 0.25, colors="black", linestyles="--", linewidth=1.2)
        ax.text(i, mean_val + 0.018, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(range(1, len(run_opts) + 1))
    ax.set_xticklabels([run_label_map.get(o, o) for o in run_opts], rotation=35, ha="right")
    ax.set_ylabel("Recall (test)")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#555", label="Mean"),
            plt.Line2D([0], [0], color="#555", linestyle="--", label="Median"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", label="Value per dataset/run"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    fig.tight_layout()
    _save_chart(fig, out_dir, "07_violin_recall_knn.png")
    saved.append("07_violin_recall_knn.png")

    generate_global_accuracy_boxplot(
        run_plot_df,
        out_dir,
        opt_order
    )
    saved.append("08_global_accuracy_distribution.png")

    generate_global_features_runtime(
        plot_df,
        out_dir,
        opt_order
    )
    saved.append("09_global_features_runtime_tradeoff.png")
    return saved

def generate_global_accuracy_boxplot(df, out_dir, opt_order):

    plot_df, opts, color_map, label_map = prepare_plot_groups(df, opt_order)

    fig, ax = plt.subplots(
        figsize=(max(12, 0.8 * len(opts) + 5), 6)
    )

    data_box = [
        plot_df[plot_df["GrupoGrafica"] == opt]["AS_test"].values
        for opt in opts
    ]

    bp = ax.boxplot(
        data_box,
        patch_artist=True,
        widths=0.55,
        showmeans=True
    )

    for patch, opt in zip(bp["boxes"], opts):

        patch.set_facecolor(color_map.get(opt, "#888"))
        patch.set_alpha(0.70)

        if label_map.get(opt) == "MaCRO-DE":
            patch.set_edgecolor("black")
            patch.set_linewidth(3.0)

    for i, opt in enumerate(opts):

        vals = plot_df[
            plot_df["GrupoGrafica"] == opt
        ]["AS_test"]

        if len(vals) > 0:
            ax.text(
                i + 1,
                np.mean(vals) + 0.01,
                f"{np.mean(vals):.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold"
            )

    ax.set_ylabel("Accuracy (test)")
    ax.set_xlabel("Metaheuristics")
    ax.set_ylim(0.50, 1.05)

    ax.set_xticks(range(1, len(opts)+1))
    ax.set_xticklabels(
        [label_map[o] for o in opts],
        rotation=45,
        ha="right"
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    _save_chart(
        fig,
        out_dir,
        "08_global_accuracy_distribution.png"
    )

def generate_global_features_runtime(df, out_dir, opt_order):

    plot_df, opts, color_map, label_map = prepare_plot_groups(df, opt_order)

    feat_med = (
        plot_df.groupby("GrupoGrafica")
        ["N_Features_Selected"]
        .mean()
    )

    rt_med = (
        plot_df.groupby("GrupoGrafica")
        ["Runtime"]
        .mean()
    )

    feat_vals = [feat_med[o] for o in opts]
    rt_vals   = [rt_med[o] for o in opts]

    x = np.arange(len(opts))
    w = 0.38

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - w/2,
        feat_vals,
        w,
        alpha=0.85
    )

    bars2 = ax2.bar(
        x + w/2,
        rt_vals,
        w,
        alpha=0.45,
        hatch="///"
    )

    for bar, opt in zip(bars1, opts):

        bar.set_color(color_map.get(opt, "#888"))

        if label_map.get(opt) == "MaCRO-DE":
            bar.set_edgecolor("black")
            bar.set_linewidth(3)

    for bar, opt in zip(bars2, opts):

        bar.set_color(color_map.get(opt, "#888"))

        if label_map.get(opt) == "MaCRO-DE":
            bar.set_edgecolor("black")
            bar.set_linewidth(3)

    for i, v in enumerate(feat_vals):

        ax1.text(
            i - w/2,
            v + 0.2,
            f"{v:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    for i, v in enumerate(rt_vals):

        ax2.text(
            i + w/2,
            v + 0.5,
            f"{v:.1f}s",
            ha="center",
            fontsize=9
        )

    ax1.set_ylabel("Average selected features")
    ax2.set_ylabel("Average runtime (sec)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [label_map[o] for o in opts],
        rotation=45,
        ha="right"
    )

    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    _save_chart(
        fig,
        out_dir,
        "09_global_features_runtime_tradeoff.png"
    )

def regenerate_figures_from_cache(paths: Paths, args: argparse.Namespace, dataset_names: List[str], cache_sig: str | Dict[str, str]):
    results_struct = load_results_from_cache(paths, args, dataset_names, cache_sig)
    summary_df = generate_summary_dataframe(results_struct, args)
    summary_csv = os.path.join(paths.res_dir, f"RESUMEN_GRAFICAS_{paths.exp_tag}.csv")
    summary_df.to_csv(summary_csv, index=False)
    generated_charts = generate_seven_global_charts(
        summary_df,
        results_struct,
        paths.fig_dir,
        list(args.optimizers),
        args,
    )
    return summary_csv, generated_charts


def _summary_value(items: List[str], label_func=str) -> str:
    return ", ".join(label_func(item) for item in items)


def print_experiment_summary(args: argparse.Namespace, paths: Paths, dataset_names: List[str], cache_sig: str | Dict[str, str], miafex_csv_path: Dict[str, Dict[str, str]] | None):
    print("=" * 60)
    print(" MIAFEx + Metaheuristic Feature Selection Framework")
    print("=" * 60)
    print(f"{'Experiment':<18}: {paths.exp_tag}")
    print(f"{'Dataset source':<18}: {args.dataset_source}")
    print(f"{'Pipeline mode':<18}: {args.pipeline_mode}")
    if args.dataset_source == "mafese":
        print(f"{'Dataset suite':<18}: {args.dataset_suite} ({len(dataset_names)} datasets)")
        print(f"{'Datasets':<18}: {_summary_value(dataset_names)}")
    else:
        print(f"{'Datasets':<18}: {_summary_value(dataset_names)}")
        print(f"{'MIAFEx training':<18}: {args.train_miafex}")
        print(f"{'Feature extraction':<18}: {args.extract_miafex}")
        print(f"{'MIAFEx epochs':<18}: {args.miafex_epochs} (neural-network training)")
        for name, csv_paths in (miafex_csv_path or {}).items():
            for split, csv_path in csv_paths.items():
                print(f"{'Features CSV':<18}: {name}/{split} -> {csv_path}")
    print(f"{'Optimizers':<18}: {_summary_value(args.optimizers, optimizer_display_label)}")
    print(f"{'Classifiers':<18}: {_summary_value(args.estimators, lambda x: str(x).upper())}")
    print(f"{'Transfer functions':<18}: {_summary_value(args.transfer_functions, lambda x: str(x).upper())}")
    print(f"{'Runs':<18}: {args.runs}")
    print(f"{'FS iterations':<18}: {args.epochs} (metaheuristic)")
    print(f"{'Population':<18}: {args.pop_size}")
    print(f"{'Parallel':<18}: {args.parallel}")
    print(f"{'Worker limit':<18}: {args.n_workers}")
    print(f"{'Cache source EXP':<18}: {args.reuse_cache_from_exp_id}")
    print(f"{'Cache signature':<18}: {cache_sig}")
    print("=" * 60)


def main():
    started_at = time.monotonic()
    args = parse_args()
    logging.disable(logging.INFO)
    logging.getLogger("mealpy").setLevel(logging.WARNING)

    execution_config = resolve_execution_config(args)
    print_backend_report(execution_config)
    validate_execution_config(execution_config)
    args.miafex_device = execution_config.miafex_device

    if args.show_backends:
        return

    if args.list_miafex_datasets:
        print_miafex_datasets(discover_miafex_datasets(args.miafex_dataset_root), args.miafex_dataset_root)
        return
    if args.list_optimizers:
        print(list_available_optimizers())
        return

    if args.pipeline_mode == "extract":
        if args.dataset_source != "miafex":
            raise ValueError("--pipeline-mode extract requires --dataset-source miafex; MAFESE already supplies feature datasets.")
        if args.figures_only:
            raise ValueError("--figures-only cannot be combined with --pipeline-mode extract.")
    else:
        validate_selection_options(args)
        args.optimizers = resolve_optimizers(args)
        if args.runs < 1:
            raise ValueError("--runs debe ser >= 1")
        if args.n_workers < 1:
            raise ValueError("--n-workers debe ser >= 1")

    if args.dataset_source == "mafese":
        dataset_names = resolve_mafese_dataset_names(args)
        miafex_csv_path = None
        cache_sig = build_cache_signature(args)
    else:
        dataset_args = resolve_miafex_dataset_args(args)
        dataset_names = list(dataset_args)
        miafex_csv_path = {name: miafex_feature_paths(scoped) for name, scoped in dataset_args.items()}
        # Plot regeneration is cache-only and never prepares neural artifacts.
        if not args.figures_only:
            for name, scoped in dataset_args.items():
                miafex_csv_path[name] = resolve_miafex_csv(scoped)
        if args.pipeline_mode == "extract":
            print("Completed MIAFEx feature preparation; feature selection was not run.")
            for name, csv_path in miafex_csv_path.items():
                print(f"  {name}: {csv_path}")
            return
        cache_sig = {name: build_cache_signature(scoped) for name, scoped in dataset_args.items()}

    paths = make_paths(args)
    show_tf = len(args.transfer_functions) > 1
    show_cls = len(args.estimators) > 1

    print_experiment_summary(args, paths, dataset_names, cache_sig, miafex_csv_path)

    if args.figures_only:
        summary_csv, generated_charts = regenerate_figures_from_cache(paths, args, dataset_names, cache_sig)
        print("Completed figures-only.")
        print(f"Cache dir: {paths.cache_dir}")
        print(f"Figures dir: {paths.fig_dir}")
        print(f"Charts summary CSV: {summary_csv}")
        if generated_charts:
            print("Charts:")
            for name in generated_charts:
                print(f"  - {os.path.join(paths.fig_dir, name)}")
        return

    results_struct = {}
    for dataset_name in dataset_names:
        results_struct[dataset_name] = {}
        dataset_cache_sig = cache_sig[dataset_name] if isinstance(cache_sig, dict) else cache_sig
        if args.dataset_source == "mafese":
            mafese_data = get_dataset(dataset_name)
            if mafese_data is None:
                raise ValueError(
                    f"mafese no pudo cargar '{dataset_name}'. "
                    "Verifica que exista en la suite 'test14' de mafese."
                )
            X = np.asarray(mafese_data.X, dtype=np.float64)
            y = np.asarray(mafese_data.y).astype(np.int32)
            data = Data(X, y)
            try:
                data.split_train_test(test_size=args.test_size, random_state=args.random_state, stratify=y)
            except ValueError:
                data.split_train_test(test_size=args.test_size, random_state=args.random_state)
        else:
            data = load_miafex_feature_data(miafex_csv_path[dataset_name])

        for estimator in args.estimators:
            cache_file, progress_file = cache_files(paths, dataset_name, estimator, dataset_cache_sig)
            cls_payload = resolve_cached_payload(
                paths, dataset_args[dataset_name] if args.dataset_source == "miafex" else args,
                dataset_name, estimator, dataset_cache_sig,
            ) or {}
            for method in args.optimizers:
                for tf in args.transfer_functions:
                    label = build_alg_label(method, tf, estimator, show_tf, show_cls)
                    prev = cls_payload.get(label, {})
                    acc_runs = list(np.asarray(prev.get("AccRuns", []), dtype=float))
                    ps_runs = list(np.asarray(prev.get("PSRuns", []), dtype=float))
                    rs_runs = list(np.asarray(prev.get("RSRuns", []), dtype=float))
                    f1_runs = list(np.asarray(prev.get("F1Runs", []), dtype=float))
                    fit_runs = list(np.asarray(prev.get("FitRuns", []), dtype=float))
                    feat_runs = list(np.asarray(prev.get("FeatRuns", []), dtype=float))
                    time_runs = list(np.asarray(prev.get("TimeRuns", []), dtype=float))
                    curves = list(prev.get("CurvesAll", []))

                    run_ids = completed_run_ids(prev)
                    pending_runs = [run for run in range(args.runs) if run not in run_ids]
                    progress_label = build_alg_label(optimizer_display_label(method), tf, estimator, show_tf, True)
                    print(
                        f"[resume] {dataset_name} | {progress_label} | completed={len(run_ids)}/{args.runs} | "
                        f"recovered={format_run_ids(run_ids)} | missing={format_run_ids(pending_runs)}",
                        flush=True,
                    )
                    if not pending_runs:
                        print(f"Running {dataset_name} | {label} | runs={args.runs} (already complete)")
                        continue
                    print(f"Running {dataset_name} | {label} | runs={args.runs} (recovered {len(run_ids)})", flush=True)

                    def checkpoint_run(run, out):
                        acc_runs.append(out["as_test"])
                        ps_runs.append(out["ps_test"])
                        rs_runs.append(out["rs_test"])
                        f1_runs.append(out["f1_test"])
                        fit_runs.append(out["fit_final"])
                        feat_runs.append(out["n_features"])
                        time_runs.append(out["runtime"])
                        curves.append(out["curve"])
                        run_ids.append(run)

                        cls_payload[label] = build_label_payload(
                            estimator,
                            acc_runs,
                            ps_runs,
                            rs_runs,
                            f1_runs,
                            fit_runs,
                            feat_runs,
                            time_runs,
                            curves,
                            args.epochs,
                            completed_run_ids=run_ids,
                        )
                        save_cache(progress_file, cls_payload)
                        save_cache(cache_file, cls_payload)
                        print(
                            f"  {dataset_name} | {progress_label} | Run {run + 1:02d}/{args.runs} | "
                            f"Accuracy={out['as_test']:.2f}% | F1={out['f1_test']:.4f} | "
                            f"Fitness={out['fit_final']:.4f} | Features={out['n_features']} | "
                            f"run time={out['runtime']:.2f}s | total elapsed={format_elapsed(time.monotonic() - started_at)}",
                            flush=True,
                        )

                    if args.parallel == "yes" and len(pending_runs) > 1:
                        print(f"  Parallel: yes | workers={min(args.n_workers, len(pending_runs))}")
                        execute_pending_runs(
                            data,
                            estimator,
                            method,
                            tf,
                            args,
                            pending_runs,
                            on_run_complete=checkpoint_run,
                            dataset_name=dataset_name,
                            completed_runs=len(run_ids),
                            started_at=started_at,
                        )
                    else:
                        for run in pending_runs:
                            checkpoint_run(
                                run,
                                run_single(data, estimator, method, tf, args, args.seed_base + run),
                            )

                    cls_payload[label] = build_label_payload(
                        estimator,
                        acc_runs,
                        ps_runs,
                        rs_runs,
                        f1_runs,
                        fit_runs,
                        feat_runs,
                        time_runs,
                        curves,
                        args.epochs,
                        completed_run_ids=run_ids,
                    )
                    save_cache(progress_file, cls_payload)
                    save_cache(cache_file, cls_payload)
            save_cache(cache_file, cls_payload)

            results_struct[dataset_name].update(cls_payload)

    excel_path = os.path.join(paths.res_dir, f"Global_Results_{paths.exp_tag}.xlsx")
    exported = export_global_excel(results_struct, dataset_names, excel_path)
    summary_df = generate_summary_dataframe(results_struct, args)
    summary_csv = os.path.join(paths.res_dir, f"RESUMEN_GRAFICAS_{paths.exp_tag}.csv")
    summary_df.to_csv(summary_csv, index=False)
    chart_dir = paths.fig_dir
    generated_charts = generate_seven_global_charts(summary_df, results_struct, chart_dir, list(args.optimizers), args)

    print("Completed.")
    print(f"Cache dir: {paths.cache_dir}")
    print(f"Figures dir: {paths.fig_dir}")
    print(f"Charts summary CSV: {summary_csv}")
    print(f"Charts dir: {chart_dir}")
    if generated_charts:
        print("Notebook-style charts:")
        for name in generated_charts:
            print(f"  - {os.path.join(chart_dir, name)}")
    print("Global results:")
    for p in exported:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
