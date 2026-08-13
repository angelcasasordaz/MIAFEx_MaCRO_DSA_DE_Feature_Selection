import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import inspect
from typing import Dict, List

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

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

DEFAULT_OPTIMIZERS = [
    "PSO",
    "GWO",
    "WOA",
    "DE",
    "HHO",
    "FOX",
    "RIME",
    "RUN",
    "MaCRO-DE",
    "DSADE",
]
DEFAULT_ESTIMATORS = ["knn", "svm"]
DEFAULT_TRANSFER_FUNCTIONS = [
    "vstf_01",
    "vstf_02",
    "vstf_03",
    "vstf_04",
    "sstf_01",
    "sstf_02",
    "sstf_03",
    "sstf_04",
]

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

@dataclass
class Paths:
    exp_tag: str
    fig_dir: str
    res_dir: str
    cache_dir: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Framework de comparacion FS (multi-dataset, multi-run, cache)")

    general = parser.add_argument_group("General")
    general.add_argument("--exp-id", type=int, default=630, help="ID numerico del experimento")
    general.add_argument("--output-root", default=".", help="Raiz para Figures/Results")

    dataset = parser.add_argument_group("Dataset")
    dataset.add_argument("--dataset-source", default="mafese", choices=["mafese", "miafex"], help="Origen de datasets")
    dataset.add_argument("--dataset-suite", default="test14", choices=["test14"], help="Suite de datasets")
    dataset.add_argument("--dataset-name", default=None, help="Nombre del dataset cuando --dataset-source=miafex")
    dataset.add_argument("--features-csv", default=None, help="CSV de features generado por MIAFEx")
    dataset.add_argument("--list-miafex-datasets", action="store_true", help="Listar datasets MIAFEx disponibles en datasets/")
    dataset.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio")
    dataset.add_argument("--random-state", type=int, default=2, help="Semilla de split")

    miafex = parser.add_argument_group("MIAFEx")
    miafex.add_argument("--dataset-root", default=None, help="Raiz del dataset MIAFEx con subdirectorios train/ y test/")
    miafex.add_argument("--train-miafex", default="no", choices=["yes", "no"], help="Entrenar MIAFEx antes de MEALPY")
    miafex.add_argument("--extract-miafex", default="no", choices=["yes", "no"], help="Extraer features MIAFEx antes de MEALPY")
    miafex.add_argument("--miafex-output", default="./outputs/miafex", help="Directorio para artefactos MIAFEx")
    miafex.add_argument("--miafex-epochs", type=int, default=10, help="Epocas para entrenar MIAFEx")
    miafex.add_argument("--miafex-batch-size", type=int, default=16, help="Batch size para MIAFEx")
    miafex.add_argument("--miafex-learning-rate", type=float, default=1e-5, help="Learning rate para MIAFEx")

    feature_selection = parser.add_argument_group("Feature Selection")
    feature_selection.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS), help="Lista de optimizadores")
    feature_selection.add_argument("--list-optimizers", action="store_true", help="Listar optimizadores MEALPY/custom disponibles")
    feature_selection.add_argument("--estimators", nargs="+", default=DEFAULT_ESTIMATORS, help="Lista de clasificadores")
    feature_selection.add_argument("--transfer-functions", nargs="+", default=DEFAULT_TRANSFER_FUNCTIONS, help="Lista de transfer functions")
    feature_selection.add_argument("--runs", type=int, default=30, help="Ejecuciones independientes por combinacion")
    feature_selection.add_argument("--epochs", type=int, default=100, help="Iteraciones del optimizador")
    feature_selection.add_argument("--pop-size", type=int, default=50, help="Tamano de poblacion")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--seed-base", type=int, default=1234, help="Semilla base por run")
    execution.add_argument("--reuse-cache", action="store_true", help="Usar cache si existe")
    execution.add_argument("--figures-only", action="store_true", help="Regenerar solo graficas desde cache existente")
    execution.add_argument("--parallel", default="yes", choices=["yes", "no"], help="Ejecutar runs en paralelo: yes/no")
    execution.add_argument("--n-workers", type=int, default=12, help="Numero de procesos paralelos si --parallel yes")

    macro_dsade = parser.add_argument_group("MaCRO-DE / DSADE")
    macro_dsade.add_argument("--dsade-beta-min", type=float, default=0.2)
    macro_dsade.add_argument("--dsade-beta-max", type=float, default=0.8)
    macro_dsade.add_argument("--dsade-pcr", type=float, default=0.2)
    macro_dsade.add_argument("--dsade-mahal-q", type=float, default=0.68)
    return parser.parse_args()

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

def make_paths(args: argparse.Namespace) -> Paths:
    exp_tag = f"EXP{args.exp_id:03d}"
    fig_dir = os.path.join(args.output_root, "Figures", exp_tag)
    res_dir = os.path.join(args.output_root, "Results", exp_tag)
    cache_dir = os.path.join(res_dir, "cache")
    for p in (fig_dir, res_dir, cache_dir):
        os.makedirs(p, exist_ok=True)
    return Paths(exp_tag=exp_tag, fig_dir=fig_dir, res_dir=res_dir, cache_dir=cache_dir)

def resolve_mafese_dataset_names(args: argparse.Namespace) -> List[str]:
    if args.dataset_suite == "test14":
        return list(TEST_datasets_clasific_14)
    raise ValueError(f"Suite de datasets no soportada: {args.dataset_suite}")


def discover_miafex_datasets(base_dir="datasets") -> Dict[str, str]:
    if not os.path.isdir(base_dir):
        return {}

    discovered = {}
    for item in os.scandir(base_dir):
        if not item.is_dir():
            continue
        dataset_root = item.path
        train_dir = os.path.join(dataset_root, "train")
        test_dir = os.path.join(dataset_root, "test")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            discovered[item.name] = dataset_root
    return dict(sorted(discovered.items(), key=lambda row: row[0].lower()))


def print_miafex_datasets(datasets: Dict[str, str]) -> None:
    if not datasets:
        print("No MIAFEx image datasets found in: datasets/")
        print()
        print("Expected structure:")
        print("datasets/DatasetName/train/")
        print("datasets/DatasetName/test/")
        return
    print("Available MIAFEx datasets:")
    for idx, name in enumerate(datasets, start=1):
        print(f"{idx}. {name}")


def resolve_miafex_dataset_root(args: argparse.Namespace) -> None:
    if args.dataset_root:
        return

    discovered = discover_miafex_datasets()
    if args.dataset_name in discovered:
        args.dataset_root = discovered[args.dataset_name]
        return

    available = ", ".join(discovered.keys()) if discovered else "none"
    raise ValueError(
        f"Dataset MIAFEx '{args.dataset_name}' no encontrado en datasets/. "
        f"Datasets disponibles: {available}"
    )


def load_miafex_csv(csv_path: str):
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

    y_values = y_series.to_numpy()
    if pd.api.types.is_numeric_dtype(y_series):
        y = y_values
    else:
        y = LabelEncoder().fit_transform(y_values.astype(str))

    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("El CSV de MIAFEx debe contener al menos 2 clases.")

    return X, y


def resolve_miafex_csv(args: argparse.Namespace) -> str:
    run_training = args.train_miafex == "yes"
    run_extraction = args.extract_miafex == "yes"

    if (run_training or run_extraction) and MIAFEX_IMPORT_ERROR is not None:
        raise ImportError(
            "No se pudieron importar las funciones MIAFEx requeridas para entrenar/extraer. "
            "Revisa la instalacion de torch/transformers en el entorno."
        ) from MIAFEX_IMPORT_ERROR

    if not run_extraction:
        if not args.features_csv:
            raise ValueError(
                "--features-csv es requerido cuando --extract-miafex no. "
                "Usa --extract-miafex yes para generarlo desde dataset-root/test."
            )
        if not os.path.isfile(args.features_csv):
            raise FileNotFoundError(f"CSV de features no encontrado: {os.path.abspath(args.features_csv)}")

    if run_training or run_extraction:
        if not args.dataset_root:
            raise ValueError("--dataset-root es requerido cuando --train-miafex yes o --extract-miafex yes")
        if not os.path.isdir(args.dataset_root):
            raise FileNotFoundError(f"dataset-root inexistente: {os.path.abspath(args.dataset_root)}")

        train_dir = os.path.join(args.dataset_root, "train")
        test_dir = os.path.join(args.dataset_root, "test")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Directorio train inexistente: {os.path.abspath(train_dir)}")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Directorio test inexistente: {os.path.abspath(test_dir)}")

        os.makedirs(args.miafex_output, exist_ok=True)
        checkpoint_path = os.path.join(args.miafex_output, "miafex_checkpoint.pth")

        if run_training:
            checkpoint_path = train_miafex(
                train_root=train_dir,
                output_dir=args.miafex_output,
                num_classes=None,
                num_epochs=args.miafex_epochs,
                batch_size=args.miafex_batch_size,
                learning_rate=args.miafex_learning_rate,
                device=None,
            )
        elif not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Checkpoint MIAFEx inexistente. Ejecuta con --train-miafex yes o coloca el archivo en: "
                f"{os.path.abspath(checkpoint_path)}"
            )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint MIAFEx inexistente: {os.path.abspath(checkpoint_path)}")

        if run_extraction:
            csv_path = extract_miafex_features(
                data_dir=test_dir,
                checkpoint_path=checkpoint_path,
                output_dir=args.miafex_output,
                batch_size=args.miafex_batch_size,
                device=None,
                run_ml_baselines=False,
            )
        else:
            csv_path = args.features_csv
    else:
        csv_path = args.features_csv

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV de features no encontrado: {os.path.abspath(csv_path)}")
    return csv_path


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

def build_cache_signature(args: argparse.Namespace) -> str:
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
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

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
    completed_by_run = {}
    next_run = min(pending_runs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_parallel_task, task) for task in tasks]
        for future in as_completed(futures):
            run, out = future.result()
            completed_by_run[run] = out
            while next_run in completed_by_run:
                item = (next_run, completed_by_run.pop(next_run))
                if on_run_complete is not None:
                    on_run_complete(*item)
                completed.append(item)
                next_run += 1
    return completed


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
):
    curve_mean = pad_mean_curves(curves, epochs)
    return {
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

def save_cache(path: str, payload: dict):
    with open(path, "wb") as f:
        pickle.dump(payload, f)

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

def load_results_from_cache(paths: Paths, args: argparse.Namespace, dataset_names: List[str], cache_sig: str) -> Dict[str, Dict]:
    results_struct = {}
    missing = []
    for dataset_name in dataset_names:
        results_struct[dataset_name] = {}
        for estimator in args.estimators:
            cache_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_results.pkl",
            )
            progress_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_progress.pkl",
            )
            payload = load_cache_safe(cache_file, "cache final")
            if payload is None:
                payload = load_cache_safe(progress_file, "checkpoint parcial")
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
        total += int(row.get("CompletedRuns", len(row.get("AccRuns", []))))
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
        i for i, opt in enumerate(opts)
        if str(method_by_group.get(opt)).upper() == "MACRO-DE"
    )

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

def regenerate_figures_from_cache(paths: Paths, args: argparse.Namespace, dataset_names: List[str], cache_sig: str):
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


def print_experiment_summary(args: argparse.Namespace, paths: Paths, dataset_names: List[str], cache_sig: str, miafex_csv_path: str | None):
    print("=" * 60)
    print(" MIAFEx + Metaheuristic Feature Selection Framework")
    print("=" * 60)
    print(f"{'Experiment':<18}: {paths.exp_tag}")
    print(f"{'Dataset source':<18}: {args.dataset_source}")
    if args.dataset_source == "mafese":
        print(f"{'Dataset suite':<18}: {args.dataset_suite} ({len(dataset_names)} datasets)")
        print(f"{'Datasets':<18}: {_summary_value(dataset_names)}")
    else:
        print(f"{'Dataset':<18}: {args.dataset_name}")
        print(f"{'MIAFEx training':<18}: {args.train_miafex}")
        print(f"{'Feature extraction':<18}: {args.extract_miafex}")
        print(f"{'Features CSV':<18}: {miafex_csv_path}")
    print(f"{'Optimizers':<18}: {_summary_value(args.optimizers, optimizer_display_label)}")
    print(f"{'Classifiers':<18}: {_summary_value(args.estimators, lambda x: str(x).upper())}")
    print(f"{'Transfer functions':<18}: {_summary_value(args.transfer_functions, lambda x: str(x).upper())}")
    print(f"{'Runs':<18}: {args.runs}")
    print(f"{'Epochs':<18}: {args.epochs}")
    print(f"{'Population':<18}: {args.pop_size}")
    print(f"{'Parallel':<18}: {args.parallel}")
    print(f"{'Cache signature':<18}: {cache_sig}")
    print("=" * 60)


def main():
    args = parse_args()
    logging.disable(logging.INFO)
    logging.getLogger("mealpy").setLevel(logging.WARNING)

    if args.list_miafex_datasets:
        print_miafex_datasets(discover_miafex_datasets())
        return
    if args.list_optimizers:
        print(list_available_optimizers())
        return

    validate_selection_options(args)
    args.optimizers = resolve_optimizers(args)
    if args.runs < 1:
        raise ValueError("--runs debe ser >= 1")
    if args.n_workers < 1:
        raise ValueError("--n-workers debe ser >= 1")
    if args.dataset_source == "miafex":
        if not args.dataset_name:
            raise ValueError("--dataset-name es requerido cuando --dataset-source=miafex")
        resolve_miafex_dataset_root(args)

    paths = make_paths(args)
    cache_sig = build_cache_signature(args)
    show_tf = len(args.transfer_functions) > 1
    show_cls = len(args.estimators) > 1

    if args.dataset_source == "mafese":
        dataset_names = resolve_mafese_dataset_names(args)
        miafex_arrays = None
        miafex_csv_path = None
    else:
        dataset_names = [args.dataset_name]
        miafex_csv_path = resolve_miafex_csv(args)
        miafex_arrays = load_miafex_csv(miafex_csv_path)

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
        if args.dataset_source == "mafese":
            mafese_data = get_dataset(dataset_name)
            if mafese_data is None:
                raise ValueError(
                    f"mafese no pudo cargar '{dataset_name}'. "
                    "Verifica que exista en la suite 'test14' de mafese."
                )
            X = np.asarray(mafese_data.X, dtype=np.float64)
            y = np.asarray(mafese_data.y).astype(np.int32)
        else:
            X, y = miafex_arrays
        data = Data(X, y)
        try:
            data.split_train_test(test_size=args.test_size, random_state=args.random_state, stratify=y)
        except ValueError:
            data.split_train_test(test_size=args.test_size, random_state=args.random_state)

        for estimator in args.estimators:
            cache_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_results.pkl",
            )
            progress_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_progress.pkl",
            )
            cache_payload = load_cache_safe(cache_file, "cache final") if args.reuse_cache else None
            progress_payload = load_cache_safe(progress_file, "checkpoint parcial")
            if cache_payload is not None and (
                progress_payload is None
                or payload_completed_runs(cache_payload) >= payload_completed_runs(progress_payload)
            ):
                print(f"[cache] {dataset_name} / {estimator}")
                cls_payload = cache_payload
            else:
                cls_payload = progress_payload or {}
                if progress_payload is not None:
                    print(f"[resume] Reanudando {dataset_name} / {estimator} desde checkpoint parcial")
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

                    done = len(acc_runs)
                    if done >= args.runs:
                        print(f"Running {dataset_name} | {label} | runs={args.runs} (already complete)")
                        continue
                    print(f"Running {dataset_name} | {label} | runs={args.runs} (resume from {done})")

                    pending_runs = list(range(done, args.runs))
                    def checkpoint_run(run, out):
                        acc_runs.append(out["as_test"])
                        ps_runs.append(out["ps_test"])
                        rs_runs.append(out["rs_test"])
                        f1_runs.append(out["f1_test"])
                        fit_runs.append(out["fit_final"])
                        feat_runs.append(out["n_features"])
                        time_runs.append(out["runtime"])
                        curves.append(out["curve"])
                        print(
                            f"  Run {run + 1:02d} | Acc={acc_runs[-1]:.2f}% | F1={f1_runs[-1]:.4f} | "
                            f"Fit={fit_runs[-1]:.4f} | Feat={feat_runs[-1]} | Time={time_runs[-1]:.2f}s"
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
                        )
                        save_cache(progress_file, cls_payload)
                        save_cache(cache_file, cls_payload)

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
