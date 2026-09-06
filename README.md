# MIAFEx_MaCRO_DSA_DE_Feature_Selection

Framework for medical image feature extraction using MIAFEx and wrapper-based
feature selection with MaCRO-DE and other metaheuristic algorithms.

## Experiment configuration

Edit the grouped configuration block near the top of `main_best.py`.
CLI arguments override those defaults. Pressing Run now selects existing
**Brain_MRI MIAFEx features + DE + KNN**, with **1 run, 100 FS epochs,
50 agents, and parallel execution disabled**. The pipeline mode is
`feature_selection`; it never retrains MIAFEx or regenerates features.
`EXP_ID` and `REUSE_CACHE_FROM_EXP_ID` both remain `602`.

Set `MIAFEX_DATASETS = None` to discover all datasets, set a list, or pass
`--miafex-datasets Brain_MRI Chest_CT` to select a subset. `--dataset-name`
remains available for a single dataset. All optimizers and classifiers remain
available through configuration/CLI; only the default selection is narrowed.

- `extract`: prepare/reuse the checkpoint and both feature CSVs, then exit before FS.
- `feature_selection`: require existing feature CSVs; never train or extract,
  even when `--train-miafex yes` or `--extract-miafex yes` is supplied.
- `full`: prepare/reuse neural artifacts, then run the existing FS experiment.
- `--dataset-source mafese`: keep the existing `test14` suite; use `full` or
  `feature_selection` because MAFESE already provides feature datasets.

Checkpoints are stored in `checkpoints/miafex/<dataset>/miafex_checkpoint.pth`.
Reusable feature datasets are stored separately in
`datasets_features/miafex/<dataset>/train_features.csv` and `test_features.csv`,
alongside separate `train_features.npy` / `test_features.npy` and
`train_class_to_idx.json` / `test_class_to_idx.json` artifacts. MIAFEx trains only
on image `train/`. The same checkpoint then extracts `train/` and `test/`
separately, preserving the prepared image split and consistent label mappings.

FS loads the two CSVs directly with `Data.set_train_test`: no concatenation or
new outer holdout split is performed for MIAFEx datasets. The metaheuristic sees
only training rows; MAFESE's internal fitness-validation subset also comes only
from those training rows. After selection, the classifier is fitted on the full
training partition and evaluated on the original test partition. MAFESE's
internal-dataset train/test splitting behavior remains unchanged.

For training, `auto` reuses the checkpoint when it exists. For extraction, `auto`
requires **both** train/test CSVs; if either is missing, both partitions are
extracted using the same checkpoint. A legacy `extracted_features.csv` alone is
not sufficient and is never randomly split. `yes` forces that stage; `no`
disables it and requires its artifact when needed. The decisions are independent:
forcing training does not refresh an existing CSV pair unless extraction is also
forced. Feature-selection-only mode needs no checkpoint or image access when the
feature CSVs already exist.

`MIAFEX_EPOCHS` / `--miafex-epochs` controls neural-network training epochs.
`FS_EPOCHS` / `--epochs` (also `--fs-epochs`) controls metaheuristic iterations.
Cache reuse is enabled; `--no-reuse-cache` disables final/source reuse while
preserving current progress resume. `--parallel yes` enables concurrent runs.
`N_WORKERS = automatic_worker_count()` uses two-thirds of usable CPUs and
available RAM after a 512 MiB reserve, budgeting 192 MiB per worker. The worker
limit is also capped by pending runs. Native BLAS/OpenMP threading is unchanged;
the automatic process limit is a resource cap, not a measured optimal count.

Cache lookup prefers the current EXP. `REUSE_CACHE_FROM_EXP_ID` (or
`--reuse-cache-from-exp-id`) selects a read-only fallback; `None`/`none` disables
that fallback. Compatible source caches and legacy signatures are migrated
atomically into the current EXP before resuming missing run IDs. Source files
are never modified. `FIGURES_ONLY=True` or `--figures-only` uses the same lookup
without feature preparation or optimizer execution. Scheduling, output locations,
and stage switches are excluded from cache signatures; scientific settings and
the prepared feature paths remain included.

`--miafex-dataset-root`, `--miafex-checkpoint-root`, and `--feature-dataset-root`
override the three roots. Single-dataset overrides `--dataset-root`,
`--features-csv`, and `--miafex-output` remain available; `--miafex-output` sets the
checkpoint directory. `--features-csv` is a legacy location hint: its parent
directory supplies `train_features.csv` and `test_features.csv`, rather than
loading or dividing the old single CSV. `--train-features-csv` and
`--test-features-csv` override the individual partition paths. Cached results
retain their existing layout and resume behavior; a MIAFEx partition-version
marker and both CSV paths prevent reuse of old single-CSV/resplit results,
with a signature scoped to each dataset and shared between `full` and
`feature_selection` when all other settings match.

Safe inspection commands (no training or feature selection):

```bash
python main_best.py --help
python main_best.py --list-miafex-datasets
```

## Python environment

Python 3.13.15 is the target interpreter. The compiled scientific and deep-learning
packages in `requirements.txt` are pinned to releases that provide CPython 3.13
Linux wheels, avoiding unsupported source builds of older releases such as
`numpy==1.26.4`.

Create a clean virtual environment on Ubuntu with:

```bash
sudo apt update
sudo apt install python3.13 python3.13-venv
cd /path/to/MIAFEx_MaCRO_DSA_DE_Feature_Selection
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The `.python-version` file records the exact tested patch release, 3.13.15.

## Pinned compatibility stack

The core migration pins are:

- NumPy 2.1.3, SciPy 1.14.1, pandas 2.2.3, scikit-learn 1.6.0, and
  matplotlib 3.9.2: the earliest practical compatible release line with CPython
  3.13 Linux wheels.
- MAFESE 1.0.0, MEALPY 3.0.2, and Permetrics 2.0.0 remain unchanged.
- Plotly 5.24.1 and Kaleido 0.2.1 preserve the API used by MAFESE 1.0.0.
- PyTorch 2.7.1 and torchvision 0.22.1 are a matched pair with Python 3.13
  support. Transformers 4.48.3 and timm 1.0.14 complete the tested MIAFEx stack.

## Execution backends

`main_best.py` exposes three compute modes:

- `--compute-mode cpu`: MIAFEx, feature selection, and sklearn classifiers use CPU.
- `--compute-mode torch-gpu`: MIAFEx training/extraction use CUDA when available;
  MAFESE, MEALPY, and sklearn remain on CPU. This is the default and recommended
  GPU mode. If CUDA is unavailable, it retains the project's previous CPU fallback.
- `--compute-mode gpu-full`: uses the same PyTorch CUDA selection and additionally
  reports whether optional CuPy and cuML modules are installed. It does not yet
  change optimizer or classifier implementations.

The ML selector is `--ml-backend {auto,sklearn,cuml}`. `auto` currently resolves
to `sklearn` to preserve existing results. `cuml` is only a validated future
integration hook: it requires `gpu-full` and an installed cuML package, while the
actual classifiers remain sklearn until explicit adapters are implemented.

Print backend availability without starting an experiment:

```bash
python main_best.py --show-backends
```

Startup reports the Python and package versions, PyTorch CUDA availability, GPU
name, CuPy and cuML availability, selected compute mode, selected ML backend, and
actual MIAFEx device.

## Optional GPU dependencies

Common dependencies remain in `requirements.txt`. CUDA-specific additions belong
in `requirements-gpu.txt`. CuPy and cuML are intentionally not pinned yet: choose
their packages only after checking the NVIDIA driver and CUDA environment.
