# MIAFEx_MaCRO_DSA_DE_Feature_Selection

Framework for medical image feature extraction using MIAFEx and wrapper-based
feature selection with MaCRO-DE and other metaheuristic algorithms.

## Experiment configuration

Edit the grouped configuration block near the top of `main_best.py`.
CLI arguments override those defaults. The current defaults select all discovered
MIAFEx datasets, MaCRO-DE-t/DE/JADE/SHADE/PSO/GWO/WOA/HHO/BRO/DBO/RUN/FOX/FLA,
KNN/SVM, `vstf_01`, 20 runs, 200 FS epochs, 30 agents, and parallel execution.
`PIPELINE_MODE = "full"`, `MIAFEX_TRAIN = "yes"`, and `MIAFEX_EXTRACT = "yes"`
force MIAFEx retraining and regeneration of both train/test feature partitions,
then proceed to feature selection.
`EXP_ID = 604`, `REUSE_CACHE = True`, and `REUSE_CACHE_FROM_EXP_ID = None`
disable reuse from other experiments while preserving cache/resume within EXP 604.
Existing compatible EXP 604 results/progress can still be reused; a fresh FS run
assumes EXP 604 has no existing compatible results/progress.

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

`MIAFEX_EPOCHS` / `--miafex-epochs` controls neural-network training epochs (50).
`MIAFEX_BATCH_SIZE` / `--miafex-batch-size` defaults to 8, and
`MIAFEX_LEARNING_RATE` / `--miafex-learning-rate` defaults to `1e-4`.
Training uses PyTorch NAdam with `CrossEntropyLoss`. The `train_miafex.py`
function and CLI share these defaults: 50 epochs, batch size 8, learning rate `1e-4`.
`FS_EPOCHS` / `--epochs` (also `--fs-epochs`) controls metaheuristic iterations
(200), and `POP_SIZE` / `--pop-size` defaults to 30 agents.
Cache reuse is enabled; `--no-reuse-cache` disables final/source reuse while
preserving current progress resume. `--parallel yes` enables concurrent runs.
`N_WORKERS = automatic_worker_count()` uses two-thirds of usable CPUs and
available RAM after a 512 MiB reserve, budgeting 1 GiB per worker (the imported
stack alone measured about 735 MiB per fresh process). The worker
limit is also capped by pending runs. Each wrapper run limits BLAS/OpenMP and
joblib to one thread during selection and final evaluation, including serial
runs. The limits are restored afterwards, so MIAFEx extraction keeps its own
thread settings. Workers use `spawn` to avoid inheriting initialized native
thread pools. A pool initializer installs read-only train/test arrays once per
worker; individual run tasks carry metadata and seeds, not the dataset arrays.
Native thread pools stay limited to one thread per worker, and the existing
per-run joblib/thread limits, heartbeat, and immediate checkpoints are retained.
The automatic process limit is a resource cap, not a measured
optimal count; `--n-workers` can lower it on memory-constrained machines.

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

## Multi-run statistical reporting

Normal result reporting and `--figures-only` also write
`Results/EXPxxx/Statistical_Results_EXPxxx.xlsx` under `--output-root`.
This additional workbook leaves Global Results, the summary CSV, and figures
unchanged. It reads the cached per-run arrays without training or optimization
in figures-only mode, including available runs in partial/resumable caches.
Cache formats and scientific signatures are unchanged.

The seven sheets are `Accuracy`, `Precision`, `Recall`, `F1Score`, `Fitness`,
`Features`, and `Time`. Each has `Dataset` and `Statistic` columns followed by
one column per `Optimizer | CLASSIFIER | TRANSFER_FUNCTION` combination in
configured optimizer/classifier/transfer order. Configured combinations with no
available runs retain blank cells; additional cached combinations are retained.
Each dataset has four rows: `Best`, `Worst`, `Mean`, and `Std`.
Best is the maximum for Accuracy/Precision/Recall/F1Score and the minimum for
Fitness/Features/Time; Worst uses the opposite direction. All statistics ignore
NaN and infinite values independently per metric. Std uses `ddof=1`, or zero
for one finite run; no finite runs produces blank cells. Values retain their
stored units (Accuracy in percent, Precision/Recall/F1Score as fractions,
Features as counts, and Time in seconds). Missing arrays are not inferred from
cached means.

`Full_Friedman_Analysis_EXPxxx.xlsx` adds the FULL-comparison analysis, separately
for every classifier/transfer-function pair. Its sheets are `FULL_Friedman`,
`FULL_Average_Ranks`, `FULL_PostHoc_Holm`, `FULL_Block_Fitness`, and
`FULL_Block_Ranks`, each identifying `Classifier` and `TransferFunction`.
Dataset blocks contain mean finite `FitRuns` values for each configured optimizer
(lower is better). Only datasets with a finite mean for every configured optimizer
enter a comparison. Average ranks use average ties; Friedman requires at least
three methods and two complete datasets. Insufficient data or an all-tied,
undefined test is reported explicitly with blank test results. A significant
Friedman result (`p < 0.05`) enables two-sided paired Wilcoxon tests with Holm
correction across all method pairs **within that classifier/transfer stratum**.
No classifiers or transfer functions are pooled, and cached means are not used
as substitutes for missing run arrays.

`FIGURES_ONLY=True` / `--figures-only` regenerates **all** derived outputs from
compatible cache: Global Results, the summary CSV, both statistical workbooks,
and the existing figures. Image/feature data are not loaded and no neural or
feature-selection execution occurs.

Before normal full/feature-selection execution loads data or prepares neural
artifacts, the framework checks compatible **current-EXP** caches for every
requested dataset/optimizer/classifier/transfer combination and every requested
run ID. A complete cache prints `[experiment-complete]` and regenerates those
same outputs immediately. Partial or missing combinations retain the existing
execution/resume flow. This check does not import source-EXP caches and respects
`--no-reuse-cache` (current progress remains resumable). Atomic cache writes,
non-contiguous run IDs, source priority, legacy handling, implementation-revision
guards, and scientific signatures are unchanged.

## Python environment

Python 3.11 and 3.13 are supported; the IDE uses 3.13.15. The compiled scientific and deep-learning
packages in `requirements.txt` are pinned to releases that provide CPython 3.13
Linux wheels, avoiding unsupported source builds of older releases such as
`numpy==1.26.4`.

Create a clean virtual environment on Ubuntu with:

```bash
sudo apt update
sudo apt install python3.13 python3.13-venv
cd /path/to/MIAFEx_MaCRO_DSA_DE_Feature_Selection
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

`requirements-gpu.txt` includes `requirements.txt` and explicitly pins the CUDA
12.6 libraries and Triton required by the existing PyPI PyTorch 2.7.1 wheel on
Linux x86_64. These are already transitive dependencies of base PyTorch on that
platform: the GPU file records the runtime explicitly rather than adding unused
CuPy/cuML. It supports Python 3.11 and 3.13. Other platforms use the base
requirements; this file only specifies a GPU runtime for Linux x86_64.
The matched PyTorch/torchvision versions and CUDA 12.6 option are listed in the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v271).
An NVIDIA host driver compatible with CUDA 12.6 is required for actual GPU use.
The runtime pins do not install a driver, and CPU fallback remains available.

Install and validate imports without training, extraction, or feature selection:

```bash
python -m pip install --only-binary=:all: -r requirements-gpu.txt
python -m pip check
python -c "import torch, torchvision, transformers, timm, mafese, mealpy; import miafex_model, train_miafex, extract_miafex_features; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Validation: Python 3.11.16 (isolated wheel installation) and 3.13.15 (existing
IDE environment) passed project/stack imports, `pip check`, and resolution of
`requirements-gpu.txt`. Both imported PyTorch/torchvision CUDA 12.6 builds;
no CUDA device was available, so no GPU execution was tested. The system's
separate Python 3.11.13 lacks `_lzma`; use a complete Python installation with
the standard `lzma` module for torchvision.

## Local dataset workflow

Git tracks source/documentation and 14 image split `.gitkeep` placeholders; it
tracks no image datasets, checkpoints, or generated features. The ignored
`datasets/`, `checkpoints/`, and `datasets_features/` trees allow `.gitkeep`
files through, including placeholders for `checkpoints/miafex/` and
`datasets_features/miafex/`. Keep downloads and generated files in these roots.
Do not use `git add -f` for data or experiment artifacts.

1. Download the chosen dataset manually from the links recorded in
   `Datasets médicos links.docx`. Keep the source archive locally (for example
   in ignored `downloads/`), and record its release/version, URL, download date,
   and SHA-256 in a local `dataset_audit/` note. The existing source mappings are:

   | Local folder | Source from the repository's dataset list |
   |---|---|
   | `Brain_MRI` | [Brain tumor](https://www.kaggle.com/datasets/sami009mr/brain-tumor-dataset) |
   | `Breast_Ultrasound` | [Breast ultrasound](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) |
   | `Chest_CT` | [Chest CT](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
   | `Eye_Fundus` | [Cataract](https://www.kaggle.com/datasets/jr2ngb/cataractdataset) |
   | `Gastrointestinal_Endoscopy` | [Kvasir](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation) |
   | `Histological_Biopsy` | [Lymphoma](https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification) |
   | `Ocular_Alignment` | [Strabismus](https://www.kaggle.com/datasets/ananthamoorthya/strabismus) |

2. Place images under `datasets/<name>/<class>/...` for an unsplit dataset, or
   `datasets/<name>/train/<class>/...` and `test/<class>/...` for supplied splits.
   Remove archive wrapper directories and normalize split names (for example
   `Training` to `train`, `Testing` to `test`). Keep class names identical across
   splits. The preparer also recognizes `valid`, `val`, and `validation`.
   For Brain_MRI, use `glioma_tumor`, `meningioma_tumor`, `no_tumor`, and
   `pituitary_tumor`. Do not mix masks into classification image folders.

3. Prepare only the selected dataset, then verify it without modifying files:

   ```bash
   python prepare_all_miafex_datasets.py --datasets Brain_MRI
   python prepare_all_miafex_datasets.py --datasets Brain_MRI --check
   ```

   Valid existing membership is preserved. Missing/mirrored splits use the
   existing seed-42, per-class 80/20 checksum-group split. Other invalid splits
   retain test copies and merge validation into train. Byte-identical duplicates
   cannot cross partitions; label conflicts are reported and kept together.
   This checks byte overlap, not patient identity: retain any supplied patient
   grouping when placing data. Preparation stages and verifies copies before
   replacing the source tree; retain your original archive. `--check` fails if
   repair is needed. Use this preparer rather than `prepare_brain_mri.py`, whose
   legacy file-level split can separate duplicate image bytes.

   Record exact membership/content locally, from the repository root:

   ```bash
   mkdir -p dataset_audit
   find datasets/Brain_MRI/train datasets/Brain_MRI/test -type f ! -name .gitkeep -print0 | sort -z | xargs -0 sha256sum > dataset_audit/Brain_MRI.sha256
   sha256sum -c dataset_audit/Brain_MRI.sha256
   ```

   Preserve this manifest with the source archive; the checksum command detects
   changed/missing listed files, while `--check` verifies split structure and
   overlap. Recreate and compare manifests to detect added files.

4. Train/reuse MIAFEx and extract both prepared partitions, stopping before FS:

   ```bash
   python main_best.py --dataset-source miafex --miafex-datasets Brain_MRI --pipeline-mode extract --compute-mode torch-gpu --train-miafex auto --extract-miafex auto
   ```

   A first extraction may fetch the configured pretrained ViT weights. Training
   reads `train/` only. The same checkpoint extracts both train and test into
   `datasets_features/miafex/Brain_MRI/`, with CSVs, NPY arrays, and label maps.
   Preserve the checkpoint, mappings, manifests, and command/configuration
   together locally to reproduce inputs to selection. If images/splits change,
   explicitly use `--train-miafex yes --extract-miafex yes`; automatic reuse checks
   artifact existence, not content hashes. Use a new experiment ID afterwards.

5. Run selection against those existing feature files. This explicit tiny
   example uses one DE/KNN run, two epochs, and five agents:

   ```bash
   python main_best.py --dataset-source miafex --miafex-datasets Brain_MRI --pipeline-mode feature_selection --optimizers DE --estimators knn --transfer-functions vstf_01 --runs 1 --fs-epochs 2 --pop-size 5 --parallel no --exp-id 990001 --output-root outputs/brain_mri_tiny --no-reuse-cache --reuse-cache-from-exp-id none
   ```

   Choose a fresh output directory/experiment ID for a fresh execution; existing
   progress can still resume with `--no-reuse-cache`. Feature selection preserves
   the outer train/test partitions and never trains or extracts in this mode.
