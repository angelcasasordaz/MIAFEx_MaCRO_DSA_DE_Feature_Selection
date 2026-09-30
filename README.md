# MIAFEx_MaCRO_DSA_DE_Feature_Selection

Framework for medical image feature extraction using MIAFEx and wrapper-based
feature selection with MaCRO-DE and other metaheuristic algorithms.

## Fresh clone / Python 3.11

**CPython 3.11.x and 3.13.x are supported. Python 3.13 is not required.**
Start with a complete Python 3.11 installation on Ubuntu/Linux, including
`venv`, `pip`, `ssl`, `lzma`, `bz2`, `sqlite3`, and `ctypes` support. On Ubuntu
releases whose configured repositories provide Python 3.11, install the
prerequisites with `sudo apt update` and
`sudo apt install git python3.11 python3.11-venv`. Package availability depends
on the Ubuntu release; if unavailable, install complete CPython 3.11 from a
trusted distribution first. Do not substitute Python 3.13 just to open the project.
Use a supported PyCharm release with Python 3.11 support.

```bash
git clone https://github.com/angelcasasordaz/MIAFEx_MaCRO_DSA_DE_Feature_Selection.git
cd MIAFEx_MaCRO_DSA_DE_Feature_Selection
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python verify_environment.py
```

Run the commands from the repository root. Alternatively, after cloning and installing Python
3.11, run `bash scripts/setup_linux_python311.sh`. The helper creates or reuses
this checkout's `.venv`, checks that it uses Python 3.11, installs requirements,
runs `pip check`, and verifies imports. It stops on errors, never deletes an
existing environment, and does not alter PyCharm settings. It can be rerun;
it does not activate the calling shell. Afterwards use
`source .venv/bin/activate` in each new terminal, or call `.venv/bin/python`
explicitly. No training, feature extraction, or feature selection occurs during setup.

`.venv/` is local and ignored by Git: a clone contains no interpreter or installed
packages. `.idea/` and `*.iml` are also local and ignored in full. No shared IDE
configuration is required by this repository; PyCharm generates its own project
files on open. Those files must not encode another computer's interpreter or
SDK name. The repository no longer supplies `.python-version`: that file used
to select 3.13.15 implicitly. Interpreter-manager users may create an ignored
local `.python-version` for either supported version. The commands above select
3.11 explicitly and do not depend on pyenv, uv, or a prior environment.

### Select the interpreter visually in PyCharm

Open the cloned **repository folder** using **File → Open**. Open **Settings →
Python → Interpreter** (or **Settings → Project → Python Interpreter** in older
versions). Choose the appropriate path:

- **A — `.venv` does not exist:** **Add Interpreter → Add Local Interpreter →
  Generate new → Virtualenv**. Set **Base Python** to the complete Python **3.11**
  executable (for example `/usr/bin/python3.11`) and **Location** to
  `<project>/.venv`. Leave **Inherit global site-packages** unchecked. Click **OK**.
  Then install dependencies and validate with the last four commands above in
  an activated terminal, or use `.venv/bin/python` explicitly.
- **B — `.venv` already exists** (including after terminal/helper setup):
  **Add Interpreter → Add Local Interpreter → Select existing → Python**.
  Browse to **`<project>/.venv/bin/python`**, then click **OK**.
  Do not use **Generate new** for an existing environment.

In both cases, select the newly added interpreter in the **project interpreter
dropdown**, click **Apply → OK**, and verify the displayed executable belongs to
this checkout. Adding an SDK entry alone is not enough if the project still
selects a previous entry. Run `verify_environment.py` from PyCharm to confirm
the executable and **Inside project .venv: True**. Set run configurations to use
the project interpreter and the repository root as their working directory.
See the [JetBrains virtualenv instructions](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html).

For a previously opened project with an `[invalid]` entry, select the valid local
interpreter and apply the change; remove the obsolete SDK registration via
**Show All / Manage Interpreters** if needed. Check that individual run
configurations also use the project interpreter. Existing local IDE state is
not rewritten by a Git pull. A fresh clone has no such state to inherit.

### Validate before running research

`verify_environment.py` prints the executable, Python version, venv status,
platform, versions of NumPy, SciPy, pandas, scikit-learn, matplotlib, MAFESE,
MEALPY, Permetrics, PyTorch, torchvision, transformers, and timm, plus CUDA
availability and GPU names when available. It checks imports and small CPU
binary compatibility operations and exits nonzero on failure. Run `python -m
pip check` as well to check declared dependency compatibility. These checks
do not access datasets or experiment artifacts. They validate the software
environment, not dataset preparation or experimental outcomes.

For a safe project startup check use `python main_best.py --help`. Before running
an experiment, follow the dataset workflow below and review the experiment
configuration. A clone supplies no medical datasets, trained models, or features.
MIAFEx model construction also loads pretrained
`google/vit-base-patch16-224-in21k` weights from Hugging Face: research execution
needs internet access on first use or an already populated local model cache.
Environment verification never constructs that model or downloads its weights.

The base installation supports CPU operation without an NVIDIA driver. On Linux
x86_64, the pinned PyPI PyTorch wheel also downloads CUDA runtime packages, so
allow several GB of disk space even for CPU use. Actual GPU use additionally
requires suitable hardware and a compatible host driver; those prerequisites
are separate from basic setup. See **Optional GPU dependencies** below.
The direct dependencies are pinned; transitive dependencies are resolved by pip,
so this is not a hash-locked, bit-for-bit environment snapshot. Keep the printed
versions (or `python -m pip freeze`) with research environment records.

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

## Pinned compatibility stack

The unchanged compatibility pins support both Python 3.11 and 3.13:

- NumPy 2.1.3, SciPy 1.14.1, pandas 2.2.3, scikit-learn 1.6.0, and
  matplotlib 3.9.2 provide Linux wheels for both supported Python versions.
- MAFESE 1.0.0, MEALPY 3.0.2, and Permetrics 2.0.0 remain unchanged.
- Plotly 5.24.1 and Kaleido 0.2.1 preserve the API used by MAFESE 1.0.0.
- PyTorch 2.7.1 and torchvision 0.22.1 are a matched pair with Python 3.13
  support as well as 3.11. Transformers 4.48.3 and timm 1.0.14 complete the stack.

For Python 3.13, substitute `python3.13 -m venv .venv` in the manual workflow,
using a separate checkout/environment rather than mixing Python versions in an
existing venv. The one-command helper intentionally targets Python 3.11.

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

Clean-clone audit, 2026-09-30: a temporary snapshot of Git-tracked source plus
the environment/documentation changes started with no `.venv` or `.idea`.
Complete CPython **3.11.16** created a fresh venv and installed the unchanged
`requirements.txt`. `pip check`, `verify_environment.py`, imports of
`miafex_model`, `train_miafex`, `extract_miafex_features`, and `main_best`,
`main_best.py --help`, and all three `tests.test_dispatch` tests passed.
The final bootstrap helper also passed when reusing that environment.
CUDA inspection detected runtime 12.6 and an NVIDIA GeForce RTX 3060;
no GPU computation, training, feature extraction, or feature selection was run.
No EXP604/EXP605 artifacts were accessed by these checks. The PyCharm API
reported a valid local Python 3.11.16 module interpreter; a separate fresh
PyCharm GUI session was not automated in this audit.

Earlier recorded validation covered Python **3.13.15** and GPU-requirements
resolution on both supported Python versions. Python 3.13 was not retested in
this audit; its dependency pins remain unchanged. A Python installation that
lacks standard-library modules such as `_lzma` is incomplete even if it can
create a venv; the verifier now reports such failures explicitly.

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

## Cache-only publication reports

Reporting is an explicit opt-in. From the project root, report the validated
EXP604 caches without training, extraction, feature selection or cache migration:

```sh
.venv/bin/python main_best.py --report-only --exp-id 604
```

In PyCharm, add `--report-only --exp-id 604` to the Run configuration's script
parameters. `REPORT_ONLY = False` preserves the ordinary Run behavior. The alias
`--full-replica-report-only` enables the same reporting path.

`--output-root` selects the source containing `Results/EXPxxx/cache`.
`--report-output-root /path/to/reports` optionally selects a separate destination.
Each successful invocation appends matching `Figures/EXPxxx/full_repN` and
`Results/EXPxxx/full_repN` directories. The report number is a presentation version,
not another scientific repetition. Existing versions are preserved.

The `reporting/` package validates final `combinations_v2` envelopes or migrated
references bound to the SHA-256 of an original final cache. It reports the stored
scientific identities, including historical revisions; it never treats EXP604 as
compatible with corrected EXP605. Progress files, cross-EXP fallback, run merging
and cache repair are excluded. No image or feature CSV contents are loaded.
Dataset/classifier/optimizer/transfer selections and run/epoch/population/seed
settings must match a complete, unambiguous cache grid. Missing data raises an
error before report allocation; it never schedules an experiment.

Outputs include 600 dpi PNGs, Global/Statistical/Friedman Excel workbooks, editable
manuscript tables, matched-block Friedman and pairwise Wilcoxon-Holm CSV/text
reports, and validation manifests in both trees. Manifests record exact source
identities/hashes, output hashes, skipped unavailable metrics/curves, and zero
scientific calls. Runtime guards reject scientific execution and writes outside
the new staging directories; publication occurs only after validation.
