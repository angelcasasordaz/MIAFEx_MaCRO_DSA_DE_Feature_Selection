# MIAFEx · Metaheuristic Feature Selection

A research framework for medical image classification that combines **MIAFEx
attention-based feature extraction** with **wrapper feature selection** using
MaCRO-DE, MaCRO-DE-t, DSADE, DBO, and MEALPY optimizers.

**Pipeline:** prepared image splits → MIAFEx training → feature extraction →
feature selection → classification → statistical reports and figures.

[Setup](#setup) · [Structure](#project-structure) · [Datasets](#datasets) ·
[Pipeline](#pipeline-modes) · [Features](#feature-extraction) ·
[Selection](#feature-selection) · [CPU/GPU](#cpugpu-behavior) ·
[Reporting](#reporting) · [Reproducibility](#reproducibility) ·
[Reference & license](#reference-and-license)

## Setup

**CPython 3.11 and 3.13 are supported.** Use a complete Python installation
with virtualenv support and a separate environment for each interpreter version.

Open the repository folder in PyCharm. In **Settings → Python Interpreter**,
create a local virtualenv at `.venv` using Python 3.11 or 3.13, or select an
existing `.venv` interpreter. Set run configurations to use that interpreter
and the repository root as the working directory.

Alternatively, create the environment manually in a terminal:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python verify_environment.py
```

For Python 3.13, substitute `python3.13` in the first command. On Windows,
activate with `.venv\Scripts\activate` and select `.venv\Scripts\python.exe`
in PyCharm. The verifier checks the installed stack without running experiments.

For the explicitly pinned PyTorch CUDA runtime on Linux x86_64, install
`requirements-gpu.txt` instead; it includes the base requirements. GPU execution
requires a compatible NVIDIA driver. `.venv/` and `.idea/` remain local and
ignored by Git.

## Project structure

| Path | Purpose |
|---|---|
| `main_best.py` | Pipeline entry point, configuration, execution, and cache orchestration |
| `miafex_model.py`, `train_miafex.py`, `extract_miafex_features.py` | Model, training, and feature extraction |
| `prepare_all_miafex_datasets.py` | Dataset preparation and read-only split checks |
| `*_optimizer.py`, `cec_de_mc_cf/` | Custom optimizers and the implementation used by MaCRO-DE-t |
| `algorithm_acronym_list.py`, `corrected_binary.py`, `scientific_cache.py` | Optimizer resolution, binary representation, and scientific cache identities |
| `reporting/`, `tests/` | Publication reporting and regression tests |
| `requirements*.txt`, `verify_environment.py` | Dependency pins and environment validation |
| `datasets/`, `checkpoints/`, `datasets_features/` | Local images, trained models, and extracted features; tracked placeholders only |
| `Results/`, `Figures/` | Generated experiment outputs; ignored by Git |
| `docs/`, `tools/` | Reference documents, technical notes, and preserved historical utilities |

## Datasets

The repository provides placeholders for seven medical image datasets:
**Brain MRI, Breast Ultrasound, Chest CT, Eye Fundus, Gastrointestinal Endoscopy,
Histological Biopsy, and Ocular Alignment**. Download sources are recorded in the
[dataset reference document](docs/references/Datasets%20m%C3%A9dicos%20links.docx).
Datasets, pretrained checkpoints, extracted features, and results are not bundled.
The pipeline also supports MAFESE's built-in feature datasets through
`--dataset-source mafese`.

Place classification images in `datasets/<name>/train/<class>/` and
`datasets/<name>/test/<class>/`, with matching class names. The preparer also
accepts unsplit class folders and validation partitions. Select a dataset with
`--datasets <name>`; add `--check` to verify it without changing files.

Valid existing splits are preserved. Missing or mirrored splits use the
preparer's deterministic, checksum-grouped 80/20 split with seed 42. Byte-identical
images remain in one partition. Preserve patient grouping when preparing inputs;
byte checks cannot identify patient overlap. Keep source archives and split
manifests locally.

## Pipeline modes

Review the configuration block in `main_best.py` before running it. Command-line
arguments override those defaults; `python main_best.py --help` lists options.
Choose datasets with `--miafex-datasets` and stages with `--pipeline-mode`.

| Mode | Behavior |
|---|---|
| `extract` | Prepare or reuse a checkpoint, extract train/test features, then stop |
| `feature_selection` | Use existing train/test feature CSVs; skip neural stages |
| `full` | Prepare or reuse neural artifacts, then run feature selection |

`--train-miafex` and `--extract-miafex` each accept `auto`, `yes`, or `no`:
reuse existing artifacts, force the stage, or require existing artifacts,
respectively. In `feature_selection` mode, neither neural stage runs.

## Feature extraction

MIAFEx refines a pretrained ViT's class-token representation into **768-dimensional
image descriptors**. Training uses only the prepared training partition; one
checkpoint extracts both partitions with consistent labels. Initial model use
requires access to the configured Hugging Face weights or a populated local cache.

Checkpoints live under `checkpoints/miafex/<dataset>/`. Extraction writes
`train_features.csv` and `test_features.csv`, NumPy arrays, and label mappings
under `datasets_features/miafex/<dataset>/`. Automatic extraction reuse requires
both CSVs and checks existence rather than image content; regenerate neural
artifacts explicitly when inputs change.

## Feature selection

MAFESE and MEALPY search binary feature subsets using classifier-based fitness.
Project optimizers include MaCRO-DE, MaCRO-DE-t, DSADE, and DBO; available choices
are exposed by the entry point. Configure optimizers, classifiers, transfer
functions, runs, epochs, and population size in `main_best.py` or through the CLI.

For MIAFEx inputs, search and internal fitness validation use training rows only.
The final classifier fits the full training partition and evaluates the original
test partition. The framework preserves the prepared outer split.

## CPU/GPU behavior

| Compute mode | Execution |
|---|---|
| `cpu` | All stages run on CPU |
| `torch-gpu` | MIAFEx uses CUDA when available, with CPU fallback; selection and sklearn classifiers run on CPU |
| `gpu-full` | Same execution as `torch-gpu`, plus optional CuPy/cuML availability checks |

`--ml-backend auto` resolves to sklearn. The cuML option is an integration hook;
classifier adapters are not implemented. Independent selection runs can execute
in parallel with worker limits based on available CPU and memory. Use
`--show-backends` to inspect device availability without starting an experiment.

## Reporting

Experiments generate summary tables, Excel workbooks, statistical comparisons,
and figures under `Results/EXP<id>/` and `Figures/EXP<id>/`.
`--figures-only` regenerates derived outputs from compatible cached runs without
training or optimization.

`--report-only --exp-id <id>` creates a versioned publication report from a
complete, validated final cache grid. Reports include high-resolution PNGs,
manuscript tables, Friedman and Wilcoxon–Holm analyses, and manifests identifying
source caches and outputs. `--report-output-root` selects a separate destination.
Report-only execution never schedules missing scientific runs.

## Reproducibility

Direct dependencies are pinned; transitive dependencies are not hash-locked.
Record the Python/package versions, hardware, configuration, dataset source and
checksums, split membership, checkpoint, feature files, and label mappings for
each study. Fixed seeds support repeatability, but results can vary across
platforms and compute backends.

Cache reuse checks scientific identities, including input contents, seeds,
parameters, implementation fingerprints, and package versions. Compatible
partial runs resume by run ID. Use a fresh experiment ID for a new study;
disabling final cache reuse does not disable current progress resume.

Keep generated data, checkpoints, features, caches, and reports in their ignored
locations. Preserve the tracked `.gitkeep` placeholders. The regression suite
lives in `tests/` and can be run with `python -m unittest discover -s tests`;
some historical validation checks require local evidence artifacts.

## Reference and license

The [MIAFEx reference paper](docs/references/MIAFEx%20-%20An%20attention-based%20feature%20extraction%20method%20for%20medical%20image%20classification.pdf)
describes the feature extraction method. Optimizer references are preserved in
`docs/references/`, with implementation provenance in
[cec_de_mc_cf/PROVENANCE.md](cec_de_mc_cf/PROVENANCE.md).

This repository includes the [GNU General Public License, version 3](LICENSE).
Consult each dataset and pretrained model's terms when obtaining those assets.
