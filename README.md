# MIAFEx_MaCRO_DSA_DE_Feature_Selection

Framework for medical image feature extraction using MIAFEx and wrapper-based
feature selection with MaCRO-DE and other metaheuristic algorithms.

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
