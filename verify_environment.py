#!/usr/bin/env python3
"""Check the runtime stack without loading datasets or running experiments."""

import importlib
from importlib.metadata import version
from pathlib import Path
import platform
import sys


PACKAGES = (
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("pandas", "pandas", "pandas"),
    ("scikit-learn", "sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib", "matplotlib"),
    ("MAFESE", "mafese", "mafese"),
    ("MEALPY", "mealpy", "mealpy"),
    ("Permetrics", "permetrics", "permetrics"),
    ("PyTorch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision"),
    ("transformers", "transformers", "transformers"),
    ("timm", "timm", "timm"),
    ("Plotly", "plotly", "plotly"),
    ("Kaleido", "kaleido", "kaleido"),
    ("joblib", "joblib", "joblib"),
    ("threadpoolctl", "threadpoolctl", "threadpoolctl"),
    ("openpyxl", "openpyxl", "openpyxl"),
    ("tqdm", "tqdm", "tqdm"),
)


def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    in_venv = sys.prefix != sys.base_prefix
    project_venv = Path(__file__).resolve().parent / ".venv"
    print(f"Virtual environment: {in_venv} ({sys.prefix})")
    print(f"Inside project .venv: {in_venv and Path(sys.prefix).resolve() == project_venv.resolve()}")
    print(f"Platform: {platform.platform()} ({platform.machine()})")
    errors = []
    if sys.version_info[:2] not in ((3, 11), (3, 13)):
        errors.append("Use a supported CPython 3.11.x or 3.13.x interpreter.")
    if platform.python_implementation() != "CPython":
        errors.append("The pinned binary stack requires CPython.")

    # Incomplete custom Python builds can create a venv but break torchvision.
    for name in ("lzma", "ssl", "sqlite3", "bz2", "ctypes"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"Standard library {name}: {type(exc).__name__}: {exc}")

    modules = {}
    for label, module_name, distribution in PACKAGES:
        try:
            modules[module_name] = importlib.import_module(module_name)
            print(f"{label}: {version(distribution)}")
        except Exception as exc:
            print(f"{label}: FAILED")
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if "transformers" in modules:
        try:
            # Resolve the lazy API used by miafex_model, without constructing
            # a model or downloading weights.
            getattr(modules["transformers"], "ViTForImageClassification")
            print("Transformers ViTForImageClassification import: OK")
        except Exception as exc:
            errors.append(f"Transformers ViT import: {type(exc).__name__}: {exc}")

    # Tiny CPU checks catch binary ABI / torch-vision mismatches that metadata
    # alone cannot detect. No model, dataset, training, extraction, or FS work.
    if all(name in modules for name in ("numpy", "torch", "torchvision")):
        try:
            np, torch, vision = (modules[name] for name in ("numpy", "torch", "torchvision"))
            torch.from_numpy(np.zeros(1, dtype=np.float32)).numpy()
            vision.ops.nms(torch.zeros((1, 4)), torch.ones(1), 0.5)
            print("NumPy / PyTorch bridge and torchvision CPU operators: OK")
        except Exception as exc:
            errors.append(f"Binary compatibility: {type(exc).__name__}: {exc}")

    if "torch" in modules:
        try:
            torch = modules["torch"]
            available = torch.cuda.is_available()
            print(f"Torch CUDA runtime: {torch.version.cuda}")
            print(f"Torch CUDA available: {available}")
            if available:
                for index in range(torch.cuda.device_count()):
                    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
            else:
                print("GPU: unavailable (CPU environment is supported)")
        except Exception as exc:
            errors.append(f"CUDA inspection: {type(exc).__name__}: {exc}")
    else:
        print("Torch CUDA availability / GPU: unknown (PyTorch import failed)")

    if errors:
        sys.stdout.flush()
        print("\nEnvironment verification FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print("Use this interpreter's -m pip install -r requirements.txt, then -m pip check. "
              "Missing standard-library modules require a complete Python installation.", file=sys.stderr)
        return 1
    print("\nEnvironment verification PASSED. No experiments were run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
