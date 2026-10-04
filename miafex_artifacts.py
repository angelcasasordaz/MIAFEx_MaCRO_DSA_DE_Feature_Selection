"""Version/provenance management only; no neural or feature-selection science."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
NEURAL_SOURCES = ("miafex_model.py", "train_miafex.py", "extract_miafex_features.py")


def validate_tag(tag):
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("MIAFEX_ARTIFACT_TAG must be a nonempty directory-safe name (e.g. paperlike_v2).")
    return tag


def resolve_roots(args):
    """Explicit path overrides remain supported; EXP_ID never enters these paths."""
    tag = validate_tag(args.miafex_artifact_tag)
    args.miafex_checkpoint_root = args.miafex_checkpoint_root or f"checkpoints/miafex_{tag}"
    args.feature_dataset_root = args.feature_dataset_root or f"datasets_features/miafex_{tag}"
    args.miafex_provenance_root = args.miafex_provenance_root or f"artifact_provenance/miafex_{tag}"
    return args


@lru_cache(maxsize=131072)
def _digest(path, size, mtime, ctime):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def image_source(dataset_root):
    """Bind content, relative filenames, labels and prepared partition membership."""
    source = {}
    for split in ("train", "test"):
        root = Path(dataset_root) / split
        if not root.is_dir():
            raise FileNotFoundError(f"Image partition unavailable: {root}")
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        digest = hashlib.sha256()
        count = 0
        for name in classes:
            for folder, _, files in sorted(os.walk(root / name, followlinks=True)):
                for filename in sorted(files):
                    path = Path(folder) / filename
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        record = [path.relative_to(root).as_posix(), file_digest(path)]
                        digest.update(json.dumps(record, separators=(",", ":")).encode())
                        digest.update(b"\n")
                        count += 1
        source[split] = {"image_count": count, "classes": classes, "sha256": digest.hexdigest()}
    return source


def requested_configuration(args, stored=None):
    root = os.path.realpath(args.dataset_root)
    # Feature-only reuse can work with offline images, but the recorded source
    # location must still match. Online partitions always get content validation.
    if all((Path(root) / split).is_dir() for split in ("train", "test")):
        source = image_source(root)
    elif stored and args.pipeline_mode == "feature_selection" and not Path(root).exists():
        source = stored["configuration"]["image_source"]
    else:
        raise FileNotFoundError(f"Cannot validate MIAFEx image source: {root}")
    code_root = Path(__file__).resolve().parent
    return {
        "dataset_name": args.dataset_name,
        "dataset_root": root,
        "backbone_model": "MIAFEx: ViTForImageClassification + element-wise refinement + Linear(768, num_classes)",
        "pretrained_source": "google/vit-base-patch16-224-in21k",
        "optimizer": "torch.optim.NAdam (library defaults except learning_rate)",
        "epochs": int(args.miafex_epochs),
        "batch_size": int(args.miafex_batch_size),
        "learning_rate": float(args.miafex_learning_rate),
        "preprocessing_transforms": {
            "loader": "torchvision.datasets.ImageFolder (RGB)",
            "train": ["Resize((224, 224), torchvision defaults)", "ToTensor()"],
            "test": ["Resize((224, 224), torchvision defaults)", "ToTensor()"],
        },
        "extraction_representation": "last hidden-state CLS token * learned refinement_weights; 768 dimensions; eval/no_grad",
        "training_seed": None,
        "training_seed_policy": "No explicit neural seed set by train_miafex; RNG state was not recorded",
        "image_source": source,
        "source_sha256": {name: file_digest(code_root / name) for name in NEURAL_SOURCES},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers")},
    }


def artifact_paths(args):
    return {
        "checkpoint": os.path.abspath(os.path.join(args.miafex_output, "miafex_checkpoint.pth")),
        "train_features_csv": os.path.abspath(args.train_features_csv),
        "test_features_csv": os.path.abspath(args.test_features_csv),
    }


def metadata_path(args):
    return Path(args.miafex_provenance_root) / args.dataset_name / "provenance.json"


def fail(message):
    raise ValueError(f"MIAFEx artifact validation failed: {message}. Use a new MIAFEX_ARTIFACT_TAG for a different configuration; existing versions are immutable.")


def read_metadata(args):
    path = metadata_path(args)
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION:
            fail(f"unsupported metadata schema in {path}")
        if metadata["MIAFEX_ARTIFACT_TAG"] != args.miafex_artifact_tag:
            fail(f"artifact tag differs in {path}")
        for key in ("configuration", "paths", "artifact_sha256", "provenance"):
            if not isinstance(metadata[key], dict):
                fail(f"invalid {key} in {path}")
        return metadata
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"invalid provenance {path}: {exc}")


def validate_existing(args, *, report=False):
    """Read-only preflight, including feature_selection and cache-complete runs."""
    paths = artifact_paths(args)
    stored = read_metadata(args)
    if stored is None:
        if any(Path(path).exists() for path in paths.values()):
            fail(f"existing files have no trusted provenance at {metadata_path(args)}; never infer legacy settings from the requested configuration")
        return None
    requested = requested_configuration(args, stored)
    differences = [key for key in requested if requested[key] != stored["configuration"].get(key)]
    differences += [key for key in stored["configuration"] if key not in requested]
    if differences:
        fail(f"configuration mismatch for {args.dataset_name}: {', '.join(differences)}")
    if paths != stored["paths"]:
        fail(f"checkpoint/feature paths differ for {args.dataset_name}")
    for key, path in paths.items():
        expected = stored["artifact_sha256"].get(key)
        if expected is None:
            if Path(path).exists():
                fail(f"unregistered {key}: {path}")
        elif not Path(path).is_file() or file_digest(path) != expected:
            fail(f"missing or changed {key}: {path}")
    if report:
        print(f"[provenance] {args.dataset_name}: VALIDATED {metadata_path(args)}")
        print(json.dumps(stored, indent=2, sort_keys=True))
        if stored["artifact_sha256"].get("checkpoint"):
            print(f"[checkpoint] {args.dataset_name}: REUSE CHECKPOINT -> {paths['checkpoint']}")
        if all(stored["artifact_sha256"].get(f"{split}_features_csv") for split in ("train", "test")):
            print(f"[features] {args.dataset_name}: REUSE FEATURES -> {paths['train_features_csv']}, {paths['test_features_csv']}")
    return stored


def new_metadata(args, *, provenance=None):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "schema_version": SCHEMA_VERSION,
        "MIAFEX_ARTIFACT_TAG": validate_tag(args.miafex_artifact_tag),
        "configuration": requested_configuration(args),
        "paths": artifact_paths(args),
        "artifact_sha256": {},
        "provenance": provenance or {
            "origin": "generated_with_version_management",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_by_exp_id": args.exp_id,
            "source_git_revision": revision,
            "pretrained_resolved_revision": None,
            "pretrained_revision_note": "The unchanged model loader requests the source's default revision; no resolved revision is recorded by the trainer.",
        },
    }


def save_metadata(args, metadata):
    """Atomic sidecar publication; never touches a scientific cache."""
    path = metadata_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".provenance-", delete=False) as stream:
            temporary = stream.name
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def record_outputs(args, metadata):
    metadata["artifact_sha256"] = {key: file_digest(path) for key, path in artifact_paths(args).items()
                                   if Path(path).is_file()}
    save_metadata(args, metadata)


@contextmanager
def artifact_lock(args):
    """Serialize publication for a tag/dataset, including initial creation."""
    path = metadata_path(args).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fail(f"another preparation (or an interrupted preparation) holds {path}")
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()
