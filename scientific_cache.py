"""Atomic per-combination storage; legacy migration only adds reference metadata.

No automatic trust of unversioned pickle filenames. A migrated reference binds
the full scientific identity to one original row and a SHA-256 checked file.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from functools import lru_cache


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=512)
def _file_digest(path, size, mtime, ctime):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".metadata-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def combination_files(paths, identity):
    prefix = Path(paths.cache_dir) / "combinations_v2" / identity_digest(identity)
    return Path(str(prefix) + "_results.pkl"), Path(str(prefix) + "_progress.pkl")


def reference_file(paths, identity):
    return combination_files(paths, identity)[0].with_suffix(".json")


def read_candidates(paths, identity, *, use_final=True):
    final, progress = combination_files(paths, identity)
    for path in ([progress, final] if use_final else [progress]):
        if not path.exists():
            continue
        try:
            with path.open("rb") as stream:
                entry = pickle.load(stream)
            if entry["identity"] != identity:
                raise ValueError("scientific identity mismatch")
            yield entry["row"]
        except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            print(f"[cache-warning] {path}: {exc}", flush=True)
    reference = reference_file(paths, identity)
    if use_final and reference.exists():
        try:
            entry = json.loads(reference.read_text())
            if entry["identity"] != identity:
                raise ValueError("scientific identity mismatch")
            for origin in entry["sources"]:
                # Only filenames in this EXP's cache; no arbitrary path traversal.
                name = origin["file"]
                if Path(name).name != name:
                    raise ValueError("invalid reference filename")
                original = Path(paths.cache_dir) / name
                if file_digest(original) != origin["sha256"]:
                    raise ValueError(f"original payload changed: {name}")
                with original.open("rb") as stream:
                    yield pickle.load(stream)[origin["label"]]
        except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            print(f"[cache-warning] {reference}: {exc}", flush=True)
