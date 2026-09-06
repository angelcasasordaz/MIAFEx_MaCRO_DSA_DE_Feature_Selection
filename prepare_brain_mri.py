"""Rebuild only datasets/Brain_MRI with a reproducible, per-class 80/20 split.

Run: python prepare_brain_mri.py
Requires Pillow (already used by the project). Images are copied byte-for-byte.
Training gets floor(0.8 * class_size); testing gets the remaining images.
Content-based names retain duplicate files and make subsequent runs stable.
"""

from collections import Counter
import hashlib
from pathlib import Path
import random
import shutil
import tempfile

from PIL import Image, UnidentifiedImageError


DATASET = Path(__file__).resolve().parent / "datasets" / "Brain_MRI"
CLASSES = ("glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor")
SPLITS = ("train", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SEED = 42


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_images():
    """Read both splits completely before changing anything on disk."""
    if DATASET.is_symlink() or DATASET.parent.is_symlink():
        raise ValueError("Dataset paths must not be symbolic links.")
    images = {name: [] for name in CLASSES}
    ignored = 0
    for split in SPLITS:
        folder = DATASET / split
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"Expected a real directory: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Refusing to replace a tree containing a symlink: {path}")
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored += 1
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except UnidentifiedImageError:
                ignored += 1
                continue
            # Other read/validation errors abort, preserving the original trees.
            relative = path.relative_to(folder)
            if len(relative.parts) < 2 or relative.parts[0] not in CLASSES:
                raise ValueError(f"Image outside the four expected classes: {path}")
            images[relative.parts[0]].append((digest(path), path.suffix.lower(), path))
    for name, records in images.items():
        if not records:
            raise ValueError(f"No images found for {name}; original folders kept.")
    return images, ignored


def main():
    images, ignored = collect_images()
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=DATASET))
    counts = {}
    try:
        rng = random.Random(SEED)
        for name, records in images.items():
            # Hash order is independent of the old split and filenames.
            records.sort()
            occurrences = Counter()
            named = []
            for checksum, suffix, source in records:
                occurrences[checksum, suffix] += 1
                filename = f"{checksum}_{occurrences[checksum, suffix]:04d}{suffix}"
                named.append((source, filename, checksum))
            rng.shuffle(named)
            train_count = len(named) * 4 // 5
            counts[name] = (train_count, len(named) - train_count)
            for split, selected in zip(SPLITS, (named[:train_count], named[train_count:])):
                destination = staging / split / name
                destination.mkdir(parents=True)
                for source, filename, checksum in selected:
                    target = destination / filename
                    # Exclusive creation prevents accidental overwrites.
                    with source.open("rb") as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    if digest(target) != checksum:
                        raise OSError(f"Copy verification failed: {source}")
    except BaseException:
        shutil.rmtree(staging)
        raise

    moved, installed = [], []
    try:
        for split in SPLITS:
            (DATASET / split).rename(staging / f"original_{split}")
            moved.append(split)
        for split in SPLITS:
            (staging / split).rename(DATASET / split)
            installed.append(split)
    except BaseException:
        # If rollback itself fails, staging (including originals) is retained.
        print(f"Replacement failed; original folders are backed up in {staging}")
        for split in reversed(installed):
            (DATASET / split).rename(staging / split)
        for split in reversed(moved):
            (staging / f"original_{split}").rename(DATASET / split)
        shutil.rmtree(staging)
        raise
    shutil.rmtree(staging)

    print(f"Brain_MRI: seed={SEED}, 80/20 per class; ignored {ignored} non-image files")
    print(f"{'Class':<22} {'Train':>7} {'Test':>7} {'Total':>7}")
    for name, (train, test) in counts.items():
        print(f"{name:<22} {train:>7} {test:>7} {train + test:>7}")
    train = sum(count[0] for count in counts.values())
    test = sum(count[1] for count in counts.values())
    print(f"{'TOTAL':<22} {train:>7} {test:>7} {train + test:>7}")


if __name__ == "__main__":
    main()
