"""Prepare the seven local MIAFEx datasets without changing image bytes.

Run: python prepare_all_miafex_datasets.py  (requires Pillow)

Valid existing train/test membership is preserved. Missing or mirrored splits
are deduplicated and split by class with seed 42 (floor(80%) training groups).
Other existing splits are repaired: validation joins training, and test wins
when identical bytes occur in both splits. Invalid layouts retain one copy per
(class, SHA-256). Conflicting labels are retained in the SAME split and reported.
Duplicate detection is byte-exact, not perceptual or patient-level detection.

Chest_CT's three long training class names are mapped to its test class names.
Masks, segmentation files, non-images and unreadable images are excluded.
Each replacement is staged and hash-verified before the originals are removed.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
import shutil
import tempfile

from PIL import Image, UnidentifiedImageError


DATASETS_ROOT = Path(__file__).resolve().parent / "datasets"
DATASETS = (
    "Brain_MRI", "Breast_Ultrasound", "Chest_CT", "Eye_Fundus",
    "Gastrointestinal_Endoscopy", "Histological_Biopsy", "Ocular_Alignment",
)
SEED = 42
SPLITS = ("train", "test")
SOURCE_SPLITS = {"train", "test", "valid", "val", "validation"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
MASK_PATTERN = re.compile(r"(^|[^a-z])(masks?|segmentations?|segmented)([^a-z]|$)")
CHEST_ALIASES = {
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "adenocarcinoma",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "large.cell.carcinoma",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "squamous.cell.carcinoma",
}


@dataclass(frozen=True)
class ImageFile:
    path: Path
    class_name: str
    split: str
    checksum: str


@dataclass
class Plan:
    root: Path
    classes: list[str]
    assignments: list[tuple[ImageFile, str]]
    ignored: Counter
    removed_duplicates: int
    conflicts: dict[str, list[ImageFile]]
    mode: str


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_mask(path):
    return any(MASK_PATTERN.search(part.lower()) for part in path.parts)


def canonical_class(dataset, name):
    return CHEST_ALIASES.get(name, name) if dataset == "Chest_CT" else name


def scan(root):
    if not root.is_dir() or root.is_symlink() or root.parent.is_symlink():
        raise ValueError(f"Expected a real dataset directory: {root}")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError(f"Symbolic links are not supported inside {root}")
    if any(path.name.startswith(".prepare-") for path in root.iterdir()):
        raise ValueError(f"Unfinished preparation in {root}; recover its backup first.")
    classes = set()
    split_classes = {split: set() for split in SPLITS}
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.startswith(".") or is_mask(Path(folder.name)):
            continue
        candidates = list(folder.iterdir()) if folder.name in SOURCE_SPLITS else [folder]
        for candidate in candidates:
            if candidate.is_dir() and not candidate.name.startswith(".") and not is_mask(Path(candidate.name)):
                name = canonical_class(root.name, candidate.name)
                classes.add(name)
                if folder.name in SPLITS:
                    split_classes[folder.name].add(name)
    records, ignored = [], Counter()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if is_mask(relative):
            ignored["mask/segmentation"] += 1
            continue
        if any(part.startswith(".") for part in relative.parts) or path.suffix.lower() not in IMAGE_EXTENSIONS:
            ignored["non-image/hidden"] += 1
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            ignored["invalid image"] += 1
            print(f"  Excluding unreadable image: {path}", flush=True)
            continue
        parts = relative.parts
        split = parts[0] if parts[0] in SOURCE_SPLITS else "unsplit"
        class_index = 0 if split == "unsplit" else 1
        if len(parts) <= class_index + 1:
            raise ValueError(f"Image has no class folder: {path}")
        name = canonical_class(root.name, parts[class_index])
        classes.add(name)
        records.append(ImageFile(path, name, split, digest(path)))
    empty = classes - {record.class_name for record in records}
    if not records or empty:
        raise ValueError(f"{root.name}: no usable images in classes {sorted(empty)}; originals kept.")
    return sorted(classes), split_classes, records, ignored


def make_plan(root):
    classes, split_classes, records, ignored = scan(root)
    groups = defaultdict(list)
    for record in records:
        groups[record.checksum].append(record)
    conflicts = {key: group for key, group in groups.items()
                 if len({record.class_name for record in group}) > 1}
    hashes = {split: {r.checksum for r in records if r.split == split} for split in SPLITS}
    populated = all(any(r.split == split and r.class_name == name for r in records)
                    for split in SPLITS for name in classes)
    valid = (split_classes["train"] == split_classes["test"] == set(classes)
             and populated and not hashes["train"] & hashes["test"]
             and all(r.split in SPLITS for r in records))
    if valid:
        return Plan(root, classes, [(r, r.split) for r in records], ignored, 0,
                    conflicts, "preserved existing split")

    # Select one source for each class/hash, preferring the original test copy.
    priority = {"test": 0, "train": 1, "valid": 2, "val": 2, "validation": 2, "unsplit": 3}
    unique = {}
    for record in sorted(records, key=lambda r: (priority[r.split], str(r.path))):
        unique.setdefault((record.class_name, record.checksum), record)
    mirrored = hashes["train"] == hashes["test"] and bool(hashes["train"])
    rebuild = mirrored or not hashes["train"] or not hashes["test"]
    destinations = {}
    rng = random.Random(SEED)
    if rebuild:
        # A checksum is an indivisible group, including when its labels conflict.
        # Assign shared groups once; their other labels inherit that assignment.
        for name in classes:
            checksums = sorted(key for cls, key in unique if cls == name)
            rng.shuffle(checksums)
            target = min(len(checksums) - 1, max(1, len(checksums) * 4 // 5))
            already_train = sum(destinations.get(key) == "train" for key in checksums)
            pending = [key for key in checksums if key not in destinations]
            train_count = max(0, min(len(pending), target - already_train))
            for index, key in enumerate(pending):
                destinations[key] = "train" if index < train_count else "test"
        mode = "rebuilt stratified 80/20 split (seed 42)"
    else:
        for key, group in groups.items():
            destinations[key] = "test" if any(r.split == "test" for r in group) else "train"
        # Repair missing class membership without moving any group twice.
        for name in classes:
            keys = sorted(key for cls, key in unique if cls == name)
            for missing in SPLITS:
                if any(destinations[key] == missing for key in keys):
                    continue
                candidates = [key for key in keys if all(
                    sum(destinations[k] == destinations[key] for c, k in unique if c == r.class_name) > 1
                    for r in groups[key])]
                if not candidates:
                    raise ValueError(f"{root.name}/{name}: cannot populate both splits without leakage.")
                destinations[rng.choice(candidates)] = missing
        mode = "repaired existing split (test copies retained; validation joins train)"
    assignments = [(record, destinations[record.checksum]) for record in unique.values()]
    counts = Counter((split, record.class_name) for record, split in assignments)
    if any(not counts[split, name] for split in SPLITS for name in classes):
        raise ValueError(f"{root.name}: insufficient independent images to populate every class in both splits.")
    return Plan(root, classes, assignments, ignored, len(records) - len(unique), conflicts, mode)


def validate_tree(root, plan):
    expected = Counter((split, record.class_name, record.checksum) for record, split in plan.assignments)
    actual, hashes = Counter(), {split: set() for split in SPLITS}
    for split in SPLITS:
        folder = root / split
        if {p.name for p in folder.iterdir()} != set(plan.classes):
            raise ValueError(f"Class folders do not match in {folder}")
        for name in plan.classes:
            for path in (folder / name).iterdir():
                if not path.is_file():
                    raise ValueError(f"Unexpected non-file: {path}")
                checksum = digest(path)
                actual[split, name, checksum] += 1
                hashes[split].add(checksum)
    if actual != expected or hashes["train"] & hashes["test"]:
        raise ValueError(f"Image preservation or split overlap validation failed: {root}")


def install(plan):
    root = plan.root
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
    prepared, backup = staging / "prepared", staging / "originals"
    try:
        for split in SPLITS:
            for name in plan.classes:
                (prepared / split / name).mkdir(parents=True)
        used = defaultdict(set)
        for record, split in plan.assignments:
            folder = prepared / split / record.class_name
            filename = record.path.name
            index = 0
            while filename.casefold() in used[folder]:
                index += 1
                filename = f"{record.checksum}_{index}{record.path.suffix.lower()}"
            used[folder].add(filename.casefold())
            with record.path.open("rb") as source, (folder / filename).open("xb") as target:
                shutil.copyfileobj(source, target)
        validate_tree(prepared, plan)
    except BaseException:
        shutil.rmtree(staging)
        raise

    backup.mkdir()
    moved, installed = [], []
    try:
        for child in sorted(root.iterdir()):
            if child != staging:
                child.rename(backup / child.name)
                moved.append(child.name)
        for split in SPLITS:
            (prepared / split).rename(root / split)
            installed.append(split)
        validate_tree(root, plan)
    except BaseException:
        print(f"Replacement failed. Original files are backed up in {backup}", flush=True)
        # If rollback fails, retain staging and all originals for recovery.
        for split in reversed(installed):
            (root / split).rename(prepared / split)
        for name in reversed(moved):
            (backup / name).rename(root / name)
        shutil.rmtree(staging)
        raise
    shutil.rmtree(staging)


def report(plan):
    counts = Counter((split, record.class_name) for record, split in plan.assignments)
    print(f"\n{plan.root.name}: {plan.mode}")
    print(f"{'Class':<28} {'Train':>7} {'Test':>7} {'Total':>7}")
    for name in plan.classes:
        train, test = counts["train", name], counts["test", name]
        print(f"{name:<28} {train:>7} {test:>7} {train + test:>7}")
    train = sum(counts["train", name] for name in plan.classes)
    test = sum(counts["test", name] for name in plan.classes)
    print(f"{'TOTAL':<28} {train:>7} {test:>7} {train + test:>7}")
    print(f"Excluded: {dict(plan.ignored)}; redundant copies removed: {plan.removed_duplicates}")
    for checksum, records in sorted(plan.conflicts.items()):
        labels = sorted({r.class_name for r in records})
        split = next(split for r, split in plan.assignments if r.checksum == checksum)
        print(f"Label conflict (retained together in {split}): {checksum}, classes={labels}")
    print("Validated matching classes, original image bytes, and zero SHA-256 overlap.", flush=True)


def main():
    plans = []
    # Preflight every dataset before changing any of them.
    for name in DATASETS:
        print(f"Inspecting {name}...", flush=True)
        plans.append(make_plan(DATASETS_ROOT / name))
    for plan in plans:
        print(f"Preparing {plan.root.name}...", flush=True)
        install(plan)
        report(plan)
    print(f"\nAll {len(plans)} datasets prepared successfully.")


if __name__ == "__main__":
    main()
