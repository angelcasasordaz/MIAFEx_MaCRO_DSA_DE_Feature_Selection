# Reusable MIAFEx artifact versions

`EXP_ID` identifies downstream feature-selection results. `MIAFEX_ARTIFACT_TAG`
identifies the neural checkpoint and fixed prepared train/test feature pair.
The default tag is `exp607`, preserving the existing directories exactly:

```text
checkpoints/miafex_exp607/<dataset>/miafex_checkpoint.pth
datasets_features/miafex_exp607/<dataset>/train_features.csv
datasets_features/miafex_exp607/<dataset>/test_features.csv
artifact_provenance/miafex_exp607/<dataset>/provenance.json
```

Changing only `EXP_ID` to 608, 609 or 610 keeps these paths. The default train
and extract switches are now `auto`, so later experiments reuse validated
artifacts. `RUNS=20` still controls downstream FS repetitions. Generation
occurs once per dataset before those repetitions; no FS run retrains the net.

The user-editable settings in `main_best.py` are `MIAFEX_ARTIFACT_TAG` and the
three derived roots. CLI equivalents are `--miafex-artifact-tag`,
`--miafex-checkpoint-root`, `--feature-dataset-root`, and
`--miafex-provenance-root`. Explicit root, checkpoint-directory and CSV
overrides remain available, but must match the registered metadata for reuse.
An explicit root override is independent of the CLI tag.

For example, reuse the preserved version with a new downstream experiment:

```bash
python main_best.py --exp-id 608 --miafex-artifact-tag exp607 \
  --train-miafex auto --extract-miafex auto
```

For intentional new neural training, choose an unused tag and the intended
neural settings before running. For example:

```bash
python main_best.py --pipeline-mode extract --miafex-artifact-tag paperlike_v2 \
  --miafex-epochs 60 --train-miafex yes --extract-miafex yes
```

These are usage examples; no experiment was launched during the refactor.

| Switch | `auto` | `yes` | `no` |
| --- | --- | --- | --- |
| Train | Reuse validated checkpoint; train if never published | Request new training; reject a populated version | Never train; require checkpoint if extracting |
| Extract | Reuse validated pair; extract if never published | Request extraction; reject existing features | Never extract; require existing pair |

Feature-selection mode always requires a validated existing pair and never
trains or extracts, regardless of switches. Checkpoints and features are
immutable after publication. Even compatible retraining requires a new tag,
because the unchanged trainer has no explicit neural seed and would generate
a different realization. A missing or changed *registered* artifact is an
integrity failure, not permission to regenerate it. A checkpoint whose feature
pair has never been published can still be used for first extraction.

Each provenance file is UTF-8 JSON with `schema_version: 1`, containing:

- `MIAFEX_ARTIFACT_TAG`.
- `configuration`: dataset name/root, backbone/model, pretrained source,
  optimizer, epochs, batch size, learning rate, transforms, extraction
  representation, neural seed policy, train/test image counts and class lists,
  partition content/name digests, neural source SHA-256 and package versions.
- `paths`: absolute checkpoint and both feature CSV paths.
- `artifact_sha256`: published checkpoint and feature CSV digests.
- `provenance`: origin, recording/creation time, source Git revision and
  historical evidence or limitations. The originating EXP ID is informational
  and never enters artifact compatibility.

Metadata is separate from checkpoint/feature directories to leave **every
existing EXP607 file and directory untouched**. New configurations reserve
metadata before training; successful checkpoint and feature publication add
their hashes atomically to that sidecar. A per-dataset lock prevents concurrent
preparation. Both feature partitions are staged and checked before publication.
After an interrupted process, a stale `provenance.lock` fails loudly; remove it
only after verifying that its preparation process is no longer running.

Reuse compares the requested configuration and registered paths, then checks
checkpoint/CSV SHA-256. Available images are validated by class, split,
relative filename and content, so count-preserving source changes also fail.
Neural source changes conservatively invalidate reuse, including changes to
optimizer, model, transforms or representation. Package changes also fail
rather than silently assuming library defaults remain equivalent. Configuration
labels describe the current unchanged neural implementation; future science
changes must update these descriptions along with the neural source.

Feature-only operation allows offline images at the recorded source location,
using the stored image fingerprint. Partially missing online partitions fail.
Unversioned files are rejected: the system never manufactures provenance by
assuming they came from whatever configuration is currently requested.

The seven EXP607 sidecars were explicitly reconstructed after a read-only
completion audit: all 154 requested combinations had 20 valid completed runs
(3,080 FS runs), and cached feature hashes matched the existing CSVs. The
records reference preserved configuration at Git revision
`f5c16eeeda81e06713ab9f97e241d1f4f6118a04`, unchanged neural source files,
50-epoch training metrics, matching train/test class maps and image/CSV counts.
These are reconstructed records, not original training-time manifests. Original
neural RNG state and resolved pretrained revision are unknown; source-image
fingerprints and package versions were captured during migration. These limits
are explicitly recorded without claiming unavailable historical evidence.

The refactor does not change neural optimization, FS science, classifiers,
transfer functions, seeds, repetitions, FS iterations, population, partitions
or scientific-cache identity/storage.
