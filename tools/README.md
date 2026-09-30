# Preserved utilities

Run utilities from the repository root using the project's interpreter.
These scripts are separate from the active pipeline and are not setup steps.

| Utility | Purpose |
|---|---|
| `cache/dry_run_exp605_cache.py` | Guarded cache audit tied to the historical EXP605 configuration; requires local migration evidence and writes only diagnostic reports |
| `cache/migrate_exp604_cache.py` | Historical EXP604 metadata bootstrap and preservation verification; depends on the original snapshots and configuration |
| `legacy/prepare_brain_mri.py` | Original Brain MRI file-level split rebuild, retained to document and reproduce the older preparation workflow |

The EXP604 migration bootstrap is preserved as historical source. Do not rerun
its migration against the current corrected framework or existing EXP604 data.
See [cache compatibility notes](../docs/technical/CACHE_COMPATIBILITY.md) for
its original prerequisites and preservation constraints.

Use `prepare_all_miafex_datasets.py` for current dataset preparation. The legacy
Brain MRI script always rebuilds its split and can place duplicate image bytes
in different partitions; its behavior differs from the general preparer.
