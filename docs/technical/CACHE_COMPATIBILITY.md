Cache compatibility is defined by `build_combination_identity`, separately for
each dataset, classifier, optimizer and transfer function. The selected lists,
EXP number, output paths and plotting/scheduling options never determine reuse.
Prepared feature CSV contents are hashed, so moving identical files is safe and
replacing a CSV in place invalidates its results. MAFESE inputs are identified
by their feature and target values.

The identity also includes runs, epochs, population size, objective and weights,
partition settings, the seed policy, individual optimizer parameters, optimizer
implementation revision and source fingerprints, binary representation and
repair revisions, wrapper science revision, and scientific package versions.
`build_cache_signature` is only a compact display of request settings. It is not
a compatibility key. All actual lookups verify the full combination identity.

New runs are stored in `Results/EXPxxx/cache/combinations_v2/`, in atomic pickle
envelopes containing both `identity` and `row`. Progress and final files are
merged by run ID. Current results take precedence on overlapping IDs; compatible
source results fill gaps. Only missing or invalid run IDs are scheduled. Prepared
features are retained whenever any runs were recovered, including partial reuse.

`REUSE_CACHE_FROM_EXP_ID` is a read-only fallback for each combination. A normal
run imports recovered rows into its own EXP. `FIGURES_ONLY` never writes caches,
prepares features or starts optimizers: it reports missing combinations/run IDs
and exports all compatible available results. If none are available, it raises
an explicit cache-only error. Convergence validation remains strict.

EXP604 migration adds JSON references to original pickle rows, including hashes
of the original files. It does not rewrite any original payload. Unversioned
group caches are never silently accepted by ordinary lookup. The completed
migration's frozen manifest and validation evidence are in
`diagnostics/exp604_cache_migration/`; the bootstrap used a pre-migration snapshot
and checked the historical science and dependency sources. The migration script
is a historical bootstrap for the legacy implementation. Do not regenerate or
migrate EXP604 with the corrected framework: EXP604 remains read-only.

EXP605 now uses the validated shared `transfer-binary-contract-v1` adapter:
MAFESE's requested bounds remain `[-8, 8]`, proposal correction remains
stochastic, and stored masks decode deterministically without consuming RNG.
Every optimizer uses `explicit-selector-seed-v1` (`seed_base + run_id` passed
to MAFESE, then to MEALPY). BRO additionally uses `bro-binary-respawn-v1`, repairing
continuous respawns before evaluation while inheriting `OriginalBRO.evolve`.
The wrapper revision is `mafese-prepared-partitions-corrected-binary-seeded-v2`.
Actual adapter sources are included in identities; BRO's adapter and upstream
inheritance sources are fingerprinted as its executed implementation.

The shared representation and seed corrections change every configured
optimizer's scientific behavior. Thus **none of EXP604's 182 combinations
(3,640 runs) are compatible with corrected EXP605**. There is no special override
to reuse DE, JADE, MaCRO-DE-t or other legacy results. This leaves all 13 original
optimizers pending across seven datasets and KNN/SVM: MaCRO-DE-t, DE, JADE,
SHADE, PSO, GWO, WOA, HHO, BRO, DBO, RUN, FOX and FLA. Prepared MIAFEx feature
CSVs are retained and read directly; training and extraction remain disabled.
Future caches with matching corrected identities are reused normally, including
partial runs; compatible combinations are not scheduled again.

Run the guarded, read-only cache audit from the repository root:

```sh
.venv/bin/python tools/cache/dry_run_exp605_cache.py
```

That command guards optimizer execution, training, extraction and cache writes;
compares frozen EXP604 identities with actual EXP605 requests; checks source
availability and current reuse; and verifies file hashes, membership and
modification times under both experiments. Only diagnostic reports are written,
under `diagnostics/exp605_integration/`. It does not call the experiment entry
point. Histories remain the genuine recorded samples: no smoothing, padding or
artificial decline is introduced.

The authoritative implementation and numerical evidence remain read-only under
`diagnostics/corrected_binary_final/`. Regression checks compare the production
adapter methods with that implementation and verify its dependency/input hashes
and all 15 saved exact fitness/mask results. Per-run history endpoints must now
equal the independent optimizer best exactly, including rejection of one-ULP
damage. Stored mean histories must equal the arithmetic mean of the genuine run
histories exactly. Seed, winning solution, decoded/returned masks, selected
indices, scored feature count and transformed column count are checked before
final evaluation, without an additional classifier fit. EXP604 scientific
execution is rejected before backend setup or output allocation.

These integrity checks do not change valid numerical results, so the validated
binary/seed/BRO revisions above are retained; no validation-only identity bump
invalidates already-corrected compatible caches. BRO's own evolution remains
unchanged, while the shared bounds, decoder and seeding corrections affect every
optimizer's execution relative to EXP604.

To keep a separate audit from previous integration evidence:

```sh
.venv/bin/python tools/cache/dry_run_exp605_cache.py --diagnostic-output diagnostics/exp605_scientific_correction
```

The audit uses its own before/after snapshot rather than a stale temporary
baseline, preserves real source fingerprints while blocking loader execution,
and exports every combination's historical/requested identity digests and reuse
decision in `combination_decisions.csv` and `cache_dry_run.json`.
