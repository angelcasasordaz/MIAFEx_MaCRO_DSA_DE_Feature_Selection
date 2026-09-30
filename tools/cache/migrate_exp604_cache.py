"""Explicit, metadata-only migration and no-execution EXP604 reporting audit.

Run once with --migrate, then use --verify to repeat the preservation checks.
The frozen manifest records the original file hashes and scientific identities;
rerunning migration never relabels old results with new implementation versions.
"""
import argparse
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

import main_best as framework
import scientific_cache as cache


AUDIT = ROOT / "diagnostics/exp604_cache_migration"
MANIFEST = AUDIT / "manifest.json"


def no_execution(stack):
    for name in ("run_single", "execute_pending_runs", "build_optimizer", "train_miafex",
                 "extract_miafex_features", "resolve_miafex_csv"):
        stack.enter_context(patch.object(framework, name, side_effect=AssertionError(f"Forbidden execution: {name}")))


def tables():
    return {p.name: pd.read_excel(p, sheet_name=None) for p in (ROOT / "Results/EXP604").glob("*.xlsx")}


def migrate():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    # This baseline was captured from EXP604 before any migration or reporting.
    with open("/tmp/exp604_before.pkl", "rb") as stream:
        before = pickle.load(stream)
    args = framework.parse_args([])
    args.optimizers = framework.resolve_optimizers(args)
    assert (args.exp_id, args.reuse_cache_from_exp_id, args.figures_only) == (604, None, True)
    assert len(args.optimizers) == 13 and "MaCRO-DE-t" in args.optimizers
    paths = framework.make_paths(args, create=False)
    scoped = framework.resolve_miafex_dataset_args(args)
    # Verify historical scientific functions against the preexisting audit snapshot.
    def science_nodes(path):
        names = {"run_single", "load_miafex_feature_data", "RobustClassificationFeatureSelectionProblem", "build_optimizer"}
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    assert science_nodes(ROOT / "main_best.py") == science_nodes(
        ROOT / "diagnostics/corrected_binary_final/main_before.py")
    for line in (ROOT / "diagnostics/exp604_convergence_audit/dependencies.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        assert cache.file_digest(ROOT / path) == digest
    records = []
    for dataset, selected in scoped.items():
        signature = framework.build_legacy_group_signature(selected)
        assert signature == before["signatures"][dataset], dataset
        for estimator in args.estimators:
            sources = framework.cache_files(paths, dataset, estimator, signature)
            rows = []
            for source in sources:
                expected, mtime = before["files"][str(Path(source))]
                assert cache.file_digest(source) == expected
                assert Path(source).stat().st_mtime_ns == mtime
                rows.append(framework.load_cache(source))
                framework.report_cached_convergence(rows[-1], args.epochs, source, figures_only=True)
            for method in args.optimizers:
                for transfer in args.transfer_functions:
                    label = framework.build_alg_label(method, transfer, estimator, False, True)
                    identity = framework.build_combination_identity(selected, dataset, estimator, method, transfer)
                    assert all(set(framework.completed_run_ids(payload[label])) == set(range(args.runs)) for payload in rows)
                    # Both existing files must agree on every row; preserve either verbatim.
                    assert pickle.dumps(rows[0][label]) == pickle.dumps(rows[1][label])
                    entry = {"identity": identity, "sources": [
                        {"file": Path(source).name, "sha256": before["files"][str(Path(source))][0], "label": label}
                        for source in sources], "migration": "EXP604-authoritative-original-files-v1"}
                    records.append(entry)
    manifest = {"experiment": 604, "combinations": len(records), "runs": len(records) * args.runs,
                "original_files": {Path(p).name: {"sha256": digest, "mtime_ns": mtime}
                                   for p, (digest, mtime) in before["files"].items()}, "records": records}
    AUDIT.mkdir(parents=True, exist_ok=True)
    # Preserve normalized reporting baselines so verification remains repeatable.
    with (AUDIT / "report_baseline.pkl").open("wb") as stream:
        pickle.dump({"tables": before["tables"], "csv": before["csv"]}, stream)
    cache.atomic_json(MANIFEST, manifest)
    return manifest


def verify(manifest):
    args = framework.parse_args([])
    args.optimizers = framework.resolve_optimizers(args)
    paths = framework.make_paths(args, create=False)
    for filename, expected in manifest["original_files"].items():
        path = Path(paths.cache_dir) / filename
        assert cache.file_digest(path) == expected["sha256"], filename
        assert path.stat().st_mtime_ns == expected["mtime_ns"], filename
    for record in manifest["records"]:
        reference = cache.reference_file(paths, record["identity"])
        if not reference.exists():
            cache.atomic_json(reference, record)
        assert json.loads(reference.read_text()) == record
        rows = list(cache.read_candidates(paths, record["identity"]))
        assert rows
        framework.report_cached_convergence({"original": rows[0]}, args.epochs, str(reference), figures_only=True)
    return args, paths


def regenerate(args):
    with ExitStack() as stack:
        no_execution(stack)
        stack.enter_context(patch.object(framework, "parse_args", return_value=args))
        framework.main()
    return compare_reports()


def compare_reports():
    with (AUDIT / "report_baseline.pkl").open("rb") as stream:
        baseline = pickle.load(stream)
    current = tables()
    checked_sheets = 0
    for path, sheets in baseline["tables"].items():
        actual = current[Path(path).name]
        assert actual.keys() == sheets.keys()
        for sheet, expected in sheets.items():
            pd.testing.assert_frame_equal(actual[sheet], expected, check_exact=True)
            checked_sheets += 1
    assert (ROOT / "Results/EXP604/RESUMEN_GRAFICAS_EXP604.csv").read_bytes() == baseline["csv"]
    return checked_sheets


def reuse_checks(args, paths, manifest):
    """Exercise EXP605 reads against real EXP604 without creating EXP605 files."""
    scoped = framework.resolve_miafex_dataset_args(args)
    destination = framework.make_paths(args, exp_id=605, create=False)
    found = 0
    for record in manifest["records"]:
        identity = record["identity"]
        selected = argparse.Namespace(**vars(scoped[identity["dataset"]]))
        selected.optimizers = [identity["optimizer"]]
        selected.estimators = [identity["classifier"]]
        selected.exp_id = 605
        selected.reuse_cache_from_exp_id = 604
        assert framework.build_combination_identity(selected, identity['dataset'], identity['classifier'],
                                                    identity['optimizer'], identity['transfer_function']) == identity
        payload = framework.resolve_cached_payload(destination, selected, identity['dataset'],
                                                   identity['classifier'], figures_only=True)
        assert len(payload) == 1
        original = next(cache.read_candidates(paths, identity))
        assert pickle.dumps(next(iter(payload.values()))) == pickle.dumps(original)
        found += 1
    subset_count = superset_count = 0
    for dataset, selected in scoped.items():
        selected = argparse.Namespace(**vars(selected))
        selected.exp_id, selected.reuse_cache_from_exp_id = 605, 604
        for classifier in args.estimators:
            selected.optimizers = ['PSO', 'BRO']
            subset = framework.resolve_cached_payload(destination, selected, dataset, classifier, figures_only=True)
            assert len(subset) == 2
            subset_count += len(subset)
            selected.optimizers = ['DE', 'JADE', 'SHADE', 'PSO', 'GWO', 'WOA', 'HHO',
                                   'DSADE', 'MaCRO-DE', 'OriginalDMOA']
            superset = framework.resolve_cached_payload(destination, selected, dataset, classifier, figures_only=True)
            assert len(superset) == 7
            superset_count += len(superset)
            selected.optimizers = ['PSO', 'BRO', 'DE']
            with patch.dict(framework.BINARY_REPRESENTATION_REVISIONS,
                            {'OriginalPSO': 'corrected-v2', 'OriginalBRO': 'corrected-v2'}):
                corrected = framework.resolve_cached_payload(destination, selected, dataset, classifier, figures_only=True)
                assert len(corrected) == 1 and next(iter(corrected)).startswith('DE')
    return {"individually_discovered": found, "cross_exp_subset_reused": subset_count,
            "subset_per_dataset_classifier": "2/2", "superset_per_dataset_classifier": "7/10",
            "cross_exp_superset_reused": superset_count, "corrected_pso_bro_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    options = parser.parse_args()
    with ExitStack() as stack:
        no_execution(stack)
        manifest = migrate() if options.migrate else json.loads(MANIFEST.read_text())
        args, paths = verify(manifest)
        sheets = regenerate(args) if options.regenerate else compare_reports()
        reuse = reuse_checks(args, paths, manifest)
        verify(manifest)
    result = {"combinations": manifest["combinations"], "runs": manifest["runs"],
              "original_files_unchanged": len(manifest["original_files"]),
              "optimizer_executions": 0, "identical_report_sheets": sheets,
              "summary_csv_byte_identical": True,
              "figures": len(list((ROOT / "Figures/EXP604").glob("*.png"))), "reuse_tests": reuse}
    cache.atomic_json(AUDIT / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
