"""Focused authoritative v2 parity, identity and cache isolation tests."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mealpy import FloatVar, TransferBinaryVar

import algorithm_acronym_list as registry
import main_best as f
import scientific_cache as cache
from macro_de_t_v2_source import macro_de_t_backend, transport
from macro_de_t_v2_source.macro_de_t_optimizer import MaCRO_DE_t as SuppliedParent


ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    "macro_de_t_backend": "360cf7f7566cf892e8c7ba9c73ab81e82bc50f31dfeddb98b5d04ec55b745354",
    "macro_de_t_optimizer": "5362cb24be130a7798e85bfdf6d3b3c710044c91f57a8f0348a73059654a677e",
    "macro_de_t_v2_optimizer": "2cea43e8c56897565bbfd73c01bb50084f76b7c1e3079f9f7b90184f1b3723d2",
}


def original_source(name):
    source = (ROOT / "macro_de_t_v2_source" / f"{name}.py").read_text()
    return (source.replace("from .macro_de_t_optimizer import", "from macro_de_t_optimizer import")
            .replace("from .macro_de_t_backend import", "from macro_de_t_backend import")
            .replace("from .transport import", "from diversity_gpu_batching import"))


def load_reference():
    modules = {}
    service = ModuleType("diversity_gpu_batching")

    class ReferenceBatcher:
        def __init__(self, *args):
            self.backend = SimpleNamespace(xp=np, to_cpu=np.asarray)

        def macro_de_t_covariance(self, operation, *args):
            kernels = modules["macro_de_t_backend"].CECCovarianceKernels(self.backend)
            return getattr(kernels, operation)(*args)

    service.DiversityMathBatcher = ReferenceBatcher
    with patch.dict(sys.modules, {"diversity_gpu_batching": service}):
        for name, digest in HASHES.items():
            source = original_source(name)
            if hashlib.sha256(source.encode()).hexdigest() != digest:
                raise AssertionError(f"Authoritative source changed: {name}")
            module = modules[name] = ModuleType(name)
            sys.modules[name] = module
            exec(compile(source, name + ".py", "exec"), module.__dict__)
    return modules["macro_de_t_v2_optimizer"].MaCRO_DE_t_v2, service


Reference, ReferenceService = load_reference()


class MaCROV2Tests(unittest.TestCase):
    def args(self):
        return f.parse_args(["--dataset-source", "mafese", "--optimizers",
                             "DSADE", "MaCRO-DE-t", "MaCRO-DE-t-v2",
                             "--epochs", "3", "--pop-size", "10", "--runs", "2"])

    def identity(self, args, method):
        with patch.object(f, "get_dataset", return_value=f.Data([[1], [2]], [0, 1])):
            return f.build_combination_identity(args, "Tiny", "knn", method, "vstf_01")

    def test_authoritative_hashes_inheritance_revision_and_defaults(self):
        for name, digest in HASHES.items():
            self.assertEqual(hashlib.sha256(original_source(name).encode()).hexdigest(), digest)
        # Existing MaCRO-DE-t source stays byte-identical for historical identities.
        self.assertEqual(cache.file_digest(ROOT / "macro_de_t_optimizer.py"),
                         "a0f415276c19bc32d55bff226da698d906d5286bf2017c6c3d99d37cc23e36d8")
        cls = f.MaCRO_DE_t_v2
        self.assertIs(cls.__bases__[0], SuppliedParent)
        self.assertEqual(cls.CANONICAL_NAME, "MaCRO-DE-t-v2")
        self.assertEqual(cls.IMPLEMENTATION_REVISION, "macro-d-scaled-coordinate-adaptive-pcr-v5")
        self.assertEqual(cls.SCIENTIFIC_PARAMETERS,
                         ("epoch", "pop_size", "beta_min", "beta_max", "mahalanobis_q"))
        model = f.build_optimizer(cls.CANONICAL_NAME, self.args())
        self.assertEqual((model.beta_min, model.beta_max, model.mahalanobis_q), (.10, .60, .50))
        self.assertEqual(model.compute_device, "cpu")
        for key in ("pcr", "wf", "cr"):
            with self.assertRaises(TypeError):
                cls(**{key: .2})

    def test_three_independent_resolution_registry_and_mafese_binding(self):
        args = self.args()
        self.assertEqual(f.resolve_optimizers(args), args.optimizers)
        with patch.object(f, "OPTIMIZERS", args.optimizers):
            self.assertEqual(f.resolve_optimizers(f.parse_args([])), args.optimizers)
        for name, cls in zip(args.optimizers, (f.DSADE, f.MaCRO_DE_t, f.MaCRO_DE_t_v2)):
            self.assertEqual(registry.resolve_optimizer_name(name.lower()), name)
            self.assertEqual(registry.optimizer_acronym(name), name)
            self.assertIn(name, registry.CUSTOM_OPTIMIZERS)
            model = f.build_optimizer(name, args)
            self.assertIs(type(model), cls)
            selector = f.MhaSelector(problem="classification", estimator="knn", optimizer=model)
            self.assertIs(selector._set_optimizer(model), model)
        self.assertEqual(f.optimizer_implementation_revisions(args), {
            name: cls.IMPLEMENTATION_REVISION
            for name, cls in zip(args.optimizers, (f.DSADE, f.MaCRO_DE_t, f.MaCRO_DE_t_v2))
        })
        self.assertEqual(len({f.optimizer_plot_color(name) for name in args.optimizers}), 3)
        self.assertNotEqual(f.optimizer_plot_style(args.optimizers[1]), f.optimizer_plot_style(args.optimizers[2]))
        self.assertEqual(sorted(args.optimizers, key=f.optimizer_order_key), args.optimizers)
        self.assertEqual(f.parse_result_label("MACRO-DE-T-V2_VSTF_01_KNN", args)["method"], "MaCRO-DE-t-v2")

    def test_seeded_scientific_parity_continuous_binary_single_and_swarm(self):
        for binary in (False, True):
            for mode in ("single", "swarm"):
                models = []
                for cls in (Reference, f.MaCRO_DE_t_v2):
                    bounds = (TransferBinaryVar(n_vars=5, tf_func="vstf_01", lb=-8, ub=8,
                                                all_zeros=False) if binary else
                              FloatVar(lb=(-5.,) * 5, ub=(5.,) * 5))
                    with patch.dict(sys.modules, {"diversity_gpu_batching": ReferenceService}):
                        model = cls(epoch=3, pop_size=10)
                    model.solve(dict(bounds=bounds, minmax="min", log_to=None,
                                     obj_func=lambda x: float(np.sum(x ** 2))), mode=mode, seed=1234)
                    models.append(model)
                for attr in ("dm_hist", "pcr_hist", "f_hist", "fmean_hist", "d_hist",
                             "div_awad_hist", "div_norm_hist"):
                    np.testing.assert_array_equal(getattr(models[0], attr), getattr(models[1], attr))
                np.testing.assert_array_equal(models[0]._positions(models[0].pop), models[1]._positions(models[1].pop))
                np.testing.assert_array_equal(models[0].history.list_global_best_fit,
                                              models[1].history.list_global_best_fit)
                np.testing.assert_array_equal(models[1].pcr_hist, .1 + .25 * (1. - models[1].dm_hist))

    def test_v2_parameters_are_independent_and_all_dependencies_hashed(self):
        args = self.args()
        identities = {name: self.identity(args, name) for name in args.optimizers}
        self.assertEqual(len({cache.identity_digest(value) for value in identities.values()}), 3)
        identity = identities["MaCRO-DE-t-v2"]
        from reporting.core import _matches_request
        self.assertTrue(_matches_request(identity, args))
        self.assertEqual(identity["optimizer_parameters"], {
            "beta_min": .1, "beta_max": .6, "mahalanobis_q": .5,
            "pcr_policy": "0.1 + 0.25 * (1.0 - dM)",
        })
        changed = copy.deepcopy(args)
        changed.dsade_beta_min, changed.dsade_beta_max = .3, .9
        changed.dsade_pcr, changed.dsade_mahal_q = .4, .9
        self.assertEqual(identity, self.identity(changed, "MaCRO-DE-t-v2"))
        for field in ("macro_de_t_v2_beta_min", "macro_de_t_v2_beta_max", "macro_de_t_v2_mahal_q"):
            changed = copy.deepcopy(args)
            setattr(changed, field, getattr(changed, field) + .01)
            self.assertNotEqual(identity, self.identity(changed, "MaCRO-DE-t-v2"))
            self.assertFalse(_matches_request(identity, changed))
            for method in ("DSADE", "MaCRO-DE-t"):
                self.assertEqual(identities[method], self.identity(changed, method))
        for module in (macro_de_t_backend, transport):
            self.assertEqual(identity["optimizer_implementation"]["modules"][module.__name__],
                             cache.file_digest(module.__file__))
            digest = cache.file_digest
            def changed_digest(path, affected=module.__file__):
                return "changed-backend" if Path(path).resolve() == Path(affected).resolve() else digest(path)
            with patch.object(cache, "file_digest", side_effect=changed_digest):
                self.assertNotEqual(identity, self.identity(args, "MaCRO-DE-t-v2"))
                self.assertEqual(identities["MaCRO-DE-t"], self.identity(args, "MaCRO-DE-t"))
        with patch.object(f.MaCRO_DE_t_v2, "IMPLEMENTATION_REVISION", "changed"):
            self.assertNotEqual(identity, self.identity(args, "MaCRO-DE-t-v2"))

    def test_old_cache_never_matches_v2_and_is_not_modified(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as directory:
            args.output_root = directory
            paths = f.make_paths(args)
            old = self.identity(args, "MaCRO-DE-t")
            new = self.identity(args, "MaCRO-DE-t-v2")
            row = f.build_label_payload("knn", *[[.2, .3] for _ in range(7)],
                                        [[.5, .4, .2], [.6, .4, .3]], 3)
            f.save_combination(paths, old, row)
            old_files = cache.combination_files(paths, old)
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in old_files}
            self.assertEqual(list(cache.read_candidates(paths, new)), [])
            self.assertEqual(f.legacy_cache_signatures(args), [])
            f.save_combination(paths, new, row)
            self.assertNotEqual(old_files, cache.combination_files(paths, new))
            self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in old_files})


if __name__ == "__main__":
    unittest.main()
