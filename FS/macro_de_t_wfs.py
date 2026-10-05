"""WFS interface adapter for the current MaCRO-DE-t, not a new variant.

Search dynamics execute in macro_de_t_optimizer.MaCRO_DE_t unchanged: its
native solve loop, mutation, AWAD routing, Mahalanobis/Cholesky arithmetic,
survivors, RNG calls and stopping are retained. Only the wrapper objective,
binary interface and returned dictionary are adapted to WFS. No MAFESE
selector or transfer-function wrapper participates in this adapter.
"""

import numpy as np
from mealpy.utils.space import FloatVar

from FS.functionHO import Fun
from macro_de_t_optimizer import MaCRO_DE_t


def _solve_wfs(optimizer, feat, label, opts):
    """Run the original optimizer directly with WFS's deterministic mask."""
    fold, k = opts["fold"], opts["k"]
    verbose = opts.get("verbose", True)
    if k != 3:
        raise ValueError("The original WFS comparison requires KNN k=3")
    dim = np.size(feat, 1)
    if dim < 1 or any(key not in fold for key in ("xt", "yt", "xv", "yv")):
        raise ValueError("WFS requires features and the prepared train/test fold")

    def objective(position):
        # No stochastic decoding, repair of empty masks, or extra RNG draws.
        return Fun(feat, label, (np.asarray(position) > 0.5).astype(int), opts)

    problem = {
        "bounds": FloatVar(lb=np.zeros(dim), ub=np.ones(dim), name="features"),
        "obj_func": objective,
        "minmax": "min",
        "log_to": "console" if verbose else None,
    }
    # The comparison supplies the run seed explicitly. The fallback supports
    # legacy jfs callers that seed numpy.random; inspecting state draws nothing.
    seed = int(opts.get("seed", np.random.get_state()[1][0]))
    best = optimizer.solve(problem, mode="single", seed=seed)
    selected = np.flatnonzero(np.asarray(best.solution) > 0.5)
    # Native history: retain every original epoch, with no padding/reindexing.
    curve = np.asarray(optimizer.history.list_global_best_fit, dtype=float).reshape(1, -1)
    return {"sf": selected, "c": curve, "nf": len(selected)}


def jfs(feat, label, opts):
    """Accept the original WFS (feat, label, opts) interface."""
    optimizer = MaCRO_DE_t(
        epoch=opts["T"], pop_size=opts["N"], wf=0.5, cr=0.9,
        mahalanobis_q=opts.get("mahalanobis_q", 0.68), compute_device="cpu",
    )
    return _solve_wfs(optimizer, feat, label, opts)
