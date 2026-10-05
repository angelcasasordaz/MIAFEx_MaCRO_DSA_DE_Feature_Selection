"""WFS interface adapter for the current MaCRO-DE-t-v2, not a new variant.

The original macro_de_t_v2_optimizer implementation supplies all search
dynamics, including coordinate scale draws, delayed AWAD diversity, adaptive
crossover, Mahalanobis/Cholesky logic and survivor selection. Only wrapper
evaluation, the deterministic WFS threshold and output format are adapted.
No search method is overridden and no MAFESE selector is used.
"""

from FS.macro_de_t_wfs import _solve_wfs
from macro_de_t_v2_optimizer import MaCRO_DE_t_v2


def jfs(feat, label, opts):
    """Accept WFS options and retain the original v2 parameter defaults."""
    optimizer = MaCRO_DE_t_v2(
        epoch=opts["T"], pop_size=opts["N"],
        beta_min=opts.get("beta_min", 0.10), beta_max=opts.get("beta_max", 0.60),
        mahalanobis_q=opts.get("mahalanobis_q", 0.50), compute_device="cpu",
    )
    return _solve_wfs(optimizer, feat, label, opts)
