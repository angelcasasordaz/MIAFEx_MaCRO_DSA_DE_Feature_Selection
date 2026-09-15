# CEC DE-MC-CF source snapshot

These four Python modules were supplied from
`Adaptive_Mahalanobis-Cholesky_DIfferential_Evolution` for MaCRO-DE-t.
Only the three sibling imports were made package-relative. The framework
adapter is `macro_de_t_optimizer.MaCRO_DE_t`; the CEC scientific implementation
and its CPU defaults are inherited unchanged.

SHA-256 of the supplied files (before import changes):

| File | SHA-256 |
| --- | --- |
| de_mc_cf_optimizer.py | 0501f80af33f3bf6d5926b45d88fa545afb261ab3c487697ca8d914c6c42f5eb |
| de_mc_optimizer.py | 424907cd4c97bf84540851a45775e83583c6cc21706fd81cdce902dc4a1a6750 |
| de_ablation_base.py | ae7ab5ef49eee6a5c536b56e5117ba7c6695562b9a590263e2477144a8d610f0 |
| compute_backend.py | a79aa07a8a53a8a2aa7a149a09e7174fd89bd346c0e26664e33c08830f036850 |
| validate_gpu_batching.py | b258e065b09c3ab4388cd2570c8d25a24eefe7cda31db84548b49571edd57c87 |

`validate_gpu_batching.py` was inspected as the authoritative validation
reference; its experiment/batching imports are not needed by this adapter.
The focused tests reproduce its donor, crossover, frozen-generation and greedy
selection checks without running that script or any experiments.

`tests/test_optimizer_integrations.py` loads the original modules independently
from the sibling CEC directory (override with `CEC_REFERENCE_DIR`). If that
directory is unavailable, it loads the hash-verified snapshot with its original
imports restored. Both paths verify the original source hashes before execution.
