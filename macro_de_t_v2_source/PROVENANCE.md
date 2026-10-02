# Authoritative MaCRO-DE-t-v2 snapshot

Supplied by `Diversity_based_Self_Adaptive_Differential_Evolution`.
The three files retain all supplied scientific code and defaults. Only imports
were adapted: the parent and backend imports are package-relative, and the
backend's `diversity_gpu_batching` import points to `.transport`. The transport
dispatches the supplied covariance kernels using the existing NumPy backend
on CPU, as the framework does for its other custom optimizers.

The existing root `macro_de_t_optimizer.py` and its CEC inheritance chain remain
untouched, preserving historical MaCRO-DE-t scientific/cache identities.

SHA-256 before import adaptations:

| File | SHA-256 |
| --- | --- |
| macro_de_t_v2_optimizer.py | 2cea43e8c56897565bbfd73c01bb50084f76b7c1e3079f9f7b90184f1b3723d2 |
| macro_de_t_optimizer.py | 5362cb24be130a7798e85bfdf6d3b3c710044c91f57a8f0348a73059654a677e |
| macro_de_t_backend.py | 360cf7f7566cf892e8c7ba9c73ab81e82bc50f31dfeddb98b5d04ec55b745354 |

Focused tests restore these three imports and verify the original hashes.
V2 cache identities hash the installed inheritance chain, covariance backend,
transport and numerical backend. Revision `macro-d-scaled-coordinate-adaptive-pcr-v5`
and canonical identity `MaCRO-DE-t-v2` come directly from the supplied class.
