"""Local kernel transport, replacing the sibling project's GPU owner service.

The framework builds custom optimizers on CPU. Kernel arithmetic lives entirely
in the supplied CECCovarianceKernels; this adapter only dispatches its calls.
"""
from cec_de_mc_cf.compute_backend import ComputeBackend


class DiversityMathBatcher:
    def __init__(self, compute_device="cpu", gpu_device_id=0, gpu_memory_fraction=0.85):
        if compute_device != "cpu":
            raise ValueError("MIAFEx MaCRO-DE-t-v2 transport uses compute_device='cpu'.")
        self.backend = ComputeBackend(compute_device, device_id=gpu_device_id)

    def macro_de_t_covariance(self, operation, *args):
        from .macro_de_t_backend import CECCovarianceKernels
        return getattr(CECCovarianceKernels(self.backend), operation)(*args)
