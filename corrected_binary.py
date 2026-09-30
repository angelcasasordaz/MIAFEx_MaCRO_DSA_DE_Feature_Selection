"""Validated binary adapters from diagnostics/corrected_binary_final.

Transfer sampling belongs to proposal correction, never to stored-mask decoding.
Optimizer evolution equations and the upstream packages remain unchanged.
"""
from contextlib import contextmanager
import inspect
from unittest.mock import patch

import mafese.wrapper.mha as mha
from mealpy import TransferBinaryVar
from mealpy.human_based.BRO import OriginalBRO
import numpy as np


class CorrectedTransferBinaryVar(TransferBinaryVar):
    """Preserve requested bounds and decode a stored binary mask without RNG."""
    IMPLEMENTATION_REVISION = "transfer-binary-contract-v1"

    def __init__(self, *args, **kwargs):
        supplied = inspect.signature(TransferBinaryVar.__init__).bind(self, *args, **kwargs)
        supplied.apply_defaults()
        super().__init__(*args, **kwargs)
        self.lb = np.broadcast_to(np.asarray(supplied.arguments["lb"], dtype=float), (self.n_vars,)).copy()
        self.ub = np.broadcast_to(np.asarray(supplied.arguments["ub"], dtype=float), (self.n_vars,)).copy()
        if not np.all(self.lb < self.ub):
            raise ValueError("Invalid caller-supplied transfer bounds")

    def decode(self, x):
        values = np.asarray(x)
        if not np.isin(values, (0, 1)).all() or (not self.all_zeros and not np.any(values)):
            raise ValueError("Stored transfer-binary solution must be a valid binary mask")
        return values.astype(int, copy=True)


class BinaryRespawnBRO(OriginalBRO):
    """Repair continuous respawns before evaluation; inherit evolve unchanged."""
    IMPLEMENTATION_REVISION = "bro-binary-respawn-v1"

    def generate_agent(self, solution=None):
        if solution is not None and (not np.isin(solution, (0, 1)).all() or not np.any(solution)):
            solution = self.correct_solution(solution)
        return super().generate_agent(solution)


@contextmanager
def corrected_transfer_binary():
    """Scope MAFESE's variable binding to a fit, restoring it even on failure.

    Framework fits run sequentially in each process; parallel runs use spawned
    processes, so this binding is never shared between concurrent worker fits.
    """
    with patch.object(mha, "TransferBinaryVar", CorrectedTransferBinaryVar):
        yield
