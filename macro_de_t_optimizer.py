"""MEALPY/MAFESE identity for the authoritative CEC DE-MC-CF optimizer."""

from cec_de_mc_cf.de_mc_cf_optimizer import DE_MC_CF


class MaCRO_DE_t(DE_MC_CF):
    """MaCRO-DE-t: CEC AWAD close/far DE/rand/1/bin, F=0.5, Cr=0.9.

    The inherited CEC implementation already implements the MEALPY interface.
    No numerical, population-update, or binary-mapping overrides are needed.
    """

    IMPLEMENTATION_REVISION = "awad-close-far-v2"
