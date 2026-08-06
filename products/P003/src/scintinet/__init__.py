"""ScintiNet — learned scintillation surrogate for FSO link planning.

Research-grade software. Not flight-qualified, not certified, not approved
for operational aerospace use. Validity: weak-fluctuation regime only.
"""

from .rytov import aperture_averaging_factor, rytov_variance, scintillation_index_weak
from .simulator import (
    SimParams,
    SimResult,
    angular_spectrum_propagate,
    kolmogorov_phase_screen,
    simulate_scintillation,
)
from .surrogate import Surrogate, rytov_baseline

__version__ = "0.1.0"

__all__ = [
    "rytov_variance",
    "aperture_averaging_factor",
    "scintillation_index_weak",
    "SimParams",
    "SimResult",
    "kolmogorov_phase_screen",
    "angular_spectrum_propagate",
    "simulate_scintillation",
    "Surrogate",
    "rytov_baseline",
    "__version__",
]
