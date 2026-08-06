"""berbench — BER computation and Monte Carlo benchmarking for OOK, BPSK and
M-ary PPM over AWGN and lognormal-fading (weak-turbulence FSO) channels.

Research-grade software. Not flight-qualified, not certified, not approved
for operational aerospace use.
"""

from ._math import log10_qfunc, log_qfunc, qfunc, wilson_interval
from .analytic import MODULATIONS, analytic_ber
from .channels import CHANNELS, lognormal_irradiance_nodes, sample_lognormal_irradiance
from .montecarlo import mc_ber, n_bits_for_target
from .results import AnalyticResult, MCResult

__version__ = "0.1.0"

__all__ = [
    "analytic_ber",
    "mc_ber",
    "n_bits_for_target",
    "AnalyticResult",
    "MCResult",
    "qfunc",
    "log_qfunc",
    "log10_qfunc",
    "wilson_interval",
    "lognormal_irradiance_nodes",
    "sample_lognormal_irradiance",
    "MODULATIONS",
    "CHANNELS",
    "__version__",
]
