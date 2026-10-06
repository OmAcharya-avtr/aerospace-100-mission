"""aperturediv: multi-aperture receive diversity for free-space optical links.

Research-grade software. **Not flight-qualified, not certified, not approved
for operational aerospace use.** It models a channel; it does not qualify a
terminal, and nothing in it substitutes for a measured link.

Layout
------
:mod:`aperturediv.channel`
    Lognormal and gamma-gamma irradiance statistics, unit mean, with the
    scintillation index checked against each model's own closed form.
:mod:`aperturediv.aperture`
    Aperture averaging from its defining integral, with two derived
    closed-form limits used as validation targets.
:mod:`aperturediv.correlation`
    Inter-aperture correlation as an explicit input, and the exact lognormal
    relation between log-domain and irradiance correlation.
:mod:`aperturediv.combining`
    MRC, EGC and SC; outage probability; finite-window diversity order.
:mod:`aperturediv.estimation`
    Stale and noisy channel-state estimates, collapsed onto one error level.
:mod:`aperturediv.ber`
    Coherent BPSK BER over fading: a sample statistic and a quadrature
    diagnostic.
:mod:`aperturediv.datasets`
    Seeded synthetic datasets for the learned combiner.
:mod:`aperturediv.learned`
    Learned combiner weighting under imperfect CSI, against four
    non-learned references, with a penalty-quantile uncertainty output.

Conventions that matter for any comparison with other software
--------------------------------------------------------------
* Irradiance is normalised to unit mean; the link budget carries the mean.
* The instantaneous SNR is **linear** in irradiance, ``gamma = gamma_bar I``,
  equivalently the amplitude gain is ``sqrt(I)``. A thermal-noise-limited
  intensity-modulated receiver would instead have ``gamma ∝ I^2``.
* Inter-aperture correlation is specified on the **log-irradiance** field.
* Diversity order is a least-squares log-log slope over a **stated** outage
  window, not an asymptotic limit.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import aperture, ber, channel, combining, correlation, datasets, estimation, learned
from .aperture import (
    aperture_averaging_factor,
    effective_scintillation_index,
    equal_area_diameter,
    fresnel_scale,
)
from .ber import bpsk_ber_lognormal_gauss_hermite, bpsk_ber_lognormal_sample
from .channel import (
    gamma_gamma_params_from_rytov,
    gamma_gamma_pdf,
    gamma_gamma_scintillation_index,
    lognormal_pdf,
    lognormal_sigma_log,
)
from .combining import (
    branch_mean_snr,
    combined_gain,
    diversity_order,
    outage_probability,
    weighted_snr,
)
from .correlation import (
    correlation_matrix,
    irradiance_correlation_matrix,
    log_to_irradiance_correlation,
    sample_correlated_gamma_gamma,
    sample_correlated_lognormal,
)
from .datasets import make_combiner_dataset, split_dataset
from .estimation import estimate_from_log_error, log_error_sigma, log_error_sigma_db
from .learned import LearnedCombiner, penalty_db, score_weights

__all__ = [
    "LearnedCombiner",
    "__version__",
    "aperture",
    "aperture_averaging_factor",
    "ber",
    "bpsk_ber_lognormal_gauss_hermite",
    "bpsk_ber_lognormal_sample",
    "branch_mean_snr",
    "channel",
    "combined_gain",
    "combining",
    "correlation",
    "correlation_matrix",
    "datasets",
    "diversity_order",
    "effective_scintillation_index",
    "equal_area_diameter",
    "estimate_from_log_error",
    "estimation",
    "fresnel_scale",
    "gamma_gamma_params_from_rytov",
    "gamma_gamma_pdf",
    "gamma_gamma_scintillation_index",
    "irradiance_correlation_matrix",
    "learned",
    "log_error_sigma",
    "log_error_sigma_db",
    "log_to_irradiance_correlation",
    "lognormal_pdf",
    "lognormal_sigma_log",
    "make_combiner_dataset",
    "outage_probability",
    "penalty_db",
    "sample_correlated_gamma_gamma",
    "sample_correlated_lognormal",
    "score_weights",
    "split_dataset",
    "weighted_snr",
]
