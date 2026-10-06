"""slotsync - slot and symbol timing recovery for OOK and PPM optical receivers.

The package is organised as one chain, measured end to end:

    pulse shape -> timing-error detector -> S-curve -> detector gain K_d
                -> second-order loop -> timing jitter variance -> cycle-slip rate

and the point of it is that no link in that chain is asserted.  ``K_d`` is the
slope of an S-curve this package computed; the jitter prediction uses that
measured gain and a measured detector-noise statistic; and the prediction is
checked against a Monte Carlo of the actual loop, with the disagreements
reported rather than tuned away.

This software is research-grade.  It is **not flight-qualified, not certified,
and not approved for operational aerospace use.**

Modules
-------
:mod:`slotsync.pulses`
    Finite-support pulse shapes in symbol-period units.
:mod:`slotsync.detectors`
    Early-late, Gardner and Mueller-Mueller detectors as pure functions.
:mod:`slotsync.ted`
    Detector configuration, sample-instant bookkeeping and dispatch.
:mod:`slotsync.stream`
    Analytic sampling of a pulse-amplitude-modulated stream, and the noise model.
:mod:`slotsync.scurve`
    Exact S-curves by data-pattern enumeration, and the gain read off them.
:mod:`slotsync.loop`
    Second-order loop design, jitter variance by three routes, cycle slips.
:mod:`slotsync.simulate`
    Open-loop detector statistics and the closed-loop Monte Carlo.
:mod:`slotsync.ppm`
    M-ary PPM slot-clock synchronisation, and what does not carry over to it.
"""

from __future__ import annotations

from .detectors import (
    DETECTOR_NAMES,
    early_late,
    early_late_dd,
    gardner,
    gardner_complex,
    mueller_muller,
    slice_antipodal,
)
from .loop import (
    LoopDesign,
    cycle_slip_rate_rice,
    error_autocorrelation_weights,
    jitter_variance_closed_form,
    jitter_variance_coloured,
    jitter_variance_exact,
    loop_snr_db,
    noise_bandwidth_closed_form,
    noise_bandwidth_numeric,
    slip_free_seconds,
)
from .ppm import (
    PpmConfig,
    PpmLoopRun,
    PpmSlotCurve,
    duty_cycle_penalty_db,
    equivalent_slot_bandwidth,
    measure_ppm_slot_statistics,
    ppm_slot_autocovariance,
    ppm_slot_scurve,
    run_ppm_slot_loop,
    slot_index_error_rate,
)
from .pulses import (
    PulseShape,
    half_sine,
    nyquist_raised_cosine,
    pulse_by_name,
    raised_cosine_time,
    rectangular,
    triangular,
)
from .scurve import SCurve, default_offsets, scurve
from .simulate import (
    LoopRun,
    TedStatistics,
    measure_ted_autocovariance,
    measure_ted_statistics,
    pulse_table,
    run_timing_loop,
)
from .stream import sample_noise_sigma, sample_stream
from .ted import TedConfig, evaluate_ted, evaluate_ted_scalar, ted_sample_slots, ted_time_offsets

__version__ = "0.1.0"

__all__ = [
    "DETECTOR_NAMES",
    "LoopDesign",
    "LoopRun",
    "PpmConfig",
    "PpmLoopRun",
    "PpmSlotCurve",
    "PulseShape",
    "SCurve",
    "TedConfig",
    "TedStatistics",
    "__version__",
    "cycle_slip_rate_rice",
    "default_offsets",
    "duty_cycle_penalty_db",
    "early_late",
    "early_late_dd",
    "equivalent_slot_bandwidth",
    "error_autocorrelation_weights",
    "evaluate_ted",
    "evaluate_ted_scalar",
    "gardner",
    "gardner_complex",
    "half_sine",
    "jitter_variance_closed_form",
    "jitter_variance_coloured",
    "jitter_variance_exact",
    "loop_snr_db",
    "measure_ppm_slot_statistics",
    "measure_ted_autocovariance",
    "measure_ted_statistics",
    "mueller_muller",
    "noise_bandwidth_closed_form",
    "noise_bandwidth_numeric",
    "nyquist_raised_cosine",
    "ppm_slot_autocovariance",
    "ppm_slot_scurve",
    "pulse_by_name",
    "pulse_table",
    "raised_cosine_time",
    "rectangular",
    "run_ppm_slot_loop",
    "run_timing_loop",
    "sample_noise_sigma",
    "sample_stream",
    "scurve",
    "slice_antipodal",
    "slip_free_seconds",
    "slot_index_error_rate",
    "ted_sample_slots",
    "ted_time_offsets",
    "triangular",
]
