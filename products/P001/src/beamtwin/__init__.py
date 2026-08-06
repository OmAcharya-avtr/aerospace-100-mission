"""BeamTwin — free-space optical (FSO) link digital twin.

Deterministic link budget + stochastic atmospheric channel + Monte Carlo
fade statistics + ML fade-probability surrogate.

Research-grade MVP (v0.1). Not flight-qualified, not certified, not
approved for operational aerospace use.
"""

from .budget import (
    LinkBudget,
    LinkParams,
    beam_radius,
    compute_budget,
    gaussian_divergence_half_angle,
    geometric_capture_fraction,
    kim_attenuation_db_per_km,
    pointing_loss_fraction,
)
from .channel import (
    ChannelModel,
    ChannelParams,
    MonteCarloResult,
    build_channel_model,
    mean_pointing_loss_fraction,
    rytov_variance_plane_wave,
    sample_received_power_dbm,
)
from .scenario import (
    Scenario,
    ScenarioError,
    format_report_text,
    load_scenario,
    run_twin,
    scenario_from_dict,
)
from .stats import (
    FadeEstimate,
    analytic_fade_probability_lognormal,
    fade_probability,
    margin_moments,
    margin_percentiles,
)
from .surrogate import FadeSurrogate, SurrogatePrediction, default_model_path

__version__ = "0.1.0"

__all__ = [
    "ChannelModel",
    "ChannelParams",
    "FadeEstimate",
    "FadeSurrogate",
    "LinkBudget",
    "LinkParams",
    "MonteCarloResult",
    "Scenario",
    "ScenarioError",
    "SurrogatePrediction",
    "__version__",
    "analytic_fade_probability_lognormal",
    "beam_radius",
    "build_channel_model",
    "compute_budget",
    "default_model_path",
    "fade_probability",
    "format_report_text",
    "gaussian_divergence_half_angle",
    "geometric_capture_fraction",
    "kim_attenuation_db_per_km",
    "load_scenario",
    "margin_moments",
    "margin_percentiles",
    "mean_pointing_loss_fraction",
    "pointing_loss_fraction",
    "run_twin",
    "rytov_variance_plane_wave",
    "sample_received_power_dbm",
    "scenario_from_dict",
]
