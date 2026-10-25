"""The assembled report for one object, and what it refuses to be.

:class:`DriftReport` is the thing a user holds at the end: a divergence curve,
the analytic growth forms fitted to it with the best-fitting one named, the
lower bound the curve supports, and -- if a population predictor was supplied
-- a staleness verdict with the measured coverage of its interval.

Two methods exist only to refuse. :meth:`DriftReport.as_covariance` raises,
and :meth:`DriftReport.to_dict` audits its own output for covariance-shaped
keys before returning it, so a future field named ``position_covariance``
fails a test rather than shipping. :mod:`orbitdrift.guard` explains why, names
P071 ConjProb as the product that does need a covariance, and says where that
covariance has to come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .divergence import DivergenceCurve, larger_error_lower_bound_km
from .features import ObjectFeatures
from .growth import GrowthFit, best_form, fit_all_forms
from .guard import audit_mapping_for_covariance_keys, refuse_covariance
from .staleness import StalenessVerdict


@dataclass(frozen=True)
class DriftReport:
    """Everything measured about one element set's divergence."""

    satnum: int
    features: ObjectFeatures
    curve: DivergenceCurve
    fits: dict[str, GrowthFit]
    verdict: StalenessVerdict | None = None

    @property
    def best_form_name(self) -> str:
        """Name of the analytic form with the smallest log-residual spread."""
        return best_form(self.fits)

    @property
    def lower_bound_km(self) -> np.ndarray:
        """Half the drift: a lower bound on the larger of the two models' errors."""
        return larger_error_lower_bound_km(self.curve.drift_km)

    def as_covariance(self) -> None:
        """Always raises :class:`orbitdrift.errors.CovarianceRefused`."""
        refuse_covariance(
            "DriftReport.as_covariance was called; a model-divergence curve is "
            "not a state covariance and must not be used as one"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary, audited for covariance-shaped keys before return."""
        payload: dict[str, Any] = {
            "satnum": int(self.satnum),
            "source": self.curve.source,
            "detail": self.curve.detail,
            "quantity": (
                "window-RMS model divergence in km against epoch age in days; "
                "half of it is a lower bound on the larger of the two models' "
                "position errors, and it is not an error estimate and not a "
                "covariance"
            ),
            "features": {
                "perigee_altitude_km": self.features.perigee_altitude_km,
                "apogee_altitude_km": self.features.apogee_altitude_km,
                "eccentricity": self.features.eccentricity,
                "inclination_deg": self.features.inclination_deg,
                "bstar": self.features.bstar,
                "period_min": self.features.period_min,
            },
            "ages_days": [float(value) for value in self.curve.ages_days],
            "drift_km": [float(value) for value in self.curve.drift_km],
            "lower_bound_km": [float(value) for value in self.lower_bound_km],
            "growth_fits": {
                name: {
                    "form": fit.form,
                    "amplitude_km": fit.amplitude_km,
                    "exponent": fit.exponent,
                    "timescale_days": fit.timescale_days,
                    "log_residual_std": fit.log_residual_std,
                    "n_points": fit.n_points,
                }
                for name, fit in self.fits.items()
            },
            "best_form": self.best_form_name if self.fits else None,
        }
        if self.verdict is not None:
            payload["staleness"] = {
                "tolerance_km": self.verdict.tolerance_km,
                "median_horizon_days": self.verdict.median_horizon_days,
                "conservative_horizon_days": self.verdict.conservative_horizon_days,
                "nominal_coverage": self.verdict.nominal_coverage,
                "measured_coverage": self.verdict.measured_coverage,
                "predictor": self.verdict.predictor_name,
            }
        audit_mapping_for_covariance_keys(payload, where="DriftReport")
        return payload

    def summary(self) -> str:
        """Plain-text report. No units are left implicit."""
        lines = [
            f"object                   : catalogue {self.satnum}",
            f"perigee altitude         : {self.features.perigee_altitude_km:.1f} km",
            f"eccentricity             : {self.features.eccentricity:.6f}",
            f"inclination              : {self.features.inclination_deg:.3f} deg",
            f"B*                       : {self.features.bstar:.6e} 1/earth-radii",
            f"divergence source        : {self.curve.source}",
            f"                           {self.curve.detail}",
            "",
            "epoch age  model divergence  lower bound on larger model error",
            "   (days)              (km)                              (km)",
        ]
        for age, drift, bound in zip(
            self.curve.ages_days, self.curve.drift_km, self.lower_bound_km, strict=True
        ):
            lines.append(f"{age:10.3f}  {drift:16.4f}  {bound:32.4f}")
        if self.fits:
            lines.append("")
            lines.append("analytic growth forms fitted to this curve:")
            lines.append("  form              amplitude(km)   exponent   tau(d)   log-resid sd")
            for name in sorted(self.fits):
                fit = self.fits[name]
                lines.append(
                    f"  {name:16s}  {fit.amplitude_km:13.5f}  "
                    f"{fit.exponent:9.4f}  {fit.timescale_days:7.2f}  "
                    f"{fit.log_residual_std:12.5f}"
                )
            lines.append(f"  best fitting form : {self.best_form_name}")
        if self.verdict is not None:
            lines.append("")
            lines.append(self.verdict.describe())
        return "\n".join(lines)


def build_report(
    satrec,
    curve: DivergenceCurve,
    *,
    verdict: StalenessVerdict | None = None,
) -> DriftReport:
    """Assemble a :class:`DriftReport` from a propagated curve."""
    from .features import features_from_satrec

    return DriftReport(
        satnum=int(satrec.satnum),
        features=features_from_satrec(satrec),
        curve=curve,
        fits=fit_all_forms(curve.ages_days, curve.drift_km),
        verdict=verdict,
    )
