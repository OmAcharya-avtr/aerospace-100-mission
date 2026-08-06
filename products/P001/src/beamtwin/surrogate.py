"""ML fade-probability surrogate for the BeamTwin digital twin.

A gradient-boosting ensemble (scikit-learn) predicts log10 fade probability
directly from link parameters, trained on the twin's own seeded Monte Carlo
outputs. Uncertainty is reported as the spread of an ensemble whose members
differ in random state and data subsample (a pragmatic, honest spread
estimate - not a calibrated Bayesian interval; see MODEL_CARD.md).

Value proposition: the analytic lognormal baseline
(beamtwin.stats.analytic_fade_probability_lognormal) is exact only for
scintillation-only fading. When pointing jitter and scintillation combine,
there is no simple closed form, and full Monte Carlo costs ~10^5 samples per
query. The surrogate answers combined-case queries in microseconds.

This model is not certified for operational flight use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .budget import LinkParams, compute_budget, gaussian_divergence_half_angle
from .channel import ChannelParams, sample_received_power_dbm
from .stats import fade_probability

#: Fade probabilities are clipped to this floor before the log10 transform.
#: It matches the resolution of the training Monte Carlo (~3 fades in 3e4
#: samples); the surrogate cannot resolve rarer fades. Documented limitation.
P_FLOOR = 1e-4

FEATURE_NAMES: tuple[str, ...] = (
    "log10_range_m",
    "log10_cn2",
    "jitter_ratio",  # pointing_jitter_rad / divergence_half_angle_rad
    "attenuation_db_per_km",
    "margin_db",  # deterministic budget margin
)

#: Training-domain bounds per feature (min, max); used for extrapolation flags.
TRAIN_DOMAIN: dict[str, tuple[float, float]] = {
    "log10_range_m": (3.0, math.log10(20_000.0)),  # 1-20 km
    "log10_cn2": (-16.0, math.log10(5e-14)),
    "jitter_ratio": (0.0, 0.5),
    "attenuation_db_per_km": (0.0, 3.0),
    "margin_db": (-5.0, 25.0),
}

_MC_SAMPLES_TRAIN = 50_000


def features_from_params(link: LinkParams, channel: ChannelParams) -> np.ndarray:
    """Build the 5-element feature vector for one link/channel configuration.

    Units per FEATURE_NAMES: log10 of range [m], log10 of Cn2 [m^-2/3]
    (floored at 1e-18 for cn2 = 0), jitter/divergence ratio [-],
    attenuation [dB/km], deterministic margin [dB].
    """
    theta = gaussian_divergence_half_angle(link.wavelength_m, link.beam_waist_radius_m)
    margin = compute_budget(link).margin_db
    return np.array(
        [
            math.log10(link.range_m),
            math.log10(max(channel.cn2, 1e-18)),
            channel.pointing_jitter_rad / theta,
            link.attenuation_db_per_km,
            margin,
        ]
    )


def in_training_domain(x: np.ndarray) -> bool:
    """True if a feature vector lies inside the training domain bounds."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {x.size}")
    for value, name in zip(x, FEATURE_NAMES):
        lo, hi = TRAIN_DOMAIN[name]
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            return False
    return True


def _sample_scenario(rng: np.random.Generator) -> tuple[LinkParams, ChannelParams]:
    """Draw one random scenario from the training domain (seeded)."""
    range_m = 10.0 ** rng.uniform(3.0, math.log10(20_000.0))
    cn2 = 10.0 ** rng.uniform(-16.0, math.log10(5e-14))
    att = rng.uniform(0.0, 3.0)
    # Draw the deterministic margin uniformly in the informative range
    # [-5, 25] dB by setting the sensitivity relative to the received power;
    # sampling the sensitivity directly leaves >75 % of scenarios fade-free
    # at the P_FLOOR resolution (imbalanced targets).
    margin_db = rng.uniform(-5.0, 25.0)
    link = LinkParams(range_m=range_m, attenuation_db_per_km=att)
    p_rx = compute_budget(link).received_power_dbm
    link = LinkParams(
        range_m=range_m,
        attenuation_db_per_km=att,
        rx_sensitivity_dbm=p_rx - margin_db,
    )
    theta = gaussian_divergence_half_angle(link.wavelength_m, link.beam_waist_radius_m)
    jitter = rng.uniform(0.0, 0.5) * theta
    return link, ChannelParams(cn2=cn2, pointing_jitter_rad=jitter)


def generate_dataset(
    n_scenarios: int = 4000,
    seed: int = 42,
    mc_samples: int = _MC_SAMPLES_TRAIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) from seeded Monte Carlo runs of the twin itself.

    y = log10(max(P_fade, P_FLOOR)). Deterministic for fixed arguments
    (independent child seeds are spawned from `seed`). Runtime: ~20 s for
    the default 4000 x 50000 on 2 CPU cores.
    """
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}")
    rng = np.random.default_rng(seed)
    x_rows = np.empty((n_scenarios, len(FEATURE_NAMES)))
    y = np.empty(n_scenarios)
    for i in range(n_scenarios):
        link, channel = _sample_scenario(rng)
        mc_seed = int(rng.integers(0, 2**31 - 1))
        result = sample_received_power_dbm(link, channel, n_samples=mc_samples, seed=mc_seed)
        est = fade_probability(result.samples_dbm, link.rx_sensitivity_dbm)
        x_rows[i] = features_from_params(link, channel)
        y[i] = math.log10(max(est.probability, P_FLOOR))
    return x_rows, y


@dataclass(frozen=True)
class SurrogatePrediction:
    """Fade-probability prediction with ensemble-spread uncertainty.

    probability : ensemble-mean fade probability (floored at P_FLOOR).
    p_low, p_high : probability at -/+ 2 ensemble standard deviations in
        log10 space (spread of ensemble members, NOT a calibrated interval).
    log10_std : ensemble std of log10(P).
    extrapolating : True when the query lies outside the training domain.
    """

    probability: float
    p_low: float
    p_high: float
    log10_std: float
    extrapolating: bool


class FadeSurrogate:
    """Ensemble of GradientBoostingRegressor models on log10 fade probability.

    n_members members are trained with distinct random_state and 80 %
    subsampling; the member spread provides the uncertainty output required
    of BeamTwin's AI component.
    """

    def __init__(self, n_members: int = 5, random_state: int = 7) -> None:
        if n_members < 2:
            raise ValueError(f"n_members must be >= 2 for a spread estimate, got {n_members}")
        self.n_members = n_members
        self.random_state = random_state
        self._members: list[GradientBoostingRegressor] = []

    @property
    def is_fitted(self) -> bool:
        return len(self._members) == self.n_members

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FadeSurrogate":
        """Fit the ensemble. x shape (n, 5), y = log10 fade probability."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"x must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}")
        if y.shape != (x.shape[0],):
            raise ValueError(f"y must have shape ({x.shape[0]},), got {y.shape}")
        self._members = []
        for m in range(self.n_members):
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=self.random_state + m,
            )
            model.fit(x, y)
            self._members.append(model)
        return self

    def predict_log10(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean and std of log10(P_fade) for feature rows x."""
        if not self.is_fitted:
            raise RuntimeError("surrogate is not fitted; call fit() or load a saved model")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"x must have {len(FEATURE_NAMES)} columns, got {x.shape[1]}")
        preds = np.stack([m.predict(x) for m in self._members])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, link: LinkParams, channel: ChannelParams) -> SurrogatePrediction:
        """Predict fade probability with uncertainty for one configuration."""
        feats = features_from_params(link, channel)
        mean_log, std_log = self.predict_log10(feats)
        mean_l, std_l = float(mean_log[0]), float(std_log[0])
        clip = lambda v: float(min(1.0, max(P_FLOOR, 10.0**v)))  # noqa: E731
        return SurrogatePrediction(
            probability=clip(mean_l),
            p_low=clip(mean_l - 2.0 * std_l),
            p_high=clip(mean_l + 2.0 * std_l),
            log10_std=std_l,
            extrapolating=not in_training_domain(feats),
        )

    def save(self, path: str | Path) -> None:
        """Persist the fitted ensemble with joblib."""
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted surrogate")
        joblib.dump(
            {
                "n_members": self.n_members,
                "random_state": self.random_state,
                "members": self._members,
                "feature_names": list(FEATURE_NAMES),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FadeSurrogate":
        """Load a surrogate saved by save(). Raises FileNotFoundError if absent."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"surrogate model file not found: {path}. "
                "Train it with scripts/train_surrogate.py"
            )
        payload = joblib.load(path)
        obj = cls(n_members=payload["n_members"], random_state=payload["random_state"])
        obj._members = payload["members"]
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("saved model feature set does not match this beamtwin version")
        return obj


def default_model_path() -> Path:
    """Path of the committed trained model (products/P001/models/surrogate.joblib)."""
    return Path(__file__).resolve().parents[2] / "models" / "surrogate.joblib"


__all__ = [
    "FEATURE_NAMES",
    "P_FLOOR",
    "TRAIN_DOMAIN",
    "FadeSurrogate",
    "SurrogatePrediction",
    "default_model_path",
    "features_from_params",
    "generate_dataset",
    "in_training_domain",
]
