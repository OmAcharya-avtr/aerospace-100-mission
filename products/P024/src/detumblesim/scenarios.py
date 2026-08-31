"""Seeded synthetic detumble scenarios.

**No flight telemetry is used anywhere in this package.**  Every scenario is
drawn from the distributions documented below by a seeded
``numpy.random.Generator``, so the whole dataset is regenerated bit-for-bit
from its seed and nothing needs to be committed.  See ``DATASET_CARD.md``.

Sampling ranges are engineering choices covering the small-satellite regime
that B-dot is normally applied to; they are not fitted to any measured
population of spacecraft, and that is stated rather than implied.

Ranges
------
inertia scale ``j``          log-uniform 0.01 - 0.30 kg m^2
principal-axis ratios        uniform 0.6 - 1.6 about ``j`` (triangle
                             inequality enforced by resampling)
initial rate ``|omega_0|``   log-uniform 3 - 20 deg/s, isotropic direction
altitude                     uniform 400 - 800 km
inclination                  uniform 20 - 100 deg
RAAN, argument of latitude,  uniform over their full ranges
Earth rotation phase
max dipole per axis          log-uniform 0.05 - 0.50 A m^2 (isotropic set)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .orbit import CircularOrbit
from .simulate import DetumbleConfig
from .spacecraft import Magnetorquer, inertia_from_diagonal

#: Default detumble threshold, 1 deg/s.  Chosen well above the orbital rate
#: (about 0.063 deg/s at 500 km), because B-dot cannot drive the body rate
#: below roughly the orbital rate - see README "Limitations".
DEFAULT_TARGET_RATE_RAD_S: float = np.radians(1.0)


@dataclass(frozen=True)
class Scenario:
    """One sampled detumble case.

    Attributes
    ----------
    inertia : ndarray (3, 3)
        Principal-axis inertia [kg m^2].
    orbit : CircularOrbit
    magnetorquer : Magnetorquer
    omega0_rad_s : ndarray (3,)
        Initial body rate [rad/s].
    q0 : ndarray (4,)
        Initial scalar-first quaternion.
    seed : int
        The seed that produced this scenario.
    """

    inertia: NDArray[np.float64]
    orbit: CircularOrbit
    magnetorquer: Magnetorquer
    omega0_rad_s: NDArray[np.float64]
    q0: NDArray[np.float64]
    seed: int

    @property
    def inertia_scale_kgm2(self) -> float:
        """Mean principal moment [kg m^2], the scalar used by ``analytic.py``."""
        return float(np.mean(np.diag(self.inertia)))

    @property
    def rate0_rad_s(self) -> float:
        """Initial rate magnitude [rad/s]."""
        return float(np.linalg.norm(self.omega0_rad_s))

    def to_config(
        self,
        duration_s: float = 23000.0,
        control_dt_s: float = 2.0,
        substeps: int = 2,
        target_rate_rad_s: float = DEFAULT_TARGET_RATE_RAD_S,
        stop_when_detumbled: bool = True,
        mag_noise_t: float = 0.0,
    ) -> DetumbleConfig:
        """Build a ``DetumbleConfig`` for this scenario."""
        return DetumbleConfig(
            inertia=self.inertia,
            orbit=self.orbit,
            magnetorquer=self.magnetorquer,
            omega0_rad_s=self.omega0_rad_s,
            q0=self.q0,
            duration_s=duration_s,
            control_dt_s=control_dt_s,
            substeps=substeps,
            target_rate_rad_s=target_rate_rad_s,
            mag_noise_t=mag_noise_t,
            seed=self.seed,
            stop_when_detumbled=stop_when_detumbled,
        )


def _random_unit(rng: np.random.Generator) -> NDArray[np.float64]:
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    while n < 1e-9:  # pragma: no cover - probability ~0
        v = rng.normal(size=3)
        n = float(np.linalg.norm(v))
    return v / n


def sample_scenario(seed: int) -> Scenario:
    """Draw one scenario deterministically from ``seed``."""
    rng = np.random.default_rng(int(seed))
    j_scale = float(10.0 ** rng.uniform(np.log10(0.01), np.log10(0.30)))
    for _ in range(200):
        ratios = rng.uniform(0.6, 1.6, size=3)
        a, b, c = (j_scale * ratios).tolist()
        if a + b >= c and b + c >= a and c + a >= b:
            inertia = inertia_from_diagonal(a, b, c)
            break
    else:  # pragma: no cover - the loop above succeeds with probability ~1
        inertia = inertia_from_diagonal(j_scale, j_scale, j_scale)
    rate = float(np.radians(10.0 ** rng.uniform(np.log10(3.0), np.log10(20.0))))
    omega0 = rate * _random_unit(rng)
    orbit = CircularOrbit(
        altitude_km=float(rng.uniform(400.0, 800.0)),
        inclination_deg=float(rng.uniform(20.0, 100.0)),
        raan_deg=float(rng.uniform(0.0, 360.0)),
        arg_lat0_deg=float(rng.uniform(0.0, 360.0)),
        gmst0_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    mtq = Magnetorquer.isotropic(
        float(10.0 ** rng.uniform(np.log10(0.05), np.log10(0.50)))
    )
    q0 = rng.normal(size=4)
    q0 = q0 / float(np.linalg.norm(q0))
    if q0[0] < 0.0:
        q0 = -q0
    return Scenario(
        inertia=inertia,
        orbit=orbit,
        magnetorquer=mtq,
        omega0_rad_s=omega0,
        q0=q0,
        seed=int(seed),
    )


def sample_scenarios(n: int, seed0: int = 0) -> list[Scenario]:
    """Draw ``n`` scenarios with seeds ``seed0, seed0+1, ...``."""
    if int(n) < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [sample_scenario(seed0 + i) for i in range(int(n))]
