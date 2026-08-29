"""Reference free-space-optical adaptive-optics configuration.

One place where the whole package's default system is defined, so that tests,
examples, validation scripts and the CLI all describe the same hardware. The
numbers are a plausible ground terminal for a 1550 nm free-space optical link;
they are a *design point*, not a measurement of any real system.

| Quantity | Value | Why |
|---|---|---|
| Aperture ``D`` | 0.50 m | typical ground FSO terminal |
| Wavelength | 1550 nm | telecom C band |
| Fried parameter ``r0`` | 0.10 m at 1550 nm | ``D/r0 = 5`` -- AO is needed but tractable |
| Wind speed | 10 m/s | single frozen layer |
| Frame rate | 1000 Hz | ``f_s / f_G = 23`` |
| Subapertures | 8 x 8 (52 lit) | ``d_sub = 62.5 mm`` |
| Actuators | 9 x 9 grid, Fried geometry | ``d_act = 62.5 mm`` |
| Grid | 64 x 64 over ``D`` | ``r0/dx = 12.8``, ``d_act/dx = 8`` |
| Screen | 1024 x 1024 (8.0 m) | 785 frames before the window wraps |

Derived, at these values: Greenwood frequency ``f_G = 0.426 v/r0 = 42.6 Hz``,
coherence time ``tau0 = 0.314 r0/v = 3.14 ms``, so a one-frame delay is
``0.318 tau0`` and the pure-delay temporal error is ``0.152 rad^2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import FrozenFlow, PhaseScreen, coherence_time, greenwood_frequency
from .dm import DeformableMirror
from .loop import AOSystem
from .pupil import Pupil
from .sensor import ShackHartmann

__all__ = ["ReferenceConfig", "build_system", "build_flow"]


@dataclass(frozen=True)
class ReferenceConfig:
    """Parameters of the reference AO design point. All SI units."""

    diameter: float = 0.50
    wavelength: float = 1550e-9
    r0: float = 0.10
    wind_speed: float = 10.0
    wind_direction: float = 0.3
    frame_rate: float = 1000.0
    n_grid: int = 64
    n_sub: int = 8
    n_act: int = 9
    coupling: float = 0.15
    stroke: float | None = None
    screen_n: int = 1024
    boiling: float = 0.0

    @property
    def dt(self) -> float:
        """Frame interval [s]."""
        return 1.0 / self.frame_rate

    @property
    def actuator_pitch(self) -> float:
        """DM actuator pitch [m]."""
        return self.diameter / (self.n_act - 1)

    @property
    def subaperture_size(self) -> float:
        """Shack-Hartmann subaperture size [m]."""
        return self.diameter / self.n_sub

    @property
    def greenwood_frequency(self) -> float:
        """Greenwood frequency [Hz]."""
        return greenwood_frequency(self.r0, self.wind_speed)

    @property
    def coherence_time(self) -> float:
        """Atmospheric coherence time [s]."""
        return coherence_time(self.r0, self.wind_speed)

    @property
    def d_over_r0(self) -> float:
        """``D/r0`` [-]."""
        return self.diameter / self.r0


def build_system(config: ReferenceConfig | None = None, rcond: float = 0.05) -> AOSystem:
    """Build and calibrate the reference :class:`~waveforge.loop.AOSystem`."""
    cfg = config or ReferenceConfig()
    pupil = Pupil(cfg.diameter, cfg.n_grid)
    pupil.check_sampling(r0=cfg.r0, actuator_pitch=cfg.actuator_pitch)
    sensor = ShackHartmann(pupil, cfg.n_sub, cfg.wavelength)
    dm = DeformableMirror(pupil, cfg.n_act, coupling=cfg.coupling, stroke=cfg.stroke)
    return AOSystem(pupil, sensor, dm, rcond=rcond)


def build_flow(
    config: ReferenceConfig | None = None, seed: int = 0
) -> FrozenFlow:
    """Build a frozen-flow atmospheric sequence for the reference configuration."""
    cfg = config or ReferenceConfig()
    dx = cfg.diameter / cfg.n_grid
    screen = PhaseScreen(n=cfg.screen_n, dx=dx, r0=cfg.r0).generate(
        np.random.default_rng(seed)
    )
    flow = FrozenFlow(
        screen,
        dx=dx,
        n_pupil=cfg.n_grid,
        wind_speed=cfg.wind_speed,
        wind_direction=cfg.wind_direction,
        dt=cfg.dt,
        boiling=cfg.boiling,
    )
    if cfg.boiling > 0.0:
        flow.set_boiling_screen(
            PhaseScreen(n=cfg.screen_n, dx=dx, r0=cfg.r0).generate(
                np.random.default_rng(seed + 500_000)
            )
        )
    return flow
