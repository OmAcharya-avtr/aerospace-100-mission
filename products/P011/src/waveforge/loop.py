"""End-to-end closed-loop adaptive-optics simulation.

Frame ``k`` of the loop, with total latency ``d`` frames (see
:mod:`waveforge.control`):

1. the atmosphere presents ``phi_k`` on the pupil;
2. the mirror holds the command ``c_{k-1}`` computed on the previous frame, so
   the residual is ``phi_res = phi_k - M^T c_{k-1}``;
3. the Shack-Hartmann sensor measures ``s_k = G phi_res + noise``;
4. the controller consumes the measurement from ``d - 1`` frames ago and
   updates ``c_k = leak c_{k-1} + g R s_{k-(d-1)}``, clipped to the actuator
   stroke.  The mirror then holds ``c_k`` for frame ``k+1``, which supplies the
   remaining frame of the total latency ``d``.

Substituting into the scalar loop equations reproduces exactly the rejection
transfer function of :mod:`waveforge.control` Eq. (2) with the same ``d``; the
agreement is measured in ``validation/validate_rejection_tf.py``.

With a predictor attached, step 4 instead uses a *pseudo-open-loop* slope
prediction.  Pseudo-open-loop (POL) slopes reconstruct what the sensor would
have seen with a flat mirror:

    s_pol,k = s_k + D_int c_{k-1}        with   D_int = G M^T                (1)

which is exact for the linear sensor model used here.  The newest POL frame
available to the controller at frame ``k`` is ``s_pol,k-(d-1)``; the command
computed now shapes the mirror for frame ``k+1``, so a predictor must forecast
exactly ``d`` frames ahead, and the controller drives the loop with

    increment = R ( s_pol_hat - D_int c_{k-1} )                              (2)

Setting the forecast to the newest available POL frame gives the *pure-delay*
baseline: identical machinery, no prediction.  Sources for POL /
pseudo-open-loop control: B. L. Ellerbroek and C. R. Vogel, *Inverse Problems*
**25**, 063001 (2009), Sec. 2; L. Gilles, *Appl. Opt.* **44**, 993-1002 (2005).

Units: phases and commands in radians at the sensing wavelength, slopes in
rad/m, times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .atmosphere import FrozenFlowAtmosphere
from .control import Integrator, noise_variance_gain
from .dm import DeformableMirror
from .errorbudget import ErrorBudget, fitting_error, noise_error
from .pupil import PupilGrid, piston_removed, strehl_from_field, variance
from .sensor import ShackHartmann
from .statistics import greenwood_time_constant

__all__ = ["AOConfig", "AOSystem", "LoopResult", "SlopePredictor"]


class SlopePredictor(Protocol):
    """Interface a predictive controller must satisfy.

    ``history`` is a ``(n_history, n_slopes)`` array of pseudo-open-loop slope
    vectors, oldest first.  ``predict`` returns the forecast slope vector and a
    one-sigma uncertainty of the same length (``None`` if the model does not
    provide one).  ``horizon`` is how many frames ahead the forecast is; it
    must equal the loop latency ``d``, and ``None`` means "any" (used by the
    identity / pure-delay baseline).
    """

    n_history: int
    horizon: int | None

    def predict(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Forecast the next pseudo-open-loop slope vector."""
        ...


@dataclass(frozen=True)
class AOConfig:
    """Configuration of a complete AO simulation.

    All defaults describe a small free-space-optical terminal: a 0.5 m
    aperture at 1.55 um through ``r0 = 0.10 m`` turbulence (``D/r0 = 5``) with
    10 m/s of wind, an 8x8 Shack-Hartmann and a 9x9 deformable mirror running
    at 1 kHz.
    """

    n_pix: int = 64
    diameter_m: float = 0.5
    r0_m: float = 0.10
    wavelength_m: float = 1.55e-6
    wind_speed_m_s: float = 10.0
    frame_rate_hz: float = 1000.0
    n_sub: int = 8
    n_act: int = 9
    coupling: float = 0.15
    stroke_rad: float = float("inf")
    margin_actuators: int = 1
    gain: float = 0.4
    delay_frames: int = 2
    leak: float = 1.0
    photon_flux: float = float("inf")
    read_noise_e: float = 0.0
    dropout_probability: float = 0.0
    screen_pixels: int = 1024
    n_subharmonics: int = 6
    seed: int = 0
    n_filtered_modes: int = 1
    condition_threshold: float = 1e-3

    def __post_init__(self) -> None:
        if not np.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0.0:
            raise ValueError(f"frame_rate_hz must be finite and > 0, got {self.frame_rate_hz!r}")
        if int(self.n_filtered_modes) != self.n_filtered_modes or self.n_filtered_modes < 0:
            raise ValueError(
                f"n_filtered_modes must be a non-negative integer, got {self.n_filtered_modes!r}"
            )
        if not (0.0 < self.condition_threshold < 1.0):
            raise ValueError(
                f"condition_threshold must lie in (0, 1), got {self.condition_threshold!r}"
            )

    @property
    def frame_time_s(self) -> float:
        """WFS frame period ``T = 1 / f_s`` [s]."""
        return 1.0 / self.frame_rate_hz

    @property
    def d_over_r0(self) -> float:
        """Aperture in Fried-parameter units."""
        return self.diameter_m / self.r0_m


@dataclass(frozen=True)
class LoopResult:
    """Per-frame record of a closed-loop run.

    Attributes
    ----------
    residual_variance:
        Piston-removed residual phase variance per frame [rad^2].
    open_loop_variance:
        Piston-removed input phase variance per frame [rad^2].
    strehl:
        Numerical Strehl from the residual complex field per frame.
    saturated_fraction:
        Fraction of actuators clipped by the stroke limit per frame.
    prediction_sigma:
        Mean predictive one-sigma per frame [rad/m], or ``None``.
    warmup_frames:
        Frames excluded from the summary statistics (loop closing transient).
    """

    residual_variance: np.ndarray
    open_loop_variance: np.ndarray
    strehl: np.ndarray
    saturated_fraction: np.ndarray
    prediction_sigma: np.ndarray | None
    warmup_frames: int
    diverged: bool

    @property
    def n_frames(self) -> int:
        """Total simulated frames."""
        return int(self.residual_variance.size)

    def _tail(self, array: np.ndarray) -> np.ndarray:
        if self.warmup_frames >= array.size:
            raise ValueError("warmup_frames covers the whole run; nothing to summarise")
        return array[self.warmup_frames :]

    @property
    def mean_residual_variance(self) -> float:
        """Mean residual variance after the warm-up [rad^2]."""
        return float(np.mean(self._tail(self.residual_variance)))

    @property
    def mean_open_loop_variance(self) -> float:
        """Mean input variance after the warm-up [rad^2]."""
        return float(np.mean(self._tail(self.open_loop_variance)))

    @property
    def mean_strehl(self) -> float:
        """Mean numerical Strehl after the warm-up."""
        return float(np.mean(self._tail(self.strehl)))

    @property
    def rejection_db(self) -> float:
        """Variance rejection ``10 log10(open loop / residual)`` [dB]."""
        return float(10.0 * np.log10(self.mean_open_loop_variance / self.mean_residual_variance))

    @property
    def max_saturated_fraction(self) -> float:
        """Worst per-frame actuator saturation fraction."""
        return float(np.max(self.saturated_fraction))


@dataclass
class AOSystem:
    """Assembled AO system: atmosphere, sensor, mirror, reconstructor, loop."""

    config: AOConfig = field(default_factory=AOConfig)
    pupil: PupilGrid = field(init=False)
    atmosphere: FrozenFlowAtmosphere = field(init=False)
    sensor: ShackHartmann = field(init=False)
    mirror: DeformableMirror = field(init=False)

    def __post_init__(self) -> None:
        c = self.config
        self.pupil = PupilGrid(c.n_pix, c.diameter_m)
        self.atmosphere = FrozenFlowAtmosphere(
            pupil=self.pupil,
            r0_m=c.r0_m,
            wind_speed_m_s=c.wind_speed_m_s,
            frame_time_s=c.frame_time_s,
            screen_pixels=c.screen_pixels,
            n_subharmonics=c.n_subharmonics,
            seed=c.seed,
        )
        self.sensor = ShackHartmann(
            pupil=self.pupil,
            n_sub=c.n_sub,
            wavelength_m=c.wavelength_m,
            photon_flux=c.photon_flux,
            read_noise_e=c.read_noise_e,
            dropout_probability=c.dropout_probability,
        )
        self.mirror = DeformableMirror(
            pupil=self.pupil,
            n_act=c.n_act,
            coupling=c.coupling,
            stroke_rad=c.stroke_rad,
            margin_actuators=c.margin_actuators,
        )
        self._interaction = self.sensor.operator @ self.mirror.influence_matrix.T
        self._reconstructor = self._build_reconstructor()

    # -- linear algebra ---------------------------------------------------
    @property
    def interaction_matrix(self) -> np.ndarray:
        """``D_int = G M^T`` with shape ``(n_slopes, n_actuators)`` [1/m]."""
        return self._interaction

    @property
    def reconstructor(self) -> np.ndarray:
        """Truncated pseudo-inverse of the interaction matrix, ``(n_act, n_slopes)``."""
        return self._reconstructor

    @property
    def propagation_matrix(self) -> np.ndarray:
        """``P = M^T R``: slope vector to pupil phase estimate ``(n_pupil, n_slopes)``."""
        return self.mirror.influence_matrix.T @ self._reconstructor

    def _build_reconstructor(self) -> np.ndarray:
        u, s, vt = np.linalg.svd(self._interaction, full_matrices=False)
        keep = s > self.config.condition_threshold * s[0]
        n_filtered = int(self.config.n_filtered_modes)
        if n_filtered > 0:
            kept_indices = np.flatnonzero(keep)
            if kept_indices.size <= n_filtered:
                raise ValueError(
                    "n_filtered_modes removes every controlled mode; "
                    f"only {kept_indices.size} survive the condition threshold"
                )
            keep[kept_indices[-n_filtered:]] = False
        s_inv = np.where(keep, 1.0 / np.where(s > 0, s, 1.0), 0.0)
        return (vt.T * s_inv) @ u.T

    @property
    def n_controlled_modes(self) -> int:
        """Number of retained singular modes in the reconstructor."""
        return int(np.linalg.matrix_rank(self._reconstructor))

    # -- open loop --------------------------------------------------------
    def open_loop_slopes(self, n_frames: int, start_frame: int = 0) -> np.ndarray:
        """Noise-free open-loop slope sequence, shape ``(n_frames, n_slopes)`` [rad/m].

        Used to build training and test data for the learned predictor.  No
        measurement noise is added: the predictor's training target is the true
        atmospheric slope, and noise is injected separately at evaluation time
        so that train/test noise realisations never coincide.
        """
        if int(n_frames) != n_frames or n_frames < 1:
            raise ValueError(f"n_frames must be an integer >= 1, got {n_frames!r}")
        if int(start_frame) != start_frame or start_frame < 0:
            raise ValueError(f"start_frame must be a non-negative integer, got {start_frame!r}")
        out = np.empty((int(n_frames), self.sensor.n_slopes))
        for k in range(int(n_frames)):
            out[k] = self.sensor.true_slopes(self.atmosphere.frame(int(start_frame) + k))
        return out

    # -- analytic budget --------------------------------------------------
    def error_budget(self, fitting_coefficient: float | None = None) -> ErrorBudget:
        """Analytic error budget for this configuration [rad^2].

        The fitting term uses :func:`waveforge.errorbudget.fitting_error`, the
        temporal term the pure-delay expression with the *effective* loop delay
        ``d / f_s``, and the noise term the exact linear propagation of the
        sensor's slope noise through this system's reconstructor, amplified by
        the closed-loop noise gain.
        """
        c = self.config
        coeff = fitting_error(
            self.mirror.pitch_m,
            c.r0_m,
            **({} if fitting_coefficient is None else {"coefficient": fitting_coefficient}),
        )
        tau0 = greenwood_time_constant(c.r0_m, c.wind_speed_m_s)
        temporal = float((c.delay_frames * c.frame_time_s / tau0) ** (5.0 / 3.0))
        eta = noise_variance_gain(c.gain, c.delay_frames, c.leak)
        noise = (
            0.0
            if not np.isfinite(eta)
            else noise_error(self.sensor.slope_noise_sigma(), self.propagation_matrix, eta)
        )
        return ErrorBudget(fitting=coeff, temporal=temporal, noise=noise)

    # -- closed loop ------------------------------------------------------
    def run(
        self,
        n_frames: int = 500,
        *,
        warmup_frames: int = 100,
        predictor: SlopePredictor | None = None,
        start_frame: int = 0,
        rng: np.random.Generator | int | None = 12345,
        divergence_threshold: float = 1e6,
        gain: float | None = None,
        delay_frames: int | None = None,
    ) -> LoopResult:
        """Run the closed loop and return per-frame diagnostics.

        Parameters
        ----------
        n_frames:
            Frames to simulate, ``>= 2``.
        warmup_frames:
            Frames excluded from the summary statistics, ``>= 0`` and
            ``< n_frames``.
        predictor:
            Optional :class:`SlopePredictor`.  When given, the controller runs
            on predicted pseudo-open-loop slopes, Eq. (2).
        start_frame:
            Offset into the frozen-flow sequence (use disjoint offsets for
            independent runs).
        rng:
            Seed or generator for sensor noise and dropout.
        divergence_threshold:
            Residual variance [rad^2] above which the run is declared diverged
            and stopped; the remaining frames are filled with the last value
            and ``LoopResult.diverged`` is set.
        gain, delay_frames:
            Override the configured loop gain and latency for this run only.
            Neither affects the atmosphere, the sensor or the mirror, so a gain
            or latency scan re-uses one assembled system instead of rebuilding
            (and re-drawing) it.
        """
        if int(n_frames) != n_frames or n_frames < 2:
            raise ValueError(f"n_frames must be an integer >= 2, got {n_frames!r}")
        if int(warmup_frames) != warmup_frames or not (0 <= warmup_frames < n_frames):
            raise ValueError(f"warmup_frames must lie in [0, {n_frames}), got {warmup_frames!r}")
        if not np.isfinite(divergence_threshold) or divergence_threshold <= 0.0:
            raise ValueError(
                f"divergence_threshold must be finite and > 0, got {divergence_threshold!r}"
            )
        c = self.config
        loop_gain = float(c.gain if gain is None else gain)
        loop_delay = int(c.delay_frames if delay_frames is None else delay_frames)
        if not np.isfinite(loop_gain) or loop_gain <= 0.0:
            raise ValueError(f"gain must be finite and > 0, got {loop_gain!r}")
        if loop_delay < 1:
            raise ValueError(f"delay_frames must be an integer >= 1, got {loop_delay!r}")
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        horizon = getattr(predictor, "horizon", None) if predictor is not None else None
        if horizon is not None and int(horizon) != loop_delay:
            raise ValueError(
                f"predictor horizon {horizon} does not match the loop latency "
                f"{loop_delay}; a predictor must forecast exactly as far ahead as the "
                "loop is late"
            )
        # All of the latency lives in the explicit measurement buffer (d - 1
        # frames) plus the one-frame DM application delay carried by the
        # integrator, so the integrator itself is used with delay_frames = 1.
        integrator = Integrator(
            n_commands=self.mirror.n_actuators,
            gain=loop_gain,
            delay_frames=1,
            leak=c.leak,
        )
        n_frames = int(n_frames)
        residual_var = np.zeros(n_frames)
        open_var = np.zeros(n_frames)
        strehl = np.zeros(n_frames)
        saturated = np.zeros(n_frames)
        pred_sigma = np.zeros(n_frames) if predictor is not None else None

        command = np.zeros(self.mirror.n_actuators)
        n_lag = loop_delay - 1
        slope_buffer: list[np.ndarray] = [np.zeros(self.sensor.n_slopes) for _ in range(n_lag)]
        pol_buffer: list[np.ndarray] = [np.zeros(self.sensor.n_slopes) for _ in range(n_lag)]
        history: list[np.ndarray] = []
        n_history = int(getattr(predictor, "n_history", 1)) if predictor is not None else 1
        diverged = False

        for k in range(n_frames):
            phi = self.atmosphere.frame(int(start_frame) + k)
            open_var[k] = variance(phi, self.pupil.mask)
            residual = piston_removed(phi - self.mirror.surface(command), self.pupil.mask)
            residual_var[k] = variance(residual, self.pupil.mask)
            strehl[k] = strehl_from_field(residual, self.pupil.mask)
            if not np.isfinite(residual_var[k]) or residual_var[k] > divergence_threshold:
                diverged = True
                residual_var[k:] = residual_var[k]
                strehl[k:] = strehl[k]
                open_var[k:] = open_var[k]
                break

            measurement = self.sensor.measure(residual, generator)
            slope_buffer.append(measurement.slopes)
            pol_buffer.append(measurement.slopes + self._interaction @ command)
            slope_available = slope_buffer.pop(0)
            pol_available = pol_buffer.pop(0)

            if predictor is None:
                increment = self._reconstructor @ slope_available
            else:
                history.append(pol_available)
                if len(history) > n_history:
                    history.pop(0)
                if len(history) < n_history:
                    increment = self._reconstructor @ slope_available
                else:
                    forecast, sigma = predictor.predict(np.stack(history))
                    if pred_sigma is not None and sigma is not None:
                        pred_sigma[k] = float(np.mean(sigma))
                    increment = self._reconstructor @ (forecast - self._interaction @ command)
            command = integrator.step(increment)
            command, sat = self.mirror.clip(command)
            saturated[k] = sat

        return LoopResult(
            residual_variance=residual_var,
            open_loop_variance=open_var,
            strehl=strehl,
            saturated_fraction=saturated,
            prediction_sigma=pred_sigma,
            warmup_frames=int(warmup_frames),
            diverged=diverged,
        )
