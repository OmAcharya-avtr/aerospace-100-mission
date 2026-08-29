"""End-to-end closed-loop adaptive-optics simulation.

Timing convention
-----------------
The loop is discrete with frame interval ``dt = 1/f_s``. At frame ``k``:

1. the atmosphere presents ``phi_atm[k]``;
2. the mirror is already holding the command ``c[k]`` decided at frame ``k-1``,
   so the residual seen by both the science channel and the sensor is
   ``phi_res[k] = phi_atm[k] + phi_DM(c[k])``;
3. the sensor measures ``s[k]`` from ``phi_res[k]`` with noise;
4. the controller consumes ``s[k]`` and, ``L`` frames later, changes the
   command.

``L >= 1`` is the loop latency in frames (see :mod:`waveforge.control`).

Controllers
-----------
``"integrator"``
    The classical baseline: ``c[k+1] = (1-leak) c[k] - g R s[k-L+1]``, with
    ``R`` the least-squares reconstructor. This is implemented first and is the
    reference every other controller is measured against.

``"polc_delay"``
    Pseudo-open-loop control with **no prediction** -- the pure-delay baseline.
    The open-loop slope estimate is reconstructed as
    ``s_OL[k] = s[k] - M c[k]`` (``M`` the interaction matrix), and the command
    applied is ``c[k+1] = (1-g) c[k] - g R s_OL[k]``. It assumes the atmosphere
    ``L`` frames from now looks like it does now; that assumption is exactly
    what a predictor is supposed to improve on.

``"polc_predict"``
    Same structure, but ``s_OL[k]`` is replaced by a forecast of
    ``s_OL[k+L]`` produced by a
    :class:`~waveforge.predictor.SlopePredictor`.

Pseudo-open-loop control is standard practice; see e.g. Piatrou, P. &
Gilles, L. (2005), "Robustness study of the pseudo-open-loop controller for
multiconjugate adaptive optics", *Applied Optics* **44**, 1003-1010.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .atmosphere import FrozenFlow
from .budget import strehl_exact
from .control import Integrator
from .dm import DeformableMirror
from .pupil import Pupil
from .sensor import ShackHartmann

__all__ = ["AOSystem", "LoopResult", "run_closed_loop"]


@dataclass
class AOSystem:
    """A calibrated AO system: pupil, sensor, mirror, interaction and reconstruction.

    Parameters
    ----------
    pupil, sensor, dm:
        The components. ``sensor`` and ``dm`` must share ``pupil``.
    n_modes:
        Number of singular values retained in the reconstructor [-]. ``None``
        (default) keeps every mode whose singular value exceeds
        ``rcond * s_max``.
    rcond:
        Relative singular-value cutoff [-]. Default 0.05, which removes the
        badly-seen waffle and edge modes of a Fried-geometry least-squares
        reconstructor.

    Attributes
    ----------
    interaction:
        ``(n_slopes, n_actuators)`` matrix ``M``: slopes produced by unit
        commands [rad/m per rad].
    reconstructor:
        ``(n_actuators, n_slopes)`` matrix ``R``: regularised pseudo-inverse
        of ``M`` [rad per rad/m].
    """

    pupil: Pupil
    sensor: ShackHartmann
    dm: DeformableMirror
    n_modes: int | None = None
    rcond: float = 0.05
    interaction: NDArray[np.float64] = field(init=False, repr=False)
    reconstructor: NDArray[np.float64] = field(init=False, repr=False)
    singular_values: NDArray[np.float64] = field(init=False, repr=False)
    n_modes_kept: int = field(init=False)

    def __post_init__(self) -> None:
        if self.sensor.pupil is not self.pupil or self.dm.pupil is not self.pupil:
            raise ValueError("sensor and dm must be built on the same Pupil object")
        rc = float(self.rcond)
        if not (0.0 < rc < 1.0):
            raise ValueError(f"rcond must be in (0, 1), got {self.rcond!r}")
        self.rcond = rc
        self.interaction = self._build_interaction()
        u, s, vt = np.linalg.svd(self.interaction, full_matrices=False)
        self.singular_values = s
        if self.n_modes is None:
            keep = int(np.count_nonzero(s > self.rcond * s[0]))
        else:
            keep = int(self.n_modes)
            if not (1 <= keep <= s.size):
                raise ValueError(f"n_modes must be in [1, {s.size}], got {self.n_modes!r}")
        self.n_modes_kept = keep
        s_inv = np.zeros_like(s)
        s_inv[:keep] = 1.0 / s[:keep]
        self.reconstructor = (vt.T * s_inv) @ u.T

    def _build_interaction(self) -> NDArray[np.float64]:
        """Poke each actuator by 1 rad and record the slope response."""
        n_act = self.dm.n_actuators
        m = np.empty((self.sensor.n_slopes, n_act), dtype=np.float64)
        cmd = np.zeros(n_act)
        for a in range(n_act):
            cmd[:] = 0.0
            cmd[a] = 1.0
            m[:, a] = self.sensor.slopes(self.dm.shape(cmd, apply_stroke=False))
        return m

    @property
    def condition_number(self) -> float:
        """Condition number of the retained part of the interaction matrix [-]."""
        s = self.singular_values
        return float(s[0] / s[self.n_modes_kept - 1])


@dataclass
class LoopResult:
    """Per-frame closed-loop telemetry.

    Attributes
    ----------
    residual_variance:
        Piston-removed residual phase variance per frame [rad^2].
    input_variance:
        Piston-removed open-loop (uncorrected) variance per frame [rad^2].
    strehl:
        Exact Strehl ratio of the residual per frame [-].
    commands:
        ``(n_frames, n_actuators)`` applied command history [rad].
    saturated_fraction:
        Fraction of actuators at the stroke limit per frame [-].
    settled_from:
        Index of the first frame counted as settled (``burn_in``).
    """

    residual_variance: NDArray[np.float64]
    input_variance: NDArray[np.float64]
    strehl: NDArray[np.float64]
    commands: NDArray[np.float64]
    saturated_fraction: NDArray[np.float64]
    settled_from: int

    @property
    def mean_residual_variance(self) -> float:
        """Mean residual variance over the settled frames [rad^2]."""
        return float(np.mean(self.residual_variance[self.settled_from :]))

    @property
    def mean_strehl(self) -> float:
        """Mean exact Strehl over the settled frames [-]."""
        return float(np.mean(self.strehl[self.settled_from :]))

    @property
    def mean_input_variance(self) -> float:
        """Mean open-loop variance over the settled frames [rad^2]."""
        return float(np.mean(self.input_variance[self.settled_from :]))

    @property
    def diverged(self) -> bool:
        """True if the residual variance grew beyond 100x the open-loop level."""
        tail = self.residual_variance[self.settled_from :]
        if tail.size == 0:  # pragma: no cover - guarded by run_closed_loop
            return False
        return bool(
            not np.all(np.isfinite(tail))
            or np.max(tail) > 100.0 * max(float(np.mean(self.input_variance)), 1e-30)
        )


def run_closed_loop(
    system: AOSystem,
    flow: FrozenFlow,
    n_frames: int,
    gain: float,
    latency: int = 1,
    n_photons: float | None = None,
    read_noise: float = 0.0,
    leak: float = 0.0,
    controller: str = "integrator",
    predictor=None,
    dropout_rate: float = 0.0,
    burn_in: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> LoopResult:
    """Run the closed loop and return per-frame telemetry.

    Parameters
    ----------
    system:
        Calibrated :class:`AOSystem`.
    flow:
        Atmospheric sequence generator; ``flow.n_pupil`` must equal
        ``system.pupil.n_grid``.
    n_frames:
        Number of frames to run [-], >= 2.
    gain:
        Loop gain [-], > 0.
    latency:
        Loop latency in frames [-], >= 1.
    n_photons:
        Detected photo-electrons per subaperture per frame; ``None`` = noiseless.
    read_noise:
        Read noise [e- rms per pixel].
    leak:
        Integrator leak [-], in ``[0, 1)``.
    controller:
        ``"integrator"``, ``"polc_delay"`` or ``"polc_predict"``.
    predictor:
        Required for ``"polc_predict"``: an object with
        ``predict(history) -> (mean, std)`` where ``history`` is
        ``(n_history, n_slopes)`` most-recent-last.
    dropout_rate:
        Probability per subaperture per frame that its measurement is lost [-].
    burn_in:
        Frames to discard before averaging. Default ``max(20, 10*latency)``.
    rng:
        Seed or ``numpy.random.Generator`` for sensor noise and dropout.

    Returns
    -------
    LoopResult
    """
    n_frames = int(n_frames)
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    if flow.n_pupil != system.pupil.n_grid:
        raise ValueError(
            f"flow.n_pupil ({flow.n_pupil}) must equal pupil.n_grid ({system.pupil.n_grid})"
        )
    if controller not in ("integrator", "polc_delay", "polc_predict"):
        raise ValueError(
            "controller must be 'integrator', 'polc_delay' or 'polc_predict', "
            f"got {controller!r}"
        )
    if controller == "polc_predict" and predictor is None:
        raise ValueError("controller='polc_predict' requires a predictor")
    if controller != "integrator" and not (0.0 < float(gain) <= 1.0):
        raise ValueError(
            f"pseudo-open-loop controllers need a gain in (0, 1], got {gain!r}"
        )
    dr = float(dropout_rate)
    if not (0.0 <= dr < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate!r}")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    latency = int(latency)
    if burn_in is None:
        burn_in = max(20, 10 * latency)
    burn_in = int(min(max(0, burn_in), n_frames - 1))

    pupil = system.pupil
    dm = system.dm
    sensor = system.sensor
    n_act = dm.n_actuators

    integ = Integrator(n_act, gain=gain, latency=latency, leak=leak)
    history: list[NDArray[np.float64]] = []
    n_hist = getattr(predictor, "n_history", 1) if predictor is not None else 1

    res_var = np.empty(n_frames)
    in_var = np.empty(n_frames)
    strehl = np.empty(n_frames)
    commands = np.empty((n_frames, n_act))
    sat_frac = np.empty(n_frames)

    command = np.zeros(n_act)
    # Queue of commands so POLC controllers respect the same latency.
    cmd_queue: list[NDArray[np.float64]] = [command.copy() for _ in range(latency - 1)]

    for k in range(n_frames):
        phi = flow.frame(k)
        dm_phase = dm.shape(command, apply_stroke=True)
        residual = pupil.piston_removed(phi + dm_phase)
        res_var[k] = pupil.variance(residual)
        in_var[k] = pupil.variance(phi)
        strehl[k] = strehl_exact(residual, pupil)
        commands[k] = command
        sat_frac[k] = float(np.mean(dm.saturated(command)))
        if not np.isfinite(res_var[k]) or res_var[k] > 1.0e12:
            res_var[k:] = np.inf
            strehl[k:] = 0.0
            commands[k:] = command
            sat_frac[k:] = sat_frac[k]
            in_var[k:] = in_var[k]
            break

        drop = None
        if dr > 0.0:
            drop = rng.random(sensor.n_valid) < dr
        slopes = sensor.measure(
            residual, n_photons=n_photons, read_noise=read_noise, rng=rng, dropout=drop
        )

        if controller == "integrator":
            command = integ.step(-system.reconstructor @ slopes)
        else:
            applied = dm.clip(command)
            open_loop = slopes - system.interaction @ applied
            history.append(open_loop)
            if len(history) > n_hist:
                history.pop(0)
            if controller == "polc_predict" and len(history) == n_hist:
                target, _ = predictor.predict(np.asarray(history))
            else:
                target = open_loop
            desired = -system.reconstructor @ target
            new_cmd = (1.0 - gain) * applied + gain * desired
            cmd_queue.append(new_cmd)
            command = cmd_queue.pop(0)

    return LoopResult(
        residual_variance=res_var,
        input_variance=in_var,
        strehl=strehl,
        commands=commands,
        saturated_fraction=sat_frac,
        settled_from=burn_in,
    )
