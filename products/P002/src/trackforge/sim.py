"""End-to-end PAT episode simulator: acquire -> track -> lose lock -> reacquire.

An episode chains the four modules:

1. **Acquire** (``trackforge.scan``): a spiral (or raster) scan sweeps the
   2-D Gaussian uncertainty region until the target falls inside the beam
   footprint and a per-dwell detection trial succeeds.
2. **Track** (``trackforge.dynamics`` + ``trackforge.control``): a two-axis
   gimbal closes a pointing loop on a noisy angle sensor while platform
   jitter, synthesised from a target PSD, perturbs the line of sight.
3. **Lose lock**: a disturbance spike (a scaled, windowed transient added
   to the jitter series) drives the LOS error past ``track_threshold`` for
   longer than ``loss_hold_s``; the tracker declares loss of lock.
4. **Reacquire** (``trackforge.reacq``): a scripted or learned policy
   chooses re-scan strategies until the target is re-found.

Every stage is driven by one seeded ``numpy.random.Generator`` chain, so a
scenario plus a seed reproduces an episode exactly.

Units: rad, rad/s, s, Hz, N m.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trackforge import reacq as _reacq
from trackforge import scan as _scan
from trackforge.control import (
    LQRController,
    PIDController,
    lqr_weights_from_bandwidth,
    pid_gains_from_bandwidth,
)
from trackforge.dynamics import AngleSensor, GimbalAxis, JitterPSD, synthesize_jitter

__all__ = [
    "Scenario",
    "EpisodeResult",
    "load_scenario",
    "run_episode",
    "run_monte_carlo",
    "sim_steps_per_second",
    "DEFAULT_SCENARIO",
]


@dataclass
class Scenario:
    """Validated scenario configuration (mirrors the YAML schema).

    See ``DEFAULT_SCENARIO`` and ``examples/scenario_leo_downlink.yaml`` for
    a fully commented instance.
    """

    name: str = "default"
    seed: int = 2026
    dt: float = 2.0e-4
    track_duration: float = 2.0

    # acquisition
    sigma_uncertainty: float = 3.0e-4
    beam_radius: float = 2.0e-5
    overlap: float = 0.25
    containment: float = 0.995
    dwell_time: float = 1.0e-3
    step_fraction: float = 0.5
    p_dwell: float = 0.9
    pattern: str = "spiral"

    # gimbal
    inertia: float = 0.05
    damping: float = 0.02
    torque_max: float = 2.0
    rate_max: float = 1.0
    accel_max: float | None = None

    # control
    controller: str = "pid"
    bandwidth_hz: float = 5.0
    damping_ratio: float = 0.707
    integral_alpha: float = 0.1
    lqr_q_angle: float = 1.0
    lqr_q_rate: float = 0.0
    lqr_r_torque: float | None = None  # None -> eq. (10) for bandwidth_hz

    # disturbance / sensor
    jitter_s0: float = 1.0e-12
    jitter_f_corner: float = 3.0
    jitter_order: float = 2.0
    spike_time: float = 1.2
    spike_amplitude: float = 1.5e-4
    spike_width: float = 0.02
    nea: float = 1.0e-6
    sensor_dropout: float = 0.0
    quantization: float | None = None

    # lock logic
    track_threshold: float = 4.0e-5
    loss_hold_s: float = 0.01

    # reacquisition
    reacq_policy: str = "always_local"
    reacq: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pattern not in ("spiral", "raster"):
            raise ValueError(f"pattern must be 'spiral' or 'raster', got {self.pattern!r}")
        if self.controller not in ("pid", "lqr"):
            raise ValueError(f"controller must be 'pid' or 'lqr', got {self.controller!r}")
        if self.reacq_policy not in ("always_local", "always_full", "learned"):
            raise ValueError(
                "reacq_policy must be one of 'always_local', 'always_full', 'learned', "
                f"got {self.reacq_policy!r}"
            )
        for name in ("dt", "track_duration", "sigma_uncertainty", "beam_radius",
                     "dwell_time", "inertia", "torque_max", "rate_max", "bandwidth_hz",
                     "track_threshold", "spike_width"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"scenario field {name!r} must be finite and > 0, got {v!r}")
            setattr(self, name, v)
        if self.spike_time < 0:
            raise ValueError(f"spike_time must be >= 0, got {self.spike_time!r}")
        if self.spike_time >= self.track_duration:
            raise ValueError(
                f"spike_time ({self.spike_time}) must be < track_duration "
                f"({self.track_duration})"
            )
        if not 0.0 < self.p_dwell <= 1.0:
            raise ValueError(f"p_dwell must be in (0, 1], got {self.p_dwell!r}")

    def to_dict(self) -> dict:
        """Return the scenario as a plain dict (YAML-serialisable)."""
        return asdict(self)

    def reacq_config(self) -> _reacq.ReacqConfig:
        """Build the reacquisition config, defaulting unspecified fields."""
        base = {
            "sigma0": self.sigma_uncertainty / 6.0,
            "sigma_lk": self.track_threshold * 1.5,
            "coverage_rate": _scan.track_spacing(self.beam_radius, self.overlap)
            * (self.step_fraction * self.beam_radius / self.dwell_time),
            "cone_radius": self.sigma_uncertainty * 3.0,
            "p_detect": self.p_dwell,
        }
        base.update(self.reacq or {})
        return _reacq.ReacqConfig(**base)


DEFAULT_SCENARIO = Scenario()


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a YAML scenario file.

    Unknown keys raise ``ValueError`` (typos are errors, not silent
    defaults).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scenario file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"scenario file must contain a mapping, got {type(raw).__name__}")
    known = set(Scenario.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown scenario keys: {unknown}; allowed keys: {sorted(known)}")
    return Scenario(**raw)


@dataclass
class EpisodeResult:
    """Outcome and time series of one end-to-end episode.

    Attributes
    ----------
    acquisition_time_s : float or None
        Time from scan start to first detection; None if never acquired.
    scan_points : int
        Dwell points in the designed pattern.
    track_rms_rad : float
        RMS LOS error over the pre-spike tracking interval.
    track_peak_rad : float
        Peak |LOS error| over the whole tracking run.
    lock_lost : bool
        Whether the spike caused the loss criterion to trigger.
    loss_time_s : float or None
        Time of loss declaration within the tracking run.
    reacq_time_s : float or None
        Elapsed time to reacquire after loss. An attempt in progress is not
        aborted, so on a timeout this can exceed ``ReacqConfig.max_time``
        (``reacq_success`` is then False). ``reacq.evaluate_policy`` censors
        at ``max_time``; this field does not.
    reacq_success : bool
        Whether reacquisition succeeded within ``max_time``.
    total_time_s : float
        acquisition + tracking + reacquisition.
    saturation_fraction : float
        Fraction of tracking steps in which the axis saturated torque or rate.
    t, los_error, torque : np.ndarray
        Tracking-phase time series ([s], [rad], [N m]).
    """

    acquisition_time_s: float | None
    scan_points: int
    scan_design_time_s: float
    track_rms_rad: float
    track_peak_rad: float
    lock_lost: bool
    loss_time_s: float | None
    reacq_time_s: float | None
    reacq_success: bool
    reacq_attempts: int
    total_time_s: float
    saturation_fraction: float
    t: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    los_error: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    torque: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    jitter: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def summary(self) -> dict:
        """Scalar metrics only (safe to tabulate / serialise)."""
        return {
            k: v
            for k, v in asdict(self).items()
            if not isinstance(v, np.ndarray)
        }


def _build_controller(sc: Scenario, axis: GimbalAxis):
    if sc.controller == "pid":
        kp, ki, kd = pid_gains_from_bandwidth(
            axis.inertia, 2.0 * math.pi * sc.bandwidth_hz, sc.damping_ratio, sc.integral_alpha
        )
        return PIDController(kp, ki, kd, axis.torque_max)
    r = sc.lqr_r_torque
    if r is None:
        _, _, r = lqr_weights_from_bandwidth(
            axis.inertia, 2.0 * math.pi * sc.bandwidth_hz, sc.lqr_q_angle
        )
    return LQRController(axis, q_angle=sc.lqr_q_angle, q_rate=sc.lqr_q_rate, r_torque=r)


def _disturbance_series(sc: Scenario, n: int, rng: np.random.Generator) -> np.ndarray:
    psd = JitterPSD(sc.jitter_s0, sc.jitter_f_corner, sc.jitter_order)
    fs = 1.0 / sc.dt
    d = synthesize_jitter(psd, n, fs, rng)
    t = np.arange(n) * sc.dt
    spike = sc.spike_amplitude * np.exp(-0.5 * ((t - sc.spike_time) / sc.spike_width) ** 2)
    return d + spike


def _detect_loss(t: np.ndarray, err: np.ndarray, sc: Scenario) -> float | None:
    """First time |err| exceeds the threshold continuously for ``loss_hold_s``."""
    over = np.abs(err) > sc.track_threshold
    need = max(int(round(sc.loss_hold_s / sc.dt)), 1)
    run = 0
    for k, flag in enumerate(over):
        run = run + 1 if flag else 0
        if run >= need:
            return float(t[k])
    return None


def run_episode(
    scenario: Scenario | None = None,
    seed: int | None = None,
    policy: Any = None,
    keep_series: bool = True,
) -> EpisodeResult:
    """Run one acquire -> track -> lose lock -> reacquire episode.

    Parameters
    ----------
    scenario : Scenario, optional
        Defaults to ``DEFAULT_SCENARIO``.
    seed : int, optional
        Overrides ``scenario.seed``.
    policy : object, optional
        Reacquisition policy exposing ``act(state)``. If None, built from
        ``scenario.reacq_policy`` (``learned`` requires an explicit policy
        and raises otherwise).
    keep_series : bool
        Keep the tracking time series in the result (set False for
        Monte Carlo runs to save memory).
    """
    sc = scenario or DEFAULT_SCENARIO
    if not isinstance(sc, Scenario):
        raise TypeError("scenario must be a Scenario instance")
    root = np.random.default_rng(sc.seed if seed is None else seed)
    # independent sub-streams so that changing one stage does not reshuffle others
    rng_acq, rng_jit, rng_sensor, rng_reacq = (
        np.random.default_rng(int(s)) for s in root.integers(0, 2**62, size=4)
    )

    # --- 1. acquisition -------------------------------------------------
    unc = _scan.GaussianUncertainty(sc.sigma_uncertainty)
    gen = _scan.spiral_scan if sc.pattern == "spiral" else _scan.raster_scan
    pattern = gen(
        unc,
        sc.beam_radius,
        overlap=sc.overlap,
        containment=sc.containment,
        dwell_time=sc.dwell_time,
        step_fraction=sc.step_fraction,
    )
    target = unc.sample(1, rng_acq)[0]
    t_acq = _scan.simulate_acquisition(pattern, target, p_dwell=sc.p_dwell, rng=rng_acq)

    # --- 2/3. tracking and loss of lock ---------------------------------
    n = int(round(sc.track_duration / sc.dt))
    axis = GimbalAxis(sc.inertia, sc.damping, sc.torque_max, sc.rate_max, sc.accel_max)
    ctrl = _build_controller(sc, axis)
    sensor = AngleSensor(sc.nea, sc.quantization, sc.sensor_dropout)
    dist = _disturbance_series(sc, n, rng_jit)

    t = np.arange(n) * sc.dt
    err = np.zeros(n)
    tau = np.zeros(n)
    n_sat = 0
    for k in range(n):
        los = axis.angle + dist[k]
        meas, _valid = sensor.measure(np.array([los, 0.0]), rng_sensor)
        u = ctrl.update(0.0, float(meas[0]), sc.dt)
        axis.step(u, sc.dt)
        # the controller clips its own output at torque_max, so command
        # saturation is detected on the command, not on the plant's clip
        n_sat += int(
            abs(u) >= 0.999 * sc.torque_max
            or axis.saturated_torque
            or axis.saturated_rate
        )
        err[k] = los
        tau[k] = u

    pre = t < max(sc.spike_time - 5.0 * sc.spike_width, sc.dt)
    track_rms = float(np.sqrt(np.mean(err[pre] ** 2))) if pre.any() else float("nan")
    loss_time = _detect_loss(t, err, sc)

    # --- 4. reacquisition ------------------------------------------------
    reacq_time: float | None = None
    reacq_ok = False
    n_attempts = 0
    if loss_time is not None:
        if policy is None:
            if sc.reacq_policy == "always_local":
                policy = _reacq.AlwaysLocalPolicy()
            elif sc.reacq_policy == "always_full":
                policy = _reacq.AlwaysFullPolicy()
            else:
                raise ValueError(
                    "scenario.reacq_policy == 'learned' but no policy was provided to "
                    "run_episode(policy=...)"
                )
        env = _reacq.ReacqEnv(sc.reacq_config())
        s = env.reset(seed=int(rng_reacq.integers(0, 2**31 - 1)))
        while True:
            s, _r, done, info = env.step(policy.act(s))
            if done:
                reacq_ok = bool(info.get("success", False))
                reacq_time = float(env.t)
                n_attempts = env.n_attempts
                break

    total = (t_acq or 0.0) + (loss_time if loss_time is not None else sc.track_duration)
    total += reacq_time or 0.0
    empty = np.zeros(0)
    return EpisodeResult(
        acquisition_time_s=t_acq,
        scan_points=pattern.n_points,
        scan_design_time_s=pattern.scan_time,
        track_rms_rad=track_rms,
        track_peak_rad=float(np.max(np.abs(err))),
        lock_lost=loss_time is not None,
        loss_time_s=loss_time,
        reacq_time_s=reacq_time,
        reacq_success=reacq_ok,
        reacq_attempts=n_attempts,
        total_time_s=float(total),
        saturation_fraction=n_sat / n,
        t=t if keep_series else empty,
        los_error=err if keep_series else empty,
        torque=tau if keep_series else empty,
        jitter=dist if keep_series else empty,
    )


def run_monte_carlo(
    scenario: Scenario | None = None,
    n_episodes: int = 20,
    base_seed: int = 1000,
    policy: Any = None,
) -> dict:
    """Run ``n_episodes`` episodes with seeds ``base_seed + i`` and aggregate.

    Returns a dict of aggregate metrics plus the per-episode summaries.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes!r}")
    sc = scenario or DEFAULT_SCENARIO
    rows = [
        run_episode(sc, seed=base_seed + i, policy=policy, keep_series=False).summary()
        for i in range(n_episodes)
    ]
    acq = np.array([r["acquisition_time_s"] for r in rows if r["acquisition_time_s"] is not None])
    rms = np.array([r["track_rms_rad"] for r in rows])
    reac = np.array([r["reacq_time_s"] for r in rows if r["reacq_time_s"] is not None])
    return {
        "scenario": sc.name,
        "n_episodes": n_episodes,
        "acquired_fraction": len(acq) / n_episodes,
        "mean_acquisition_time_s": float(np.mean(acq)) if acq.size else float("nan"),
        "mean_track_rms_rad": float(np.mean(rms)),
        "lock_loss_fraction": float(np.mean([r["lock_lost"] for r in rows])),
        "mean_reacq_time_s": float(np.mean(reac)) if reac.size else float("nan"),
        "reacq_success_rate": float(np.mean([r["reacq_success"] for r in rows])),
        "episodes": rows,
    }


def sim_steps_per_second(
    scenario: Scenario | None = None, duration: float = 0.5
) -> dict:
    """Measure closed-loop simulation throughput [steps/s].

    Runs the tracking inner loop only (no acquisition or reacquisition) for
    ``duration`` seconds of simulated time and reports wall-clock rate.
    """
    import time

    sc = scenario or DEFAULT_SCENARIO
    n = int(round(duration / sc.dt))
    axis = GimbalAxis(sc.inertia, sc.damping, sc.torque_max, sc.rate_max, sc.accel_max)
    ctrl = _build_controller(sc, axis)
    sensor = AngleSensor(sc.nea)
    rng = np.random.default_rng(0)
    dist = _disturbance_series(sc, n, rng)
    t0 = time.perf_counter()
    for k in range(n):
        los = axis.angle + dist[k]
        meas, _ = sensor.measure(np.array([los, 0.0]), rng)
        axis.step(ctrl.update(0.0, float(meas[0]), sc.dt), sc.dt)
    wall = time.perf_counter() - t0
    return {
        "steps": n,
        "wall_time_s": wall,
        "steps_per_second": n / wall,
        "realtime_factor": (n * sc.dt) / wall,
    }
