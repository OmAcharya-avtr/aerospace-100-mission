"""Command-line interface: ``python -m waveforge``.

Subcommands
-----------
``budget``
    Print the analytic residual-error budget and Strehl for a design point.
``stability``
    Print the integrator stability gain limit versus loop latency.
``loop``
    Run a closed-loop simulation and print the measured residual variance,
    Strehl and error-budget comparison.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .budget import (
    ErrorBudget,
    fitting_error,
    noise_error,
    noise_propagation_coefficient,
    strehl_marechal,
    temporal_error,
)
from .control import stability_gain_limit
from .loop import run_closed_loop
from .presets import ReferenceConfig, build_flow, build_system


def _config_from_args(args: argparse.Namespace) -> ReferenceConfig:
    return ReferenceConfig(
        diameter=args.diameter,
        wavelength=args.wavelength,
        r0=args.r0,
        wind_speed=args.wind_speed,
        frame_rate=args.frame_rate,
        n_grid=args.n_grid,
        n_sub=args.n_sub,
        n_act=args.n_act,
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--diameter", type=float, default=0.50, help="aperture diameter [m]")
    p.add_argument("--wavelength", type=float, default=1550e-9, help="wavelength [m]")
    p.add_argument("--r0", type=float, default=0.10, help="Fried parameter [m]")
    p.add_argument("--wind-speed", type=float, default=10.0, help="wind speed [m/s]")
    p.add_argument("--frame-rate", type=float, default=1000.0, help="frame rate [Hz]")
    p.add_argument("--n-grid", type=int, default=64, help="pupil samples across D")
    p.add_argument("--n-sub", type=int, default=8, help="subapertures across D")
    p.add_argument("--n-act", type=int, default=9, help="actuators across D")
    p.add_argument("--latency", type=int, default=2, help="loop latency [frames]")
    p.add_argument("--gain", type=float, default=0.4, help="loop gain [-]")
    p.add_argument(
        "--photons", type=float, default=None, help="detected e- per subaperture per frame"
    )
    p.add_argument("--read-noise", type=float, default=0.0, help="read noise [e- rms/px]")


def _budget(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    system = build_system(cfg)
    sig_fit = fitting_error(cfg.actuator_pitch, cfg.r0)
    sig_tmp = temporal_error(args.latency * cfg.dt, cfg.coherence_time)
    if args.photons is None:
        sig_noise = 0.0
        prop = 0.0
        slope_var = 0.0
    else:
        prop = noise_propagation_coefficient(
            system.reconstructor, system.dm.influence, system.pupil.n_valid
        )
        slope_var = system.sensor.noise_variance(args.photons, args.read_noise)
        sig_noise = noise_error(slope_var, prop)
    budget = ErrorBudget(fitting=sig_fit, temporal=sig_tmp, noise=sig_noise)
    print(f"waveforge {__version__} -- analytic residual-error budget")
    print(f"  D = {cfg.diameter:.3f} m, lambda = {cfg.wavelength * 1e9:.0f} nm, "
          f"r0 = {cfg.r0:.3f} m, D/r0 = {cfg.d_over_r0:.2f}")
    print(f"  actuator pitch {cfg.actuator_pitch * 1e3:.1f} mm, subaperture "
          f"{cfg.subaperture_size * 1e3:.1f} mm, {system.sensor.n_valid} lit subapertures")
    print(f"  f_G = {cfg.greenwood_frequency:.2f} Hz, tau0 = {cfg.coherence_time * 1e3:.3f} ms, "
          f"frame {cfg.dt * 1e3:.3f} ms, latency {args.latency} frames")
    if args.photons is not None:
        print(f"  slope noise sigma = {np.sqrt(slope_var):.4g} rad/m, "
              f"noise propagation p = {prop:.4g} m^2")
    print("  ---- variance [rad^2] ----")
    for key, value in budget.as_dict().items():
        if key == "strehl":
            print(f"  {key:<9s} {value:10.4f}")
        else:
            print(f"  {key:<9s} {value:10.4f}")
    print(f"  residual wavefront RMS = {budget.rms_wavefront(cfg.wavelength) * 1e9:.2f} nm")
    return 0


def _stability(args: argparse.Namespace) -> int:
    print("integrator stability limit vs loop latency (pure frame delay)")
    print("  latency [frames]    max gain")
    for lat in range(1, args.max_latency + 1):
        print(f"  {lat:<18d}  {stability_gain_limit(lat):.6f}")
    return 0


def _loop(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    system = build_system(cfg)
    flow = build_flow(cfg, seed=args.seed)
    result = run_closed_loop(
        system,
        flow,
        n_frames=args.frames,
        gain=args.gain,
        latency=args.latency,
        n_photons=args.photons,
        read_noise=args.read_noise,
        rng=args.seed,
    )
    sig_fit = fitting_error(cfg.actuator_pitch, cfg.r0)
    sig_tmp = temporal_error(args.latency * cfg.dt, cfg.coherence_time)
    print(f"waveforge {__version__} -- closed loop, {args.frames} frames, "
          f"gain {args.gain}, latency {args.latency}")
    print(f"  open-loop variance      {result.mean_input_variance:10.4f} rad^2")
    print(f"  residual variance       {result.mean_residual_variance:10.4f} rad^2")
    print(f"  Strehl (exact)          {result.mean_strehl:10.4f}")
    print(f"  Strehl (Marechal)       {strehl_marechal(result.mean_residual_variance):10.4f}")
    print(f"  analytic fitting term   {sig_fit:10.4f} rad^2")
    print(f"  analytic temporal term  {sig_tmp:10.4f} rad^2")
    print(f"  saturated actuators     {float(np.mean(result.saturated_fraction)) * 100:9.2f} %")
    print(f"  diverged                {str(result.diverged):>10s}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="waveforge", description="Adaptive-optics design and simulation toolkit"
    )
    parser.add_argument("--version", action="version", version=f"waveforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_budget = sub.add_parser("budget", help="analytic residual-error budget")
    _add_common(p_budget)
    p_budget.set_defaults(func=_budget)

    p_stab = sub.add_parser("stability", help="integrator stability limit vs latency")
    p_stab.add_argument("--max-latency", type=int, default=6)
    p_stab.set_defaults(func=_stability)

    p_loop = sub.add_parser("loop", help="run a closed-loop simulation")
    _add_common(p_loop)
    p_loop.add_argument("--frames", type=int, default=600)
    p_loop.add_argument("--seed", type=int, default=0)
    p_loop.set_defaults(func=_loop)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
