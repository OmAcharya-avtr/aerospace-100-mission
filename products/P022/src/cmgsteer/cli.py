"""Command-line interface: ``python -m cmgsteer {array,singularity,steer,manoeuvre}``.

Exit codes: 0 on success, 1 when a computed result fails the subcommand's own
acceptance check (a steering command that cannot be met, or a manoeuvre that
ends inside a singular region), 2 on invalid input.  Invalid input produces a
one-line diagnostic, never a traceback.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .arrays import STANDARD_PYRAMID_SKEW_DEG, CMGArray, pyramid_array, roof_array
from .dataset import manoeuvre_suite
from .nullmotion import GradientNullMotion, NoNullMotion
from .simulate import run_steering
from .singularity import classify_singularity, singular_configuration, singularity_measure
from .steering import METHODS, steer

CONFIGS = ("pyramid", "roof")
NULL_POLICIES = ("none", "gradient")


def _build_array(args: argparse.Namespace) -> CMGArray:
    if args.config == "pyramid":
        skew = STANDARD_PYRAMID_SKEW_DEG if args.skew is None else args.skew
        array = pyramid_array(skew_angle_deg=skew, rotor_momentum=args.rotor_momentum)
    else:
        skew = 45.0 if args.skew is None else args.skew
        array = roof_array(skew_angle_deg=skew, rotor_momentum=args.rotor_momentum)
    if args.failed:
        array = array.with_locked(args.failed)
    return array


def _deltas(args: argparse.Namespace, array: CMGArray) -> np.ndarray:
    if args.deltas is None:
        return np.zeros(array.n_cmgs)
    d = np.asarray(args.deltas, dtype=float)
    if d.shape[0] != array.n_cmgs:
        raise ValueError(f"--deltas needs {array.n_cmgs} values, got {d.shape[0]}")
    return np.radians(d)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", choices=CONFIGS, default="pyramid", help="array geometry")
    parser.add_argument("--skew", type=float, default=None, help="skew angle [deg]")
    parser.add_argument(
        "--rotor-momentum", type=float, default=1.0, help="rotor momentum h0 [N*m*s]"
    )
    parser.add_argument(
        "--failed", type=int, nargs="*", default=None, help="indices of locked (failed) gimbals"
    )
    parser.add_argument(
        "--deltas", type=float, nargs="*", default=None, help="gimbal angles [deg]"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cmgsteer",
        description="CMG array geometry, singularity analysis and steering laws.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_array = sub.add_parser("array", help="describe an array geometry")
    _add_common(p_array)

    p_sing = sub.add_parser("singularity", help="classify a configuration")
    _add_common(p_sing)
    p_sing.add_argument(
        "--direction",
        type=float,
        nargs=3,
        default=None,
        help="build an analytically singular configuration along this body direction",
    )
    p_sing.add_argument(
        "--signs",
        type=float,
        nargs="*",
        default=None,
        help="sign vector for --direction (+1/-1 per CMG); default all +1",
    )
    p_sing.add_argument("--tol", type=float, default=1e-8, help="relative singularity tolerance")

    p_steer = sub.add_parser("steer", help="one steering-law evaluation")
    _add_common(p_steer)
    p_steer.add_argument("--torque", type=float, nargs=3, required=True, help="body torque [N*m]")
    p_steer.add_argument("--method", choices=METHODS, default="sr", help="steering law")
    p_steer.add_argument("--lam", type=float, default=None, help="absolute robustness parameter")
    p_steer.add_argument("--lam0", type=float, default=0.01, help="adaptive lam0")
    p_steer.add_argument("--mu", type=float, default=10.0, help="adaptive mu")
    p_steer.add_argument(
        "--max-gimbal-rate", type=float, default=None, help="gimbal rate limit [rad/s]"
    )
    p_steer.add_argument(
        "--tolerance", type=float, default=1e-6, help="torque error [N*m] above which exit is 1"
    )

    p_man = sub.add_parser("manoeuvre", help="run one seeded manoeuvre")
    _add_common(p_man)
    p_man.add_argument("--seed", type=int, default=2026, help="manoeuvre suite seed")
    p_man.add_argument("--index", type=int, default=0, help="which manoeuvre of the suite")
    p_man.add_argument("--method", choices=METHODS, default="sr", help="steering law")
    p_man.add_argument("--null", choices=NULL_POLICIES, default="none", help="null-motion policy")
    p_man.add_argument("--gain", type=float, default=1.0, help="gradient null-motion gain")
    p_man.add_argument(
        "--max-gimbal-rate", type=float, default=2.0, help="gimbal rate limit [rad/s]"
    )
    p_man.add_argument(
        "--measure-floor",
        type=float,
        default=0.0,
        help="exit 1 if the singularity measure ever falls below this",
    )
    return parser


def _cmd_array(args: argparse.Namespace) -> int:
    array = _build_array(args)
    d = _deltas(args, array)
    print(array.summary())
    print(f"  momentum at these angles [N*m*s] : {np.array2string(array.momentum(d), precision=6)}")
    print(f"  singularity measure m            : {singularity_measure(array.jacobian(d)):.9g}")
    print(f"  free gimbals / null-space dim    : {array.n_free} / {max(0, array.n_free - 3)}")
    return 0


def _cmd_singularity(args: argparse.Namespace) -> int:
    array = _build_array(args)
    if args.direction is not None:
        signs = None if args.signs is None else np.asarray(args.signs, dtype=float)
        d = singular_configuration(array, np.asarray(args.direction, dtype=float), signs)
    else:
        d = _deltas(args, array)
    info = classify_singularity(array, d, tol=args.tol)
    print(f"gimbal angles [deg]      : {np.array2string(np.degrees(d), precision=6)}")
    print(f"momentum [N*m*s]         : {np.array2string(info.momentum, precision=9)}")
    print(f"singularity measure m    : {info.measure:.9e}")
    print(f"sigma_min [N*m*s/rad]    : {info.min_singular_value:.9e}")
    print(f"condition number         : {info.condition_number:.6e}")
    print(f"singular (tol={args.tol:g})     : {info.singular}")
    print(f"kind                     : {info.kind}")
    print(f"passability              : {info.passability}")
    print(f"singular direction u     : {np.array2string(info.direction, precision=6)}")
    print(f"signs eps_i              : {np.array2string(info.signs, precision=0)}")
    print(f"rank of A                : {info.rank}")
    return 0


def _cmd_steer(args: argparse.Namespace) -> int:
    array = _build_array(args)
    d = _deltas(args, array)
    kwargs: dict[str, object] = {
        "max_gimbal_rate": args.max_gimbal_rate,
    }
    if args.method in ("sr", "gsr"):
        kwargs["lam"] = args.lam
        kwargs["lam0"] = args.lam0
        kwargs["mu"] = args.mu
    result = steer(array, d, np.asarray(args.torque, dtype=float), method=args.method, **kwargs)
    print(f"config={args.config} method={result.method}")
    print(f"  gimbal angles [deg]      : {np.array2string(np.degrees(d), precision=4)}")
    print(f"  commanded torque [N*m]   : {np.array2string(result.commanded_torque, precision=6)}")
    print(f"  achieved torque [N*m]    : {np.array2string(result.achieved_torque, precision=6)}")
    print(f"  torque error norm [N*m]  : {result.torque_error_norm:.6e}")
    print(f"  gimbal rates [rad/s]     : {np.array2string(result.gimbal_rates, precision=6)}")
    print(f"  singularity measure m    : {result.measure:.9g}")
    print(f"  sigma_min                : {result.min_singular_value:.6e}")
    print(f"  lam used                 : {result.lam:.6e}")
    print(f"  rate limited             : {result.rate_limited} ({result.n_rate_limited} gimbals)")
    if result.torque_error_norm > args.tolerance:
        print(
            f"  note                     : torque error {result.torque_error_norm:.6e} N*m "
            f"exceeds the {args.tolerance:g} N*m tolerance"
        )
        return 1
    return 0


def _cmd_manoeuvre(args: argparse.Namespace) -> int:
    array = _build_array(args)
    suite = manoeuvre_suite(array, args.index + 1, seed=args.seed)
    profile = suite.profiles[args.index]
    start = suite.initial_deltas[args.index]
    policy = (
        NoNullMotion()
        if args.null == "none"
        else GradientNullMotion(gain=args.gain, max_rate=0.5)
    )
    history = run_steering(
        array,
        start,
        profile,
        method=args.method,
        null_policy=policy,
        max_gimbal_rate=args.max_gimbal_rate,
    )
    print(f"config={args.config} method={args.method} null={policy.name} seed={args.seed}")
    print(f"  steps / duration [s]         : {profile.n_steps} / {profile.duration:.2f}")
    print(f"  peak commanded momentum      : {profile.peak_momentum:.6f} N*m*s")
    print(f"  max instantaneous torque err : {history.max_torque_error:.6e} N*m")
    print(f"  rms instantaneous torque err : {history.rms_torque_error:.6e} N*m")
    print(f"  accumulated momentum error   : {history.accumulated_momentum_error:.6e} N*m*s")
    print(f"  path momentum error          : {history.total_momentum_error_path:.6e} N*m*s")
    print(f"  minimum singularity measure  : {history.min_measure:.6e}")
    print(f"  peak gimbal rate [rad/s]     : {history.peak_gimbal_rate:.6f}")
    print(f"  rate-limited steps           : {history.n_rate_limited} of {profile.n_steps}")
    if history.min_measure < args.measure_floor:
        print(
            f"  note                         : singularity measure fell to "
            f"{history.min_measure:.6e}, below the {args.measure_floor:g} floor"
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.  Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "array": _cmd_array,
        "singularity": _cmd_singularity,
        "steer": _cmd_steer,
        "manoeuvre": _cmd_manoeuvre,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, TypeError, IndexError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
