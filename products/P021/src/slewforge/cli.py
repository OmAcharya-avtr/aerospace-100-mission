"""Command-line interface: ``python -m slewforge``.

Four subcommands:

``profile``   closed-form timing for a single-axis rest-to-rest slew
``cones``     keep-out cone geometry and the violation verdict for a boresight
``plan``      plan a constrained slew between two attitudes and print the verdict
``dataset``   generate and summarise a batch of planning problems

Every subcommand accepts ``--json`` and writes a single JSON object to stdout.
Invalid input exits 2 with one line on stderr and no traceback; an infeasible
plan exits 1, because "no path exists" is a result, not a usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

import numpy as np

from .dataset import generate_problems, reference_spacecraft
from .keepout import KeepOutCone, KeepOutSet
from .planner import Instrument, SlewProblem, direct_violations, plan
from .profiles import PROFILE_NAMES, make_profile

_JSON_HELP = "print a single JSON object instead of a text report"


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError(f"vector {list(v)} has zero length")
    return a / n


def _quat_from_axis_angle_deg(spec: Sequence[float]) -> np.ndarray:
    axis = _unit(spec[:3])
    half = 0.5 * math.radians(float(spec[3]))
    return np.concatenate([[math.cos(half)], math.sin(half) * axis])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m slewforge",
        description=(
            "Constrained slew planning: profile timing, keep-out cone geometry, "
            "constrained path planning and problem generation."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    prof = sub.add_parser("profile", help="closed-form rest-to-rest slew timing")
    prof.add_argument("--angle", type=float, required=True, help="slew angle [deg]")
    prof.add_argument("--inertia", type=float, default=100.0, help="effective inertia [kg m^2]")
    prof.add_argument("--torque", type=float, default=0.15, help="available torque [N*m]")
    prof.add_argument("--rate-limit", type=float, default=None, help="rate cap [deg/s]")
    prof.add_argument("--kind", choices=PROFILE_NAMES, default="bang_bang")
    prof.add_argument("--json", action="store_true", help=_JSON_HELP)

    cones = sub.add_parser("cones", help="keep-out verdict for one boresight")
    cones.add_argument(
        "--boresight", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    cones.add_argument(
        "--cone",
        nargs=4,
        type=float,
        action="append",
        default=None,
        metavar=("X", "Y", "Z", "HALF_ANGLE_DEG"),
        help="repeatable: cone axis and half-angle in degrees",
    )
    cones.add_argument("--json", action="store_true", help=_JSON_HELP)

    pl = sub.add_parser("plan", help="plan a constrained rest-to-rest slew")
    pl.add_argument(
        "--start",
        nargs=4,
        type=float,
        default=[0.0, 0.0, 1.0, 0.0],
        metavar=("AX", "AY", "AZ", "ANGLE_DEG"),
        help="start attitude as axis and angle in degrees from identity",
    )
    pl.add_argument(
        "--goal",
        nargs=4,
        type=float,
        required=True,
        metavar=("AX", "AY", "AZ", "ANGLE_DEG"),
    )
    pl.add_argument(
        "--boresight", nargs=3, type=float, default=[1.0, 0.0, 0.0], metavar=("X", "Y", "Z")
    )
    pl.add_argument(
        "--cone",
        nargs=4,
        type=float,
        action="append",
        default=None,
        metavar=("X", "Y", "Z", "HALF_ANGLE_DEG"),
    )
    pl.add_argument("--profile", choices=PROFILE_NAMES, default="bang_bang")
    pl.add_argument("--margin", type=float, default=0.0, help="required clearance [deg]")
    pl.add_argument("--max-via", type=int, default=2)
    pl.add_argument("--max-time", type=float, default=None, help="time budget [s]")
    pl.add_argument("--json", action="store_true", help=_JSON_HELP)

    ds = sub.add_parser("dataset", help="generate planning problems and summarise them")
    ds.add_argument("--n", type=int, default=20)
    ds.add_argument("--seed", type=int, default=0)
    ds.add_argument("--json", action="store_true", help=_JSON_HELP)

    return p


def _cones_from_args(spec) -> KeepOutSet:
    if not spec:
        return KeepOutSet()
    return KeepOutSet(
        tuple(
            KeepOutCone(_unit(c[:3]), math.radians(float(c[3])), f"cone{i}")
            for i, c in enumerate(spec)
        )
    )


def _cmd_profile(args) -> tuple[int, dict]:
    angle = math.radians(args.angle)
    if args.inertia <= 0.0:
        raise ValueError(f"--inertia must be > 0 kg m^2, got {args.inertia}")
    if args.torque <= 0.0:
        raise ValueError(f"--torque must be > 0 N*m, got {args.torque}")
    alpha = args.torque / args.inertia
    rate = None if args.rate_limit is None else math.radians(args.rate_limit)
    p = make_profile(args.kind, angle, alpha, args.inertia, rate)
    return 0, {
        "kind": p.kind,
        "angle_deg": args.angle,
        "effective_inertia_kg_m2": args.inertia,
        "available_torque_N_m": args.torque,
        "peak_accel_rad_s2": p.peak_accel,
        "duration_s": p.duration,
        "peak_rate_deg_s": math.degrees(p.peak_rate),
        "coast_time_s": p.coast_time,
        "peak_momentum_N_m_s": p.peak_momentum,
        "momentum_throughput_N_m_s": p.momentum_throughput,
    }


def _cmd_cones(args) -> tuple[int, dict]:
    ko = _cones_from_args(args.cone)
    b = _unit(args.boresight)
    margins = [float(c.margin(b)) for c in ko]
    return 0, {
        "boresight": b.tolist(),
        "cones": [
            {"name": c.name, "half_angle_deg": c.half_angle_deg, "axis": c.axis.tolist()}
            for c in ko
        ],
        "margins_deg": [math.degrees(m) for m in margins],
        "worst_margin_deg": math.degrees(float(ko.margin(b))) if len(ko) else None,
        "violations": list(ko.violations(b)),
        "allowed": bool(ko.is_allowed(b)),
    }


def _cmd_plan(args) -> tuple[int, dict]:
    if args.max_via < 1:
        raise ValueError(f"--max-via must be >= 1, got {args.max_via}")
    body = reference_spacecraft()
    problem = SlewProblem(
        quat_start=_quat_from_axis_angle_deg(args.start),
        quat_goal=_quat_from_axis_angle_deg(args.goal),
        body=body,
        keepout=_cones_from_args(args.cone),
        instruments=(Instrument("telescope", _unit(args.boresight)),),
        profile=args.profile,
        required_margin=math.radians(args.margin),
        max_time=args.max_time,
    )
    result = plan(problem, max_via=args.max_via)
    payload = {
        "feasible": result.feasible,
        "reason": result.reason,
        "detail": result.detail,
        "slew_angle_deg": math.degrees(problem.slew_angle),
        "direct_feasible": result.direct_feasible,
        "direct_min_margin_deg": math.degrees(result.direct_min_margin),
        "direct_violations": [
            {
                "cone": v.cone,
                "boresight": v.boresight,
                "psi_start_deg": math.degrees(v.psi_start),
                "psi_end_deg": math.degrees(v.psi_end),
                "depth_deg": math.degrees(v.depth),
            }
            for v in direct_violations(problem)
        ],
        "n_via": result.n_via,
        "total_time_s": result.total_time,
        "peak_momentum_N_m_s": result.peak_momentum,
        "momentum_throughput_N_m_s": result.momentum_throughput,
        "min_margin_deg": math.degrees(result.min_margin) if result.path else None,
        "solve_time_s": result.solve_time_s,
        "objective_evals": result.n_objective_evals,
        "torque_utilisation": (
            result.actuators.torque_utilisation if result.actuators else None
        ),
        "momentum_utilisation": (
            result.actuators.momentum_utilisation if result.actuators else None
        ),
        "gyroscopic_fraction": (
            result.actuators.gyroscopic_fraction if result.actuators else None
        ),
    }
    return (0 if result.feasible else 1), payload


def _cmd_dataset(args) -> tuple[int, dict]:
    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")
    problems = generate_problems(args.n, args.seed)
    angles = np.array([p.slew_angle for p in problems])
    margins = np.array(
        [
            p.keepout.min_margin_on_arc(
                p.instruments[0].direction(p.quat_start), p.eigenaxis, p.slew_angle
            )
            for p in problems
        ]
    )
    counts = np.array([len(p.keepout) for p in problems])
    return 0, {
        "n_problems": len(problems),
        "seed": args.seed,
        "slew_angle_deg": {
            "min": math.degrees(float(angles.min())),
            "mean": math.degrees(float(angles.mean())),
            "max": math.degrees(float(angles.max())),
        },
        "direct_min_margin_deg": {
            "min": math.degrees(float(margins.min())),
            "mean": math.degrees(float(margins.mean())),
            "max": math.degrees(float(margins.max())),
        },
        "cones_per_problem": {
            "min": int(counts.min()),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
        },
    }


_COMMANDS = {
    "profile": _cmd_profile,
    "cones": _cmd_cones,
    "plan": _cmd_plan,
    "dataset": _cmd_dataset,
}


def _print_text(payload: dict, indent: int = 0) -> None:
    pad = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_text(value, indent + 2)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"{pad}{key}:")
            for item in value:
                _print_text(item, indent + 2)
                print()
        else:
            print(f"{pad}{key:<28} {value}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code: 0 ok, 1 infeasible, 2 bad input."""
    args = _parser().parse_args(argv)
    try:
        code, payload = _COMMANDS[args.command](args)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, default=float))
    else:
        _print_text(payload)
    return code
