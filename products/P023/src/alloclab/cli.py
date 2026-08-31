"""Command-line interface: ``python -m alloclab``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .allocation import METHODS, allocate
from .ams import attainable_moment_set, expected_vertex_count, zonotope_volume
from .dataset import reference_thruster_cluster
from .effectors import EffectorSet, orthogonal_effectors, pyramid_reaction_wheels
from .failure import reallocate_after_failure

_CONFIGS = ("thrusters", "pyramid", "triad")


def _build(config: str) -> EffectorSet:
    if config == "thrusters":
        return reference_thruster_cluster(max_thrust=1.0, arm=0.5)
    if config == "pyramid":
        return pyramid_reaction_wheels(max_torque=0.1)
    if config == "triad":
        return orthogonal_effectors(1.0)
    raise ValueError(f"unknown config {config!r}; expected one of {_CONFIGS}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m alloclab",
        description="Control allocation: effector configurations, allocation methods, "
        "attainable moment sets and failure reallocation.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", choices=_CONFIGS, default="thrusters", help="built-in effector set"
    )

    sub.add_parser("config", parents=[common], help="print the effectiveness matrix")

    alloc = sub.add_parser("allocate", parents=[common], help="allocate one torque command")
    alloc.add_argument("--torque", nargs=3, type=float, required=True, metavar=("TX", "TY", "TZ"))
    alloc.add_argument("--method", choices=METHODS, default="qp")
    alloc.add_argument(
        "--failed", nargs="*", type=int, default=[], help="indices of failed effectors"
    )

    ams = sub.add_parser("ams", parents=[common], help="attainable moment set summary")
    ams.add_argument("--method", choices=("pairwise", "bruteforce"), default="pairwise")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parser().parse_args(argv)
    eset = _build(args.config)

    if args.command == "config":
        print(eset.summary())
        return 0

    if args.command == "ams":
        ams = attainable_moment_set(eset, method=args.method)
        print(f"config={args.config} method={args.method}")
        rows = [
            ("vertices", str(ams.n_vertices)),
            ("general-position n_v", str(expected_vertex_count(eset))),
            ("hull volume [(N*m)^3]", f"{ams.volume:.12g}"),
            ("closed-form volume", f"{zonotope_volume(eset):.12g}"),
            ("surface area [(N*m)^2]", f"{ams.area:.12g}"),
        ]
        for label, value in rows:
            print(f"  {label:<28}: {value}")
        return 0

    if args.command == "allocate":
        tau = np.asarray(args.torque, dtype=float)
        if args.failed:
            report = reallocate_after_failure(eset, tau, args.failed, method=args.method)
            res = report.degraded
            print(f"config={args.config} method={args.method} failed={list(report.failed)}")
            print(f"  attainable by degraded set : {report.attainable}")
            print(f"  remaining rank             : {report.remaining_rank}")
            if report.volume_ratio is not None:
                print(f"  AMS volume ratio           : {report.volume_ratio:.6f}")
        else:
            res = allocate(eset, tau, method=args.method)
            print(f"config={args.config} method={args.method}")
        width = 28
        rows = [
            ("status", res.status),
            (f"commands [{eset.units}]", np.array2string(res.commands, precision=6)),
            ("achieved torque [N*m]", np.array2string(res.achieved_torque, precision=6)),
            ("residual norm [N*m]", f"{res.residual_norm:.6e}"),
            (f"bound violation [{eset.units}]", f"{res.bound_violation:.6e}"),
        ]
        if res.message:
            rows.append(("note", res.message))
        for label, value in rows:
            print(f"  {label:<{width}}: {value}")
        return 0 if res.feasible else 1

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
