"""Command-line interface: ``python -m wahbakit``.

Two subcommands, both deterministic:

``demo``
    Build a seeded synthetic problem, solve it by all four methods, and print
    the attitude error against the known truth alongside the analytic 1-sigma
    covariance.  Use it to check an install and to see the conventions applied.

``conventions``
    Print the frame, quaternion and covariance conventions this package uses.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .conventions import angle_between_dcm, dcm_from_quat
from .covariance import attitude_covariance, covariance_axis_sigmas_deg
from .observations import DegenerateObservationsError, VectorObservations
from .solve import solve_wahba

CONVENTIONS_TEXT = """\
wahbakit conventions
  observation      pair (b_i, r_i); b_i in the BODY frame, r_i in the REFERENCE
                   frame; argument order is always (body, reference)
  attitude matrix  A with b_i ~= A r_i  (reference-to-body DCM, det A = +1)
  quaternion       scalar first, q = [w, x, y, z], Hamilton product, w >= 0;
                   A = dcm_from_quat(q) matches
                   scipy Rotation.from_quat([x, y, z, w]).as_matrix()
  Shuster's papers write A(q) transposed, with the scalar part last; the
                   Davenport eigenvector is converted inside davenport.py
  sigmas           transverse angular standard deviation per observation, rad
  covariance       E[dtheta dtheta^T] in rad^2, dtheta = log(A_est A_true^T),
                   expressed in the BODY frame
  degeneracy       every solver gates on lambda_min of
                   sum_i w_i (I - b_i b_i^T) >= 1e-6 and raises
                   DegenerateObservationsError below it
"""


def _synthetic(seed: int, sigma: float, n: int) -> tuple[VectorObservations, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_true = rng.normal(size=4)
    dcm_true = dcm_from_quat(q_true)
    reference = rng.normal(size=(n, 3))
    reference /= np.linalg.norm(reference, axis=1)[:, None]
    body = reference @ dcm_true.T + rng.normal(scale=sigma, size=(n, 3))
    sigmas = np.full(n, sigma)
    return VectorObservations(body, reference, sigmas=sigmas), dcm_true


def _demo(args: argparse.Namespace) -> int:
    obs, dcm_true = _synthetic(args.seed, args.sigma, args.n)
    print(f"synthetic problem: n = {obs.n}, sigma = {args.sigma:g} rad, seed = {args.seed}")
    observability = obs.observability()
    print(
        f"observability lambda_min = {observability.lambda_min:.6f} "
        f"({observability.limiting_frame} frame), smallest body separation "
        f"{observability.min_separation_deg:.3f} deg"
    )
    print()
    print(f"{'method':>10}  {'error [deg]':>12}  {'loss':>12}  {'1-sigma rss [deg]':>18}")
    for method in ("triad", "q-method", "quest", "olae"):
        target = obs.subset([0, 1]) if method == "triad" else obs
        solution = solve_wahba(target, method)
        error_deg = np.degrees(angle_between_dcm(solution.dcm, dcm_true))
        covariance = attitude_covariance(target, method)
        rss = float(np.degrees(np.sqrt(np.trace(covariance))))
        print(f"{method:>10}  {error_deg:12.6f}  {solution.loss:12.4e}  {rss:18.6f}")
    print()
    covariance = attitude_covariance(obs, "optimal")
    axes = covariance_axis_sigmas_deg(covariance)
    print(
        "optimal per-axis 1-sigma [deg]: "
        f"x {axes[0]:.6f}, y {axes[1]:.6f}, z {axes[2]:.6f}"
    )
    return 0


def _conventions(_: argparse.Namespace) -> int:
    print(CONVENTIONS_TEXT, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m wahbakit``."""
    parser = argparse.ArgumentParser(
        prog="wahbakit",
        description="Static attitude determination from vector observations.",
    )
    parser.add_argument("--version", action="version", version=f"wahbakit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="solve a seeded synthetic problem by all four methods")
    demo.add_argument("--seed", type=int, default=2026, help="RNG seed (default 2026)")
    demo.add_argument("--sigma", type=float, default=1e-3, help="sensor sigma [rad], default 1e-3")
    demo.add_argument("--n", type=int, default=4, help="number of observations, default 4")
    demo.set_defaults(func=_demo)

    conventions = sub.add_parser("conventions", help="print the conventions used")
    conventions.set_defaults(func=_conventions)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 2 on an input or geometry error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo" and (args.sigma <= 0 or args.n < 2):
        parser.error("--sigma must be positive and --n must be at least 2")
    try:
        return int(args.func(args))
    except (DegenerateObservationsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
