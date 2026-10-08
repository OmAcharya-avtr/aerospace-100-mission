"""Command-line interface: `python -m invariantset`.

This is the only module in the package that writes to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .diagnostics import growth_table, tolerance_sweep
from .invariant import (
    CONVERGED,
    maximal_robust_invariant_set,
    verify_robust_invariance,
)
from .polytope import DEFAULT_REDUNDANCY_TOL, Polytope
from .setalgebra import (
    is_subset,
    minkowski_sum,
    pontryagin_difference,
)
from .systems import get_system, system_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m invariantset",
        description=(
            "Maximal robust invariant sets for discrete-time linear systems "
            "with bounded disturbances. Research-grade; not flight-qualified, "
            "not certified, not approved for operational aerospace use."
        ),
        epilog="This computes the MAXIMAL robust invariant set, not the "
        "minimal robust positively invariant set of Rakovic et al. 2005.",
    )
    parser.add_argument("--version", action="version", version=f"invariantset {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("systems", help="list the built-in example systems")
    p_list.add_argument("--json", action="store_true", help="emit JSON")

    p_inv = sub.add_parser("invariant", help="compute the maximal robust invariant set")
    p_inv.add_argument("--system", required=True, choices=system_names())
    p_inv.add_argument("--max-iter", type=int, default=50, help="iteration cap (default 50)")
    p_inv.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-9,
        help="absolute slack on the invariance test, in b units (default 1e-9)",
    )
    p_inv.add_argument(
        "--redundancy-tol",
        type=float,
        default=DEFAULT_REDUNDANCY_TOL,
        help=f"redundant-row slack, in b units (default {DEFAULT_REDUNDANCY_TOL:g})",
    )
    p_inv.add_argument("--no-geometry", action="store_true", help="skip vertex and volume tracking")
    p_inv.add_argument("--json", action="store_true", help="emit JSON")

    p_sweep = sub.add_parser(
        "tolerance-sweep",
        help="run the recursion once per redundancy tolerance and compare answers",
    )
    p_sweep.add_argument("--system", required=True, choices=system_names())
    p_sweep.add_argument("--max-iter", type=int, default=60)
    p_sweep.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 3e-3, 4e-3, 1e-2],
    )
    p_sweep.add_argument("--json", action="store_true")

    p_alg = sub.add_parser(
        "algebra",
        help="measure the Minkowski/Pontryagin asymmetry on a worked 2-D case",
    )
    p_alg.add_argument("--json", action="store_true")
    return parser


def _systems(args) -> int:
    rows = []
    for name in system_names():
        s = get_system(name)
        rows.append(
            {
                "name": name,
                "dim": s.dim,
                "halfspaces_X": s.X.n_halfspaces,
                "state_units": list(s.state_units),
                "description": s.description,
                "expected": s.expected,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        print(f"{row['name']}  (dim {row['dim']}, {row['halfspaces_X']} rows in X)")
        print(f"    {row['description']}")
        print(f"    expected: {row['expected']}")
    return 0


def _invariant(args) -> int:
    s = get_system(args.system)
    res = maximal_robust_invariant_set(
        s.A,
        s.X,
        s.W,
        max_iter=args.max_iter,
        convergence_tol=args.convergence_tol,
        redundancy_tol=args.redundancy_tol,
        track_geometry=not args.no_geometry,
    )
    invariant = None
    margin = None
    volume = None
    if res.polytope is not None:
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        if res.polytope.is_bounded():
            volume = res.polytope.volume()
    if args.json:
        print(
            json.dumps(
                {
                    "system": s.name,
                    "termination": res.termination,
                    "converged": res.converged,
                    "iterations": res.iterations,
                    "max_iter": res.max_iter,
                    "facets": None if res.polytope is None else res.polytope.n_halfspaces,
                    "volume": volume,
                    "independently_verified_invariant": invariant,
                    "worst_facet_margin": margin,
                    "history": [
                        {
                            "k": g.k,
                            "halfspaces_raw": g.halfspaces_raw,
                            "halfspaces": g.halfspaces,
                            "removed": g.removed,
                            "vertices": g.vertices,
                            "volume": g.volume,
                            "shrink_margin": g.shrink_margin,
                            "margin_ratio": g.margin_ratio,
                        }
                        for g in growth_table(res)
                    ],
                },
                indent=2,
            )
        )
    else:
        print(f"system: {s.name}  ({s.description})")
        print(f"units : {', '.join(s.state_units)}")
        print(res.report())
        if res.polytope is not None:
            print(f"  independent invariance check: invariant={invariant} "
                  f"worst facet margin={margin:.6e}")
            if volume is not None:
                print(f"  volume of returned set     : {volume:.9f}")
        if res.termination != CONVERGED:
            print("  NOTE: this is not S_inf. See the termination line above.")
    return 0 if res.termination == CONVERGED or res.is_empty else 2


def _tolerance_sweep(args) -> int:
    s = get_system(args.system)
    rows = tolerance_sweep(s.A, s.X, s.W, list(args.tolerances), max_iter=args.max_iter)
    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2))
        return 0
    print(f"system: {s.name}  redundancy-tolerance sweep, max_iter={args.max_iter}")
    print("  tol        termination     iters facets volume        inv_margin    vol/ref")
    for r in rows:
        vol = "-" if r.volume is None else f"{r.volume:<13.9f}"
        marg = "-" if r.invariance_margin is None else f"{r.invariance_margin:<13.6e}"
        ratio = "-" if r.volume_vs_reference is None else f"{r.volume_vs_reference:.9f}"
        facets = "-" if r.facets is None else str(r.facets)
        print(
            f"  {r.redundancy_tol:<10.1e} {r.termination:<15s} {r.iterations:<5d} "
            f"{facets:<6s} {vol} {marg} {ratio}"
        )
    distinct = {
        None if r.volume is None else round(r.volume, 9) for r in rows
    }
    if len(distinct) > 1:
        print("  the redundancy tolerance CHANGED the computed answer across this sweep")
    return 0


def _algebra(args) -> int:
    # P is a triangle, Q a small box.  (P (-) Q) (+) Q is a strict subset of P;
    # (P (+) Q) (-) Q equals P.  Areas measured, not asserted.
    P = Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
        np.array([0.0, 0.0, 1.0]),
    )
    Q = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
    eroded = pontryagin_difference(P, Q)
    reopened = minkowski_sum(eroded, Q)
    dilated = minkowski_sum(P, Q)
    closed = pontryagin_difference(dilated, Q)
    sub_open, margin_open = is_subset(reopened, P)
    sub_close, margin_close = is_subset(closed, P)
    sub_close_rev, margin_close_rev = is_subset(P, closed)
    payload = {
        "area_P": P.volume(),
        "area_P_erode_Q": eroded.volume(),
        "area_erode_then_dilate": reopened.volume(),
        "area_deficit": P.volume() - reopened.volume(),
        "area_deficit_fraction": 1.0 - reopened.volume() / P.volume(),
        "erode_then_dilate_subset_of_P": sub_open,
        "erode_then_dilate_margin": margin_open,
        "dilate_then_erode_equals_P": bool(sub_close and sub_close_rev),
        "dilate_then_erode_margins": [margin_close, margin_close_rev],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("P = triangle {x1>=0, x2>=0, x1+x2<=1}, Q = box of half-width 0.1")
    print(f"  area(P)                      = {payload['area_P']:.9f}")
    print(f"  area(P (-) Q)                = {payload['area_P_erode_Q']:.9f}")
    print(f"  area((P (-) Q) (+) Q)        = {payload['area_erode_then_dilate']:.9f}")
    print(f"  area deficit                 = {payload['area_deficit']:.9f} "
          f"({100.0 * payload['area_deficit_fraction']:.4f} % of P)")
    print(f"  (P (-) Q) (+) Q subset P     : {sub_open} "
          f"(worst margin {margin_open:.3e})  <- identity (I3), holds")
    print(f"  (P (-) Q) (+) Q == P         : {not (payload['area_deficit'] > 0)} "
          "  <- the identity people assume, FALSE")
    print(f"  (P (+) Q) (-) Q == P         : {payload['dilate_then_erode_equals_P']} "
          f"(margins {margin_close:.3e}, {margin_close_rev:.3e})  <- identity (I4), holds")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns the process exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "systems": _systems,
        "invariant": _invariant,
        "tolerance-sweep": _tolerance_sweep,
        "algebra": _algebra,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
