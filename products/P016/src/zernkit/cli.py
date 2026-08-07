"""Command-line interface: ``python -m zernkit``.

Two subcommands, both aimed at the failure mode this package exists to
prevent -- quoting a Zernike coefficient without saying which index convention
it belongs to.

``index``
    Convert between Noll, OSA/ANSI and ``(n, m)``, or print the full
    correspondence table up to a given order.
``noll-table``
    Print computed Noll residual variances ``Delta_J`` alongside the published
    values, with the relative difference.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .indexing import mode_name, nm_to_noll, nm_to_osa, noll_to_nm, osa_to_nm
from .statistics import NOLL_TABLE_IV, residual_variance


def _format_index_row(n: int, m: int) -> str:
    return (
        f"{nm_to_noll(n, m):>6d}  {nm_to_osa(n, m):>6d}  {n:>3d} {m:>+4d}   {mode_name(n, m)}"
    )


def _cmd_index(args: argparse.Namespace) -> int:
    header = f"{'Noll':>6}  {'OSA':>6}  {'n':>3} {'m':>4}   name"
    if args.noll is not None:
        n, m = noll_to_nm(args.noll)
    elif args.osa is not None:
        n, m = osa_to_nm(args.osa)
    elif args.nm is not None:
        n, m = args.nm
    else:
        rows = []
        for n in range(args.max_order + 1):
            for m in range(-n, n + 1, 2):
                rows.append((nm_to_noll(n, m), n, m))
        rows.sort()
        print("Noll <-> OSA/ANSI correspondence (Noll is 1-based, OSA is 0-based)")
        print(header)
        for _, n, m in rows:
            print(_format_index_row(n, m))
        return 0
    print(header)
    print(_format_index_row(n, m))
    return 0


def _cmd_noll_table(args: argparse.Namespace) -> int:
    print(f"Noll residual variance Delta_J in units of (D/r0)^(5/3), D/r0 = {args.d_over_r0}")
    print(f"{'J':>4}  {'computed':>12}  {'Noll 1976':>10}  {'rel. diff':>10}")
    for j in range(1, args.j_max + 1):
        computed = residual_variance(j, d_over_r0=args.d_over_r0)
        published = NOLL_TABLE_IV.get(j)
        if published is None:
            print(f"{j:>4}  {computed:>12.6f}  {'-':>10}  {'-':>10}")
        else:
            scaled = published * args.d_over_r0 ** (5.0 / 3.0)
            rel = (computed - scaled) / scaled
            print(f"{j:>4}  {computed:>12.6f}  {scaled:>10.4f}  {rel:>+9.2%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``zernkit`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="zernkit",
        description="Zernike indexing and Kolmogorov residual-variance utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="convert between Noll, OSA/ANSI and (n, m)")
    group = idx.add_mutually_exclusive_group()
    group.add_argument("--noll", type=int, help="Noll index (1-based)")
    group.add_argument("--osa", type=int, help="OSA/ANSI index (0-based)")
    group.add_argument(
        "--nm", type=int, nargs=2, metavar=("N", "M"), help="radial degree n and azimuthal m"
    )
    idx.add_argument(
        "--max-order", type=int, default=4, help="max radial order for the full table (default 4)"
    )
    idx.set_defaults(func=_cmd_index)

    tab = sub.add_parser("noll-table", help="computed vs published Noll residual variances")
    tab.add_argument("--j-max", type=int, default=21, help="largest J (default 21)")
    tab.add_argument("--d-over-r0", type=float, default=1.0, help="D/r0 (default 1.0)")
    tab.set_defaults(func=_cmd_noll_table)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 2 on invalid input)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2  # pragma: no cover - parser.exit raises
