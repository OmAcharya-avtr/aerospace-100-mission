"""Command-line interface: ``python -m wavelab``.

Three subcommands:

``geometry``
    Report the Hudgin/Fried zonal geometry matrix shape and detected null
    space dimension for a given grid size -- the quickest way to see the
    piston-only vs. piston-plus-waffle distinction from the command line.
``reconstruct``
    Generate one synthetic Kolmogorov-screen sample, reconstruct it with the
    regularized modal (Zernike) least-squares baseline, and report the
    coefficient RMS error.
``demo-benchmark``
    A small, fast (few-second) flux sweep comparing the modal least-squares
    baseline against a freshly trained learned ensemble on the same held-out
    data. This is a **reduced-size illustration**, not the mission validation
    run -- the numbers reported in `MODEL_CARD.md` and
    `validation/VALIDATION.md` come from the scripts in `validation/`, with
    larger sample sizes and saved raw output, not from this command.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from .dataset import build_modal_geometry, generate_batch
from .geometry import PupilGrid, fried_matrix, hudgin_matrix, prune_unconstrained
from .linalg import null_space
from .ml import ZernikeSlopeEnsemble
from .modal import ModalReconstructor
from .zernike import fit_zernike, unit_disc_grid
from .screens import kolmogorov_screen


def _cmd_geometry(args: argparse.Namespace) -> int:
    grid = PupilGrid(args.n_grid)
    print(f"PupilGrid(n_grid={args.n_grid}): {grid.n_active} active phase points")
    for name, builder in (("hudgin", hudgin_matrix), ("fried", fried_matrix)):
        full_matrix = builder(grid)
        matrix, keep_idx = prune_unconstrained(full_matrix)
        dim = null_space(matrix, rel_tol=1e-6).shape[1]
        n_dropped = full_matrix.shape[1] - keep_idx.size
        print(
            f"  {name:7s} matrix shape {matrix.shape} ({n_dropped} unconstrained points "
            f"dropped), null space dimension = {dim}"
        )
    return 0


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    noll = list(range(2, args.j_max + 1))
    geometry = build_modal_geometry(noll, args.n_side)
    recon = ModalReconstructor(noll, geometry.sub_x, geometry.sub_y, method="tsvd", reg=1e-6)

    x, y, mask = unit_disc_grid(64)
    screen = kolmogorov_screen(64, args.r0_over_d, seed=args.seed)
    a_true = fit_zernike(noll, x[mask], y[mask], screen[mask])
    s_true = geometry.matrix @ a_true

    rng = np.random.default_rng(args.seed + 1)
    from .noise import add_slope_noise, apply_dropout

    s_noisy = add_slope_noise(s_true, args.flux, rng, sigma_ref=1.0, flux_ref=100.0)
    active = apply_dropout(geometry.n_sub, args.dropout, rng)
    a_hat = recon.reconstruct(s_noisy, active=active)

    rms_true = float(np.sqrt(np.mean(a_true**2)))
    rms_err = float(np.sqrt(np.mean((a_hat - a_true) ** 2)))
    print(f"n_sub={geometry.n_sub}, n_modes={geometry.n_modes}, active={int(active.sum())}")
    print(f"true coefficient RMS   : {rms_true:.6f} rad")
    print(f"reconstruction RMS err : {rms_err:.6f} rad")
    return 0


def _cmd_demo_benchmark(args: argparse.Namespace) -> int:
    noll = list(range(2, args.j_max + 1))
    geometry = build_modal_geometry(noll, args.n_side)
    baseline = ModalReconstructor(noll, geometry.sub_x, geometry.sub_y, method="tikhonov", reg=1e-3)

    train = generate_batch(geometry, args.n_train, photon_flux=1000.0, dropout_rate=0.15, seed=1)
    model = ZernikeSlopeEnsemble(geometry.n_sub, geometry.n_modes, n_estimators=3, max_iter=150)
    model.fit(train.slopes, train.active, train.coeffs)

    print(f"{'flux':>8}  {'baseline RMS':>13}  {'ML RMS':>10}  {'winner':>10}")
    for flux in (100.0, 1000.0, 10000.0):
        test = generate_batch(geometry, args.n_test, photon_flux=flux, dropout_rate=0.15, seed=999)
        base_err = np.array(
            [
                baseline.reconstruct(test.slopes[i], active=test.active[i]) - test.coeffs[i]
                for i in range(len(test))
            ]
        )
        ml_pred = model.predict(test.slopes, test.active)
        ml_err = ml_pred - test.coeffs
        base_rms = float(np.sqrt(np.mean(base_err**2)))
        ml_rms = float(np.sqrt(np.mean(ml_err**2)))
        winner = "baseline" if base_rms < ml_rms else "ML"
        print(f"{flux:8.0f}  {base_rms:13.6f}  {ml_rms:10.6f}  {winner:>10}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``wavelab`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="wavelab",
        description="Slope-to-phase wavefront reconstruction: zonal (Hudgin/Fried) and "
        "modal (Zernike) least squares, plus a learned slopes-to-Zernike ensemble.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    geo = sub.add_parser("geometry", help="report zonal geometry matrix shape and null space")
    geo.add_argument("--n-grid", type=int, default=9, help="phase points per side (default 9)")
    geo.set_defaults(func=_cmd_geometry)

    rec = sub.add_parser("reconstruct", help="reconstruct one synthetic sample")
    rec.add_argument("--n-side", type=int, default=8, help="subapertures per side (default 8)")
    rec.add_argument("--j-max", type=int, default=15, help="max Noll index (default 15)")
    rec.add_argument("--r0-over-d", type=float, default=0.15, help="Fried parameter / pupil diameter")
    rec.add_argument("--flux", type=float, default=1000.0, help="photons per subaperture")
    rec.add_argument("--dropout", type=float, default=0.1, help="subaperture dropout rate")
    rec.add_argument("--seed", type=int, default=0, help="random seed")
    rec.set_defaults(func=_cmd_reconstruct)

    bench = sub.add_parser("demo-benchmark", help="small illustrative baseline-vs-ML flux sweep")
    bench.add_argument("--n-side", type=int, default=8, help="subapertures per side (default 8)")
    bench.add_argument("--j-max", type=int, default=15, help="max Noll index (default 15)")
    bench.add_argument("--n-train", type=int, default=300, help="training samples (default 300)")
    bench.add_argument("--n-test", type=int, default=150, help="test samples per flux (default 150)")
    bench.set_defaults(func=_cmd_demo_benchmark)

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
