"""Command line interface: ``python -m skymatch``.

Four subcommands, all seeded and all printing numbers rather than pictures:

``catalogue``
    Generate a synthetic catalogue and report its statistics against Eq. C1
    and Eq. P1, including the pair-table size that is the real cost of a
    magnitude limit.
``identify``
    Simulate one frame and identify it with the triangle rule and the Pyramid
    rule, printing the correspondence and the attitude error against truth.
``sweep``
    Identification rate and false-identification rate against centroid noise
    or false-star count, with Wilson intervals.
``conventions``
    The frame, quaternion and tolerance conventions, so they can be read
    rather than inferred.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .benchmark import run_trials
from .camera import CameraModel
from .catalogue import expected_close_pairs, generate_catalogue, predicted_count
from .geometry import angle_between_dcm
from .identify import gather_candidates, pyramid_decision, resolve, triangle_decision
from .pairtable import PairTable, expected_pair_count
from .scene import SceneConfig, simulate_scene
from .triangle import separation_tolerance

CONVENTIONS = """skymatch conventions
  direction        dimensionless unit 3-vector; ra/dec in radians, equatorial,
                   r = (cos d cos a, cos d sin a, sin d)                (Eq. G1)
  attitude         DCM A with v_camera = A r_catalogue; boresight is +z in the
                   camera frame, detector axes +x (columns) and +y (rows)
  quaternion       scalar first, q = [w, x, y, z], Hamilton; dcm_from_quat(q)
                   matches scipy Rotation.from_quat([x, y, z, w]).as_matrix()
  projection       ideal pinhole, x = f v_x / v_z, f = (pixels/2)/tan(FOV/2)
                                                                        (Eq. K1-K2)
  tolerance        tau = k sqrt(2) sigma, k = 3 by default, for a per-axis
                   centroid sigma in radians                            (Eq. T2)
  magnitude        smaller is brighter; the catalogue is SYNTHETIC, see
                   DATASET_CARD.md
  scoring          identified / false-identification / no-solution, exhaustive
                   and mutually exclusive; scored on the accepted triangle"""


def _catalogue_command(args: argparse.Namespace) -> int:
    cam = CameraModel(fov_deg=args.fov_deg, pixels=args.pixels)
    cat = generate_catalogue(
        args.mag_limit, seed=args.seed, min_separation_rad=np.radians(args.min_sep_deg)
    )
    table = PairTable(cat, cam.max_separation_rad)
    print(f"synthetic catalogue, magnitude limit {args.mag_limit:.2f}, seed {args.seed}")
    print(f"  stars                        {cat.n_stars}")
    print(f"  Eq. C1 predicted count       {predicted_count(args.mag_limit):.1f}")
    print(f"  close pairs removed (stars)  {cat.removed_close_pairs}")
    if args.min_sep_deg > 0.0:
        n_pred = predicted_count(args.mag_limit)
        expected = expected_close_pairs(int(round(n_pred)), np.radians(args.min_sep_deg))
        print(f"  Eq. C3 expected close pairs  {expected:.2f}")
    print(f"  brightest / faintest         {cat.magnitude[0]:.3f} / {cat.magnitude[-1]:.3f}")
    print(f"  density                      {cat.density_per_steradian:.1f} stars/sr")
    print()
    print(f"camera: {cam.fov_deg:.2f} deg over {cam.pixels} px")
    print(f"  plate scale                  {cam.arcsec_per_pixel:.3f} arcsec/pixel")
    print(f"  field solid angle            {cam.solid_angle_sqdeg:.3f} sq.deg")
    print(f"  expected stars in field      {cat.expected_in_solid_angle(cam.solid_angle_sr):.2f}")
    print(f"  max pair separation          {np.degrees(cam.max_separation_rad):.4f} deg")
    print()
    print("pair table")
    print(f"  pairs                        {table.n_pairs}")
    n_pairs_expected = expected_pair_count(cat.n_stars, cam.max_separation_rad)
    print(f"  Eq. P1 expected pairs        {n_pairs_expected:.0f}")
    print(f"  memory                       {table.nbytes / 1e6:.2f} MB")
    return 0


def _identify_command(args: argparse.Namespace) -> int:
    cam = CameraModel(fov_deg=args.fov_deg, pixels=args.pixels)
    cat = generate_catalogue(args.mag_limit, seed=args.catalogue_seed)
    table = PairTable(cat, cam.max_separation_rad)
    cfg = SceneConfig(
        camera=cam,
        centroid_sigma_arcsec=args.sigma,
        n_false_stars=args.false_stars,
        max_stars=args.max_stars,
    )
    rng = np.random.default_rng(args.seed)
    scene = simulate_scene(cat, cfg, rng)
    tol = separation_tolerance(max(args.sigma, 0.5))
    print(
        f"frame: seed {args.seed}, sigma {args.sigma:g} arcsec, "
        f"{args.false_stars} false star(s), tolerance {np.degrees(tol) * 3600.0:.2f} arcsec"
    )
    print(
        f"  {scene.n_in_field} real stars on the detector, {scene.n_spots} spots handed to "
        f"the matcher ({scene.n_false_stars} false)"
    )
    if scene.n_spots < 4:
        print("  fewer than four spots: no identification is possible")
        return 1
    candidates, diag = gather_candidates(scene.vectors, scene.magnitudes, table, tol, cam)
    print(f"  {len(candidates)} candidate(s) from {int(diag['triples_tried'])} triple(s)")
    print()
    ok = True
    for name, decision in (("triangle", triangle_decision), ("pyramid", pyramid_decision)):
        cand = decision(candidates)
        ident = resolve(cand, scene.vectors, cat, cam, tol, n_candidates=len(candidates))
        if not ident.identified or cand is None:
            print(f"  {name:9s} no solution")
            ok = False
            continue
        correct = cand.is_correct(scene.truth_index)
        err = np.degrees(angle_between_dcm(ident.attitude, scene.attitude)) * 3600.0
        verdict = "CORRECT" if correct else "FALSE IDENTIFICATION"
        print(
            f"  {name:9s} {verdict}: spots {cand.observed} -> stars {cand.catalogue}, "
            f"{cand.n_confirm} confirmation(s), {len(ident.observed_indices)} matched in all"
        )
        print(f"  {'':9s} attitude error {err:.2f} arcsec")
        ok = ok and correct
    return 0 if ok else 1


def _sweep_command(args: argparse.Namespace) -> int:
    cam = CameraModel(fov_deg=args.fov_deg, pixels=args.pixels)
    cat = generate_catalogue(args.mag_limit, seed=args.catalogue_seed)
    table = PairTable(cat, cam.max_separation_rad)
    values = [float(v) for v in args.values]
    print(f"{args.over} sweep, {args.trials} trials per point, magnitude limit {args.mag_limit}")
    header = (
        f"{args.over:>12} {'method':>10} {'ident':>8} {'false ID':>9} "
        f"{'none':>7} {'95% CI on false ID':>22}"
    )
    print(header)
    print("-" * len(header))
    for value in values:
        if args.over == "sigma":
            cfg = SceneConfig(
                camera=cam, centroid_sigma_arcsec=value, n_false_stars=args.false_stars
            )
        else:
            cfg = SceneConfig(
                camera=cam, centroid_sigma_arcsec=args.sigma, n_false_stars=int(value)
            )
        point = run_trials(
            cat, table, cam, cfg, args.trials, args.seed, with_attitude=False
        )
        for name, result in point.methods.items():
            lo, hi = result.false_identification_ci
            print(
                f"{value:12g} {name:>10} {result.identification_rate:8.3f} "
                f"{result.false_identification_rate:9.3f} {result.no_solution_rate:7.3f} "
                f"     [{lo:.4f}, {hi:.4f}]"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m skymatch``."""
    parser = argparse.ArgumentParser(
        prog="skymatch",
        description="Lost-in-space star identification on a synthetic catalogue.",
    )
    parser.add_argument("--version", action="version", version=f"skymatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fov-deg", type=float, default=12.0, help="field of view [deg]")
    common.add_argument("--pixels", type=int, default=1024, help="detector width [pixels]")
    common.add_argument("--mag-limit", type=float, default=6.0, help="limiting magnitude")

    p_cat = sub.add_parser("catalogue", parents=[common], help="generate and describe a catalogue")
    p_cat.add_argument("--seed", type=int, default=20260902)
    p_cat.add_argument("--min-sep-deg", type=float, default=0.02, help="close-pair removal [deg]")
    p_cat.set_defaults(func=_catalogue_command)

    p_id = sub.add_parser("identify", parents=[common], help="identify one simulated frame")
    p_id.add_argument("--seed", type=int, default=1, help="frame seed")
    p_id.add_argument("--catalogue-seed", type=int, default=20260902)
    p_id.add_argument("--sigma", type=float, default=5.0, help="centroid noise [arcsec]")
    p_id.add_argument("--false-stars", type=int, default=0)
    p_id.add_argument("--max-stars", type=int, default=10)
    p_id.set_defaults(func=_identify_command)

    p_sw = sub.add_parser("sweep", parents=[common], help="rates against noise or false stars")
    p_sw.add_argument("--over", choices=("sigma", "false"), default="sigma")
    p_sw.add_argument("--values", nargs="+", default=["1", "5", "20", "40"])
    p_sw.add_argument("--trials", type=int, default=100)
    p_sw.add_argument("--seed", type=int, default=20260902)
    p_sw.add_argument("--catalogue-seed", type=int, default=20260902)
    p_sw.add_argument("--sigma", type=float, default=5.0)
    p_sw.add_argument("--false-stars", type=int, default=0)
    p_sw.set_defaults(func=_sweep_command)

    p_cv = sub.add_parser("conventions", help="print the frame and tolerance conventions")
    p_cv.set_defaults(func=lambda _args: (print(CONVENTIONS), 0)[1])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError) as exc:
        print(f"skymatch: {exc}", file=sys.stderr)
        return 2
