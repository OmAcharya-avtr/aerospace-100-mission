"""Command-line interface: ``python -m fdiscope <command>``.

Four commands, each a thin wrapper over the library:

``design``     thresholds and expected delays for a given false-alarm target
``simulate``   inject one fault and print what the residual and detectors did
``signatures`` the fault-signature Gram matrix of the GLR bank
``benchmark``  a small end-to-end detection benchmark

``benchmark`` runs a reduced campaign so that it finishes in well under a
minute; the numbers quoted in the README and ``validation/VALIDATION.md`` come
from the full validation scripts, not from this command, and the command says
so on every run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .analytic import (
    chi2_threshold,
    cusum_arl0_siegmund,
    cusum_delay_siegmund,
    cusum_delay_wald,
    cusum_threshold_for_arl0,
    normalised_bias_signature,
)
from .classifier import FaultClassifier
from .detectors import ChiSquaredDetector, detection_delay
from .evaluate import (
    BenchmarkConfig,
    build_cusum_bank,
    build_default_bank,
    calibrate_all_thresholds,
    default_scenario_sets,
    evaluate_detection,
    harvest_training_rows,
    healthy_calibration_runs,
    method_names,
    run_scenarios,
)
from .faults import FaultSpec, FaultType
from .metrics import mean_ci
from .plant import PlantConfig, loop_matrices
from .simulate import LoopConfig, build_filter, simulate_loop

_FAULT_CHOICES = tuple(f.value for f in FaultType)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fdiscope",
        description=(
            "Residual-based fault detection and isolation for a simulated GNC loop. "
            "Research-grade; not flight-qualified and not certified."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    design = sub.add_parser("design", help="threshold design and expected delays")
    design.add_argument("--alpha", type=float, default=1e-3, help="chi-squared level per window")
    design.add_argument("--window", type=int, default=25, help="window length [samples]")
    design.add_argument("--arl0", type=float, default=2000.0, help="CUSUM mean time to false alarm")
    design.add_argument(
        "--bias-sigma", type=float, default=4.0, help="gyro bias in measurement sigmas"
    )

    sim = sub.add_parser("simulate", help="inject one fault and report detector behaviour")
    sim.add_argument("--fault", choices=_FAULT_CHOICES, default="sensor_bias")
    sim.add_argument("--magnitude", type=float, default=4.0, help="see --units")
    sim.add_argument(
        "--units",
        choices=("sigma", "physical"),
        default="sigma",
        help="interpret --magnitude in measurement sigmas or in SI units",
    )
    sim.add_argument("--channel", type=int, choices=(0, 1), default=1)
    sim.add_argument("--onset", type=int, default=600, help="onset sample")
    sim.add_argument("--steps", type=int, default=1600)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--alpha", type=float, default=1e-3)
    sim.add_argument("--window", type=int, default=25)

    sig = sub.add_parser("signatures", help="fault-signature Gram matrix")
    sig.add_argument("--window", type=int, default=100)
    sig.add_argument("--onsets", type=int, default=8, help="onset phases to average over")

    bench = sub.add_parser("benchmark", help="reduced end-to-end detection benchmark")
    bench.add_argument("--train", type=int, default=80)
    bench.add_argument("--test", type=int, default=80)
    bench.add_argument("--calib", type=int, default=60)
    bench.add_argument("--trees", type=int, default=100)
    return parser


def _cmd_design(args: argparse.Namespace) -> int:
    plant = PlantConfig()
    kf = build_filter(loop_matrices(plant))
    dof = int(args.window) * 2
    threshold = chi2_threshold(args.alpha, dof)
    sigma_rate = float(np.sqrt(plant.gyro_var_rad2_s2))
    direction, mu = normalised_bias_signature(kf, [0.0, args.bias_sigma * sigma_rate])
    h = cusum_threshold_for_arl0(args.arl0, mu)
    print("chi-squared test")
    print(f"  window                    = {args.window} samples ({dof} degrees of freedom)")
    print(f"  design false-alarm rate   = {args.alpha:g} per window")
    print(f"  threshold                 = {threshold:.6f}")
    print("CUSUM, gyro bias hypothesis")
    print(f"  bias                      = {args.bias_sigma:g} sigma "
          f"({args.bias_sigma * sigma_rate:.6e} rad/s)")
    print(f"  steady-state residual mu  = {mu:.6f}")
    print(f"  residual direction        = [{direction[0]:+.6f} {direction[1]:+.6f}]")
    print(f"  target mean time to false alarm = {args.arl0:g} samples")
    print(f"  threshold                 = {h:.6f}")
    print(f"  achieved analytic ARL0    = {cusum_arl0_siegmund(h, mu):.2f} samples")
    print(f"  Wald mean delay h/K       = {cusum_delay_wald(h, mu):.3f} samples")
    print(f"  Siegmund mean delay       = {cusum_delay_siegmund(h, mu):.3f} samples")
    print("Note: the Siegmund expression assumes the residual mean steps to mu at")
    print("onset.  In the closed loop it rises over the estimator time constant, so")
    print("the realised delay is longer; see validation/cusum_delay.py.")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    plant = PlantConfig()
    kind = FaultType(args.fault)
    sigma = float(
        np.sqrt(plant.attitude_var_rad2 if args.channel == 0 else plant.gyro_var_rad2_s2)
    )
    magnitude = float(args.magnitude)
    if args.units == "sigma" and kind in (FaultType.SENSOR_BIAS, FaultType.SENSOR_DRIFT):
        magnitude *= sigma
    if kind is FaultType.NONE:
        spec = FaultSpec()
    else:
        spec = FaultSpec(
            kind=kind, onset_step=int(args.onset), magnitude=magnitude, channel=int(args.channel)
        )
    cfg = LoopConfig(n_steps=int(args.steps), seed=int(args.seed))
    run = simulate_loop(cfg, spec)
    onset = int(args.onset)

    det = ChiSquaredDetector(window=int(args.window), dim=2, alpha=float(args.alpha))
    chi = det.run(run.residual)
    cusum = build_cusum_bank(1.0, cusum_threshold_for_arl0(2000.0, 1.0))
    cus = cusum.run(run.residual)

    print(f"fault                     = {kind.value}")
    if kind is not FaultType.NONE:
        print(f"magnitude                 = {magnitude:.6e} (channel {args.channel})")
        print(f"onset sample              = {onset}")
    print(f"steps                     = {args.steps} at dt = {plant.dt_s} s, seed {args.seed}")
    pre = run.nis[:onset] if onset > 0 else run.nis
    post = run.nis[onset:]
    print(f"mean NIS before onset     = {np.mean(pre):.4f}   (expect 2.0)")
    print(f"mean NIS after onset      = {np.mean(post):.4f}")
    print(f"chi2 threshold            = {chi.threshold:.4f} "
          f"(window {args.window}, alpha {args.alpha:g})")
    print(f"CUSUM threshold           = {cus.threshold:.4f} (ARL0 2000, mu 1.0)")
    if kind is FaultType.NONE:
        print(f"chi2 alarm fraction       = {chi.alarm_fraction:.6f}")
        print(f"CUSUM alarm fraction      = {cus.alarm_fraction:.6f}")
    else:
        d_chi = detection_delay(chi.alarm, onset)
        d_cus = detection_delay(cus.alarm, onset)
        print(f"chi2 detection delay      = {d_chi:.0f} samples"
              if np.isfinite(d_chi) else "chi2 detection delay      = never detected")
        print(f"CUSUM detection delay     = {d_cus:.0f} samples"
              if np.isfinite(d_cus) else "CUSUM detection delay     = never detected")
        print(f"false alarms before onset = chi2 {int(np.count_nonzero(chi.alarm[:onset]))}, "
              f"CUSUM {int(np.count_nonzero(cus.alarm[:onset]))}")
    return 0


def _cmd_signatures(args: argparse.Namespace) -> int:
    cfg = BenchmarkConfig(iso_window=int(args.window))
    bank = build_default_bank(cfg, n_onsets=int(args.onsets))
    names = [f.value for f in bank.faults]
    gram = bank.gram()
    width = 14
    print(f"signature window {args.window} samples, averaged over {args.onsets} onset phases")
    print(" " * 32 + "".join(f"{n[:width - 2]:>{width}}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>30}  " + "".join(f"{gram[i, j]:>{width}.4f}" for j in range(len(names))))
    off = np.abs(gram - np.eye(len(names)))
    i, j = np.unravel_index(int(np.argmax(off)), off.shape)
    print(f"\nworst off-diagonal |cosine| = {abs(gram[i, j]):.4f} "
          f"between {names[i]} and {names[j]}")
    print("A matched-filter bank squares the projection, so a cosine near -1 is as")
    print("bad as one near +1: those hypotheses cannot be separated at any sample size.")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = BenchmarkConfig()
    bank = build_default_bank(cfg)
    train, test = default_scenario_sets(int(args.train), int(args.test))
    train_runs = run_scenarios(train, cfg)
    test_runs = run_scenarios(test, cfg)
    x, y = harvest_training_rows(train, train_runs, cfg)
    clf = FaultClassifier(n_estimators=int(args.trees), random_state=0).fit(x, y)
    _, calib = healthy_calibration_runs(int(args.calib), 9000, cfg)
    thresholds = calibrate_all_thresholds(calib, cfg, bank, clf, 0.10)
    results = evaluate_detection(test, test_runs, cfg, bank, clf, thresholds)
    print(f"reduced benchmark: {args.train} training / {args.test} held-out scenarios, "
          f"{args.calib} calibration runs, {args.trees} trees")
    print("These are NOT the published numbers; run validation/detection_benchmark.py")
    print("for the full campaign the README and VALIDATION.md quote.")
    print(f"{'method':>12} {'threshold':>11} {'FAR/run':>9} {'det rate':>9} "
          f"{'mean delay':>11} {'95 % CI':>20}")
    for name in method_names():
        m = results[name]
        far = m.far_runs[0] / m.far_runs[1] if m.far_runs[1] else float("nan")
        iv = mean_ci(m.delays) if m.delays.size >= 2 else None
        ci = f"[{iv.low:.2f}, {iv.high:.2f}]" if iv else "-"
        mean = f"{iv.point:.2f}" if iv else "-"
        print(f"{name:>12} {m.threshold:>11.4f} {far:>9.4f} {m.detection_rate:>9.4f} "
              f"{mean:>11} {ci:>20}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "design": _cmd_design,
        "simulate": _cmd_simulate,
        "signatures": _cmd_signatures,
        "benchmark": _cmd_benchmark,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
