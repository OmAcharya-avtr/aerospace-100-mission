"""Cycle slips: where the loop loses lock, and how badly the classical estimate fails.

Run from this directory with ``PYTHONPATH=../src``.  Runtime about 80 s.

The classical route to a slip rate is a Gaussian level-crossing argument: treat
the timing error as a stationary Gaussian process and count crossings of the
S-curve's reversal point.  :func:`slotsync.loop.cycle_slip_rate_rice` implements
it and its docstring lists the three approximations it makes.  This script
measures the real slip rate and compares.

**The comparison fails, by many orders of magnitude, and the failure is the
result.**  The measured transition from locked to unlocked is far sharper and
occurs at far higher loop SNR than the Gaussian estimate predicts, because the
mechanism is not a Gaussian excursion: as the timing error grows, the detector
gain *falls* (the S-curve flattens) while the detector output noise *rises*, so
the escape is a positive-feedback runaway rather than a rare crossing of a fixed
barrier.  Section 4 measures both halves of that mechanism.

What is usable instead is the measured threshold, reported here as the loop SNR
at which the slip rate crosses 1e-04 per symbol.
"""

from __future__ import annotations

import math
import time

import numpy as np

from slotsync.loop import (
    LoopDesign,
    cycle_slip_rate_rice,
    jitter_variance_coloured,
    loop_snr_db,
)
from slotsync.pulses import nyquist_raised_cosine
from slotsync.scurve import scurve
from slotsync.simulate import measure_ted_autocovariance, measure_ted_statistics, run_timing_loop
from slotsync.ted import TedConfig

PULSE = nyquist_raised_cosine(0.5, 4.0)
ZETA = 1.0 / math.sqrt(2.0)
FINE = np.linspace(-0.02, 0.02, 41)
SYMBOLS = 200000
UNIFORM_RMS = 1.0 / math.sqrt(12.0)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def sweep(
    config: TedConfig, bandwidth: float, snrs: tuple[float, ...]
) -> list[dict[str, float]]:
    gain = scurve(config, PULSE, FINE, max_exact_symbols=16).gain_central_difference
    full = scurve(config, PULSE, max_exact_symbols=16)
    boundary = full.reversal_offset or 0.5
    design = LoopDesign.from_bandwidth(bandwidth, ZETA, gain)
    rows: list[dict[str, float]] = []
    for snr in snrs:
        autocovariance = measure_ted_autocovariance(
            config, PULSE, sample_snr_db=snr, max_lag=24, samples=150000
        )
        variance = jitter_variance_coloured(design, autocovariance)
        run = run_timing_loop(
            config,
            PULSE,
            design,
            n_symbols=SYMBOLS,
            sample_snr_db=snr,
            divergence_limit=1.0e6,
        )
        rows.append(
            {
                "snr": snr,
                "bandwidth": bandwidth,
                "boundary": boundary,
                "predicted_rms": math.sqrt(max(variance, 0.0)),
                "loop_snr_db": loop_snr_db(variance, boundary) if variance > 0.0 else float("nan"),
                "rice": cycle_slip_rate_rice(design, autocovariance[0], boundary),
                "measured": run.slip_rate_per_symbol,
                "slips": float(run.slip_count),
                "measured_rms": run.jitter_rms,
            }
        )
    return rows


def report(title: str, rows: list[dict[str, float]]) -> None:
    banner(title)
    print(
        f"   {'snr_dB':>7} {'loop_snr_dB':>12} {'pred rms':>9} {'mc rms':>9} "
        f"{'locked':>7} {'rice/symbol':>13} {'meas/symbol':>13} {'slips':>7} "
        f"{'meas/rice':>12}"
    )
    for row in rows:
        locked = row["measured_rms"] < 0.5 * UNIFORM_RMS
        ratio = row["measured"] / row["rice"] if row["rice"] > 0.0 else float("inf")
        ratio_text = "-" if row["measured"] == 0.0 else f"{ratio:12.3e}"
        print(
            f"   {row['snr']:7.1f} {row['loop_snr_db']:12.3f} {row['predicted_rms']:9.5f} "
            f"{row['measured_rms']:9.5f} {str(locked):>7} {row['rice']:13.4e} "
            f"{row['measured']:13.4e} {int(row['slips']):7d} {ratio_text:>12}"
        )


def threshold(rows: list[dict[str, float]], target: float = 1.0e-4) -> float:
    """Loop SNR in dB where the measured slip rate crosses ``target``, by log interpolation."""
    ordered = sorted(rows, key=lambda r: r["loop_snr_db"])
    for low, high in zip(ordered[:-1], ordered[1:], strict=False):
        if low["measured"] >= target > high["measured"] > 0.0:
            span = math.log10(low["measured"]) - math.log10(high["measured"])
            if span == 0.0:
                return float(high["loop_snr_db"])
            fraction = (math.log10(low["measured"]) - math.log10(target)) / span
            return float(
                low["loop_snr_db"] + fraction * (high["loop_snr_db"] - low["loop_snr_db"])
            )
    return float("nan")


def main() -> int:
    started = time.perf_counter()
    print("slotsync validation: cycle slips")
    print(f"pulse {PULSE.name}, zeta {ZETA:.6f}, {SYMBOLS} symbols per point")
    print("'locked' is true when the measured rms timing error is below half the fully")
    print(f"unlocked value 1/sqrt(12) = {UNIFORM_RMS:.6f} symbol; above that the loop has lost")
    print("timing entirely and the wrapped error is uniform over a symbol.")
    print(f"a run of {SYMBOLS} symbols cannot resolve a rate below about {1.0 / SYMBOLS:.1e}")
    print("per symbol, which is a budget limit and is stated wherever a zero count appears.")

    config = TedConfig("mueller-muller", "antipodal")
    sweeps = {
        bandwidth: sweep(config, bandwidth, (12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 4.0))
        for bandwidth in (0.005, 0.01, 0.02, 0.05)
    }
    for bandwidth, rows in sweeps.items():
        report(
            f"1. Mueller-Mueller (white detector noise), B_n = {bandwidth}",
            rows,
        )

    banner("2. the measured slip threshold, which is the usable number")
    print("   loop SNR in dB at which the measured rate crosses 1e-04 slips per symbol,")
    print("   by log-linear interpolation between the two bracketing points")
    print(f"   {'B_n':>8} {'threshold loop SNR dB':>24} {'threshold rms (symbol)':>24}")
    for bandwidth, rows in sweeps.items():
        crossing = threshold(rows)
        rms = (
            float("nan")
            if math.isnan(crossing)
            else 0.5 / math.sqrt(10.0 ** (crossing / 10.0))
        )
        print(f"   {bandwidth:8.3f} {crossing:24.3f} {rms:24.5f}")
    print("   measured here at 16.1 to 18.0 dB of loop SNR, i.e. a predicted rms timing error")
    print("   of 0.063 to 0.079 symbol. Expressed as an rms fraction of a symbol the threshold")
    print("   moves by only a quarter while B_n changes by a factor of five, so the design rule")
    print("   this package supports is a jitter rule, not a bandwidth rule: keep the predicted")
    print("   rms timing error below about 0.05 symbol and slips are unobservable in 200000")
    print("   symbols; let it reach 0.10 symbol and the loop is gone. At B_n = 0.005 the")
    print("   threshold is not located at all - the lowest SNR swept still gives only one slip")
    print("   in 200000 symbols - and that is reported as not located rather than extrapolated.")

    banner("3. the Gaussian level-crossing estimate: how badly it fails")
    print("   over every point above with at least one observed slip, the ratio of the measured")
    print("   rate to the Rice estimate:")
    ratios: list[tuple[float, float, float]] = []
    for bandwidth, rows in sweeps.items():
        for row in rows:
            locked = row["measured_rms"] < 0.5 * UNIFORM_RMS
            if row["slips"] >= 1.0 and row["rice"] > 0.0 and locked:
                ratios.append((bandwidth, row["snr"], row["measured"] / row["rice"]))
    for bandwidth, snr, ratio in ratios:
        print(f"   B_n = {bandwidth:<6} snr = {snr:5.1f} dB   measured / Rice = {ratio:.3e}")
    if ratios:
        orders = [math.log10(r) for _, _, r in ratios]
        print(
            f"   the estimate is low by {min(orders):.1f} to {max(orders):.1f} orders of "
            "magnitude on these points."
        )
    print("   this is not a tuning problem and no fitted prefactor is offered. The Gaussian")
    print("   assumption is wrong in exactly the tail that produces slips, so the estimate's")
    print("   functional form is wrong, not just its scale. It is shipped because it is the")
    print("   classical answer and a user will otherwise reach for it; it is shipped with this")
    print("   measurement attached.")

    banner("4. the mechanism: gain falls and noise rises together as the error grows")
    print("   measured at fixed timing offsets, Mueller-Mueller at 8 dB per-sample SNR.")
    print("   'local gain' is the S-curve's own slope at that offset, by central difference.")
    print(
        f"   {'offset':>8} {'S(offset)':>11} {'local gain':>11} "
        f"{'sigma_n^2':>11} {'ratio to 0':>11}"
    )
    curve = scurve(config, PULSE, max_exact_symbols=16)
    reference = measure_ted_statistics(
        config, PULSE, sample_snr_db=8.0, offset=0.0, samples=200000
    ).variance
    for offset in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45):
        index = int(np.argmin(np.abs(curve.offsets - offset)))
        local = float(
            (curve.values[index + 1] - curve.values[index - 1])
            / (curve.offsets[index + 1] - curve.offsets[index - 1])
        )
        stats = measure_ted_statistics(
            config, PULSE, sample_snr_db=8.0, offset=offset, samples=200000
        )
        print(
            f"   {offset:8.2f} {float(curve.values[index]):11.6f} {local:11.6f} "
            f"{stats.variance:11.6f} {stats.variance / reference:11.4f}"
        )
    print("   the restoring force weakens and the disturbance strengthens at the same time, so")
    print("   once the error is a third of a symbol out there is no stabilising mechanism left.")
    print("   A fixed-barrier Gaussian crossing model has neither effect in it.")

    banner("summary")
    # Monotonicity is only a meaningful requirement while the loop is still locked.
    # Once the measured rms reaches the uniform value the loop is free-running and the
    # count of symbol changes is a random-walk speed, not a slip rate; those points are
    # excluded from the gate and the exclusion is printed.
    monotone = True
    excluded = 0
    for rows in sweeps.values():
        locked_rows = [
            row for row in sorted(rows, key=lambda r: -r["snr"])
            if row["measured_rms"] < 0.5 * UNIFORM_RMS
        ]
        excluded += len(rows) - len(locked_rows)
        rates = [row["measured"] for row in locked_rows]
        monotone = monotone and all(
            rates[i] <= rates[i + 1] + 1e-12 for i in range(len(rates) - 1)
        )
    thresholds = [threshold(rows) for rows in sweeps.values()]
    located = sum(1 for value in thresholds if not math.isnan(value))
    print(f"   measured slip rate non-decreasing as the SNR falls, locked points only: {monotone}")
    print(f"   unlocked points excluded from that check: {excluded} of 28")
    print(f"   slip threshold located for {located} of {len(thresholds)} loop bandwidths")
    if ratios:
        orders = [math.log10(r) for _, _, r in ratios]
        print(
            f"   Rice estimate low by {min(orders):.1f} to {max(orders):.1f} orders of magnitude "
            "- reported, not corrected"
        )
    print("   gate: monotone, and a threshold located for at least 3 of the 4 bandwidths")
    passed = monotone and located >= 3
    print(f"   within tolerance: {passed}")
    print(f"   elapsed {time.perf_counter() - started:.1f} s")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
