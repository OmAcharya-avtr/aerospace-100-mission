"""Measure detection power against the injected corpus and plot it.

Baseline 3 in one figure: the measured detection power of the run-and-diff
design at each injected rate, with its exact interval, against the closed form.
The point of the picture is the left end of it -- at the rate this mission
actually shipped, the method everyone uses finds the defect about one time in
twenty-two.

Writes ``screenshots/measured_vs_derived.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from seedaudit.corpus import (  # noqa: E402
    INJECTED_RATES,
    block_detection,
    generate_corpus,
    manifestation_indicators,
)
from seedaudit.power import (  # noqa: E402
    MISSION_RATE,
    bootstrap_interval,
    detection_probability_pairwise,
)
from seedaudit.report import BOOTSTRAP_RESAMPLES, CONFIDENCE  # noqa: E402

#: Independent process runs per rate. Kept modest so the example finishes in
#: about a minute on two cores; the published figures come from
#: validation/validate_injected_corpus.py, which uses far more.
RUNS = 440

#: Audit size. 2 is Baseline 1: run each script twice and diff.
AUDIT_SIZE = 2


def exact_interval(hits: int, runs: int) -> tuple[float, float]:
    """Clopper-Pearson interval, which reaches the boundary the bootstrap cannot."""
    tail = 0.5 * (1.0 - CONFIDENCE)
    low = 0.0 if hits == 0 else float(beta.ppf(tail, hits, runs - hits + 1))
    high = 1.0 if hits == runs else float(beta.ppf(1.0 - tail, hits + 1, runs - hits))
    return low, high


def main() -> int:
    """Measure the corpus and write the figure."""
    rng = np.random.default_rng(_bootstrap.SEED)
    rates: list[float] = []
    measured: list[float] = []
    low_bars: list[float] = []
    high_bars: list[float] = []
    boot_low: list[float] = []
    boot_high: list[float] = []
    derived: list[float] = []

    print(f"runs per rate: {RUNS}, audit size: {AUDIT_SIZE}, seed: {_bootstrap.SEED}")
    print(
        f"{'rate':>10}  {'p_hat':>9}  {'measured':>9}  {'exact lo':>9}  {'exact hi':>9}"
        f"  {'boot lo':>9}  {'boot hi':>9}  {'derived':>9}  {'audits':>7}"
    )
    with tempfile.TemporaryDirectory(prefix="seedaudit-example-") as temporary:
        manifest = generate_corpus(Path(temporary) / "corpus")
        for rate in INJECTED_RATES:
            name = manifest.rate_scripts[rate]
            indicators = manifestation_indicators(
                manifest.root, name, manifest.references[name], RUNS
            )
            _reference, pairwise = block_detection(indicators, AUDIT_SIZE)
            point, blow, bhigh = bootstrap_interval(
                pairwise, rng, BOOTSTRAP_RESAMPLES, CONFIDENCE
            )
            elow, ehigh = exact_interval(int(pairwise.sum()), pairwise.size)
            value = detection_probability_pairwise(rate, AUDIT_SIZE)
            rates.append(rate)
            measured.append(point)
            low_bars.append(max(point - elow, 0.0))
            high_bars.append(max(ehigh - point, 0.0))
            boot_low.append(blow)
            boot_high.append(bhigh)
            derived.append(value)
            print(
                f"{rate:>10.6f}  {indicators.mean():>9.6f}  {point:>9.6f}  {elow:>9.6f}"
                f"  {ehigh:>9.6f}  {blow:>9.6f}  {bhigh:>9.6f}  {value:>9.6f}"
                f"  {pairwise.size:>7d}"
            )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    left = axes[0]
    order = np.argsort(rates)
    x = np.array(rates)[order]
    left.errorbar(
        x,
        np.array(measured)[order],
        yerr=[np.array(low_bars)[order], np.array(high_bars)[order]],
        fmt="o",
        ms=5,
        lw=1.2,
        capsize=3,
        color="#b2182b",
        label=f"measured, exact {int(CONFIDENCE * 100)} per cent interval",
    )
    grid = np.logspace(-3.2, 0.0, 200)
    left.plot(
        grid,
        [detection_probability_pairwise(float(p), AUDIT_SIZE) for p in grid],
        lw=1.5,
        color="#2166ac",
        label=r"derived $1 - p^2 - (1-p)^2$",
    )
    left.axvline(MISSION_RATE, color="0.4", ls=":", lw=1.0)
    left.annotate("1/44", xy=(MISSION_RATE, 1e-3), xytext=(5, 0), textcoords="offset points",
                  color="0.4", fontsize=8)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("injected rate p")
    left.set_ylabel("detection probability, n = 2")
    left.set_title("Baseline 1: run twice and diff")
    left.legend(loc="upper left", fontsize=8)

    right = axes[1]
    labels = [f"{rate:g}" if rate != MISSION_RATE else "1/44" for rate in rates]
    positions = np.arange(len(rates))
    width = 0.38
    right.bar(positions - width / 2, measured, width, color="#b2182b", label="measured")
    right.bar(positions + width / 2, derived, width, color="#2166ac", label="derived")
    right.errorbar(
        positions - width / 2,
        measured,
        yerr=[low_bars, high_bars],
        fmt="none",
        ecolor="0.2",
        lw=1.0,
        capsize=3,
    )
    right.set_xticks(positions)
    right.set_xticklabels(labels)
    right.set_xlabel("injected rate")
    right.set_ylabel("detection probability, n = 2")
    right.set_title("Measured against derived, per rate")
    right.legend(loc="upper right", fontsize=8)
    right.text(
        0.46,
        0.58,
        "at p = 1 every run manifests,\nso the runs agree and\nrun-and-diff sees nothing",
        transform=right.transAxes,
        fontsize=7.5,
        color="0.25",
        va="top",
    )
    right.annotate(
        "",
        xy=(positions[rates.index(1.0)], 0.03),
        xytext=(0.45, 0.50),
        textcoords=right.transAxes,
        arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.9},
    )

    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "measured_vs_derived.png"
    figure.savefig(target)
    plt.close(figure)
    print()
    print(f"figure written to {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
