"""Empirical coverage of the Wilson 95% confidence interval.

Runs R independent seeded Monte Carlo estimates per case and counts how
often the (independently verified) analytic BER falls inside the reported
95% CI. A correct implementation should show ~95% for the binary
modulations; for PPM the documented residual narrowing (the CI neglects the
variance of the wrong-bits-per-symbol-error ratio) predicts slightly lower,
~92-94%.

This test was the one that CAUGHT a real defect during development: a
bit-level Wilson interval for PPM gave only ~76% coverage for M=16 because
bit errors within a symbol are correlated. The interval is now built on
symbol errors and scaled (see montecarlo.py docstring).

Writes ci_coverage_output.txt. Runtime ~60 s.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from berbench import analytic_ber, mc_ber  # noqa: E402

OUT = Path(__file__).resolve().parent / "ci_coverage_output.txt"
R = 200  # replicates per case

CASES = [
    ("bpsk", {}, 4.0, 100_000),
    ("ook", {}, 6.0, 100_000),
    ("ppm", {"M": 4}, 4.0, 120_000),
    ("ppm", {"M": 16}, 4.0, 120_000),
    ("ook", {"channel": "lognormal", "sigma_i2": 0.3}, 8.0, 100_000),
]


def main() -> None:
    t0 = time.perf_counter()
    lines = [f"Wilson 95% CI empirical coverage, R={R} replicates/case "
             "(berbench 0.1.0)", "=" * 70]
    for mod, kw, snr, n in CASES:
        ana = analytic_ber(mod, snr, **kw).ber[0]
        hits = sum(
            1 for s in range(R)
            if (lambda mc: mc.ci_low[0] <= ana <= mc.ci_high[0])(
                mc_ber(mod, snr, n=n, seed=50_000 + s, **kw))
        )
        lines.append(f"{mod:>5} {str(kw):<45} snr={snr:g} dB  "
                     f"coverage {hits}/{R} = {hits / R:.3f}")
    lines.append("")
    lines.append(f"nominal: 0.95; binomial 2-sigma band for R={R}: "
                 f"+/-{2 * (0.95 * 0.05 / R) ** 0.5:.3f}")
    lines.append("PPM cases are expected ~0.92-0.94 (documented residual CI "
                 "narrowing; see montecarlo.py).")
    lines.append(f"wall time: {time.perf_counter() - t0:.1f} s")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
