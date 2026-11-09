"""The profile likelihood: flat where the covariance claims an interval.

What to notice: on the left, the rate-only set. All five parameter profiles lie
on the zero line -- the data contain no information about any of them -- and the
vertical bars beneath show the half-widths the linearised covariance claims for
the same five parameters. The covariance is wrong about all five. On the right,
the same campaign with a reaction-torque sensor added: the profiles are now
curved and the covariance becomes checkable -- except for the drag coefficient,
whose profile is flat there too and whose amber bar says so.

The figure uses the 20 s campaign so that it draws in a few seconds. The 200 s
campaign gives the same verdicts -- five flat profiles and five overrides on the
rate-only set, none on the two-channel set -- and its numbers, which are the
ones quoted in README.md, are in
``validation/validate_profile_output.txt``.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from _common import ANALYTIC, LEARNED, REFUSED, WARN, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.attribution import CONFIDENCE_LEVEL  # noqa: E402
from twindiverge.identifiability import analyse  # noqa: E402
from twindiverge.plant import NOMINAL, PARAMETER_NAMES  # noqa: E402
from twindiverge.profile import fit, profile_parameters  # noqa: E402
from twindiverge.synthetic import generate_cases, observe  # noqa: E402
from twindiverge.telemetry import TELEMETRY_SETS  # noqa: E402

POINTS = 7
SEED = 20000
STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def main() -> int:
    style()
    critical = float(stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
    threshold = float(stats.chi2.ppf(CONFIDENCE_LEVEL, 1))
    case = generate_cases((SEED,), "single")[0]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.4),
                               height_ratios=[2.4, 1.0])
    summary = []
    for column, name in enumerate(("rate_short", "rate_torque_short")):
        telemetry = TELEMETRY_SETS[name]
        report = analyse(telemetry, NOMINAL)
        observed = observe(telemetry, case)
        centre, _ = fit(telemetry, observed)
        results = profile_parameters(
            telemetry,
            observed,
            critical * report.pinv_diagonal_not_a_bound,
            points=POINTS,
            centre=centre,
        )
        upper = axes[0, column]
        lower = axes[1, column]
        for index, result in enumerate(results):
            upper.plot(
                result.grid - result.minimum_at,
                np.maximum(result.delta_chi2, 1e-12),
                linestyle=STYLES[index],
                color=ANALYTIC if name.startswith("rate_short") else LEARNED,
                linewidth=1.4,
                label=PARAMETER_NAMES[index],
            )
            lower.barh(
                PARAMETER_NAMES[index],
                result.wald_half_width,
                color=WARN if result.overrides_covariance else REFUSED,
                edgecolor="black",
                linewidth=0.4,
            )
            summary.append(
                (name, result.label, result.max_delta_chi2, result.wald_half_width,
                 result.overrides_covariance, result.sides_crossed)
            )
        upper.axhline(threshold, color="black", linestyle="--", linewidth=1.0)
        upper.text(
            0.02, threshold * 1.3, f"95 per cent threshold {threshold:.4f}",
            transform=upper.get_yaxis_transform(), fontsize=7,
        )
        upper.set_yscale("log")
        upper.set_ylim(1e-12, 1e6)
        upper.set_xlabel("offset from the profile minimum, log-parameter units")
        upper.set_ylabel("delta chi-square")
        upper.set_title(f"{name}: rank {report.rank} of 5")
        upper.legend(fontsize=7, ncols=5, loc="upper center")
        lower.set_xscale("log")
        lower.set_xlabel("linearised (Wald) half-width the covariance claims")
        flat_here = sum(1 for result in results if result.overrides_covariance)
        lower.set_title(
            f"amber where the profile is flat and this number is wrong "
            f"({flat_here} of 5); grey where it is checkable"
        )
    figure.suptitle(
        "twindiverge: a flat profile overrides whatever the covariance claims",
        fontsize=11,
    )
    path = save(figure, "profile_likelihood.png")
    print(f"written: {path}")
    for name, label, maximum, wald, override, sides in summary:
        print(
            f"  {name:<18s} {label:<10s} max delta chi2 {maximum:.6e} "
            f"Wald half-width {wald:.6e} overrides {override} sides {sides}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
