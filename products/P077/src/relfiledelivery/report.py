"""Text rendering for the command line and for committed validation output.

Nothing in this module computes anything. It exists so that the same numbers
are formatted the same way in the command-line output, in the validation
transcripts and in the README, and so that no library module needs a ``print``.
"""

from __future__ import annotations

import numpy as np

from .analytic import AnalyticDelivery
from .link import LinkConditions
from .modes import TransferMode
from .montecarlo import DeliveryDistribution
from .payload import FileTransferSpec
from .simulate import TransferResult

RULE_WIDTH = 78


def rule(title: str) -> list[str]:
    """A titled horizontal rule."""
    return ["", "=" * RULE_WIDTH, title, "=" * RULE_WIDTH]


def scenario_report(spec: FileTransferSpec, link: LinkConditions) -> list[str]:
    """The declared scenario, file first and link second."""
    return [*rule("declared scenario"), *spec.describe(), *link.describe()]


def analytic_report(prediction: AnalyticDelivery) -> list[str]:
    """The closed-form prediction with its assumptions restated."""
    return [
        *rule(f"closed form, {prediction.mode.value}"),
        "assumptions A1-A4: independent per-PDU loss, no outages, continuous",
        "contact, fixed-size control PDUs, back-to-back transmission.",
        *prediction.describe(),
    ]


def simulation_report(result: TransferResult) -> list[str]:
    """One simulated run."""
    return [*rule(f"one simulated run, {result.mode.value}"), *result.describe()]


def distribution_report(distribution: DeliveryDistribution) -> list[str]:
    """One Monte Carlo ensemble, quantiles first and the tail stated apart."""
    return [
        *rule(f"Monte Carlo, {distribution.mode.value}"),
        *distribution.describe(),
    ]


def mode_comparison(distributions: dict[TransferMode, DeliveryDistribution]) -> list[str]:
    """One row per mode: completion, median, q95, tail and overhead.

    No mean. The columns are chosen so that a reader cannot take a mean
    delivery time out of this table, because over an intermittent link it does
    not describe the distribution it came from.
    """
    header = (
        f"{'mode':<16}{'complete':>10}{'q0.50 s':>12}{'q0.95 s':>12}"
        f"{'max s':>12}{'max/med':>10}{'overhead':>10}{'tx':>8}"
    )
    lines = [header, "-" * len(header)]
    for mode, distribution in distributions.items():
        if distribution.completed == 0:
            lines.append(
                f"{mode.value:<16}{distribution.completion_fraction:>10.4f}"
                f"{'n/a':>12}{'n/a':>12}{'n/a':>12}{'n/a':>10}"
                f"{float(distribution.overhead_ratios.mean()):>10.4f}"
                f"{float(distribution.transmissions.mean()):>8.0f}"
            )
            continue
        lines.append(
            f"{mode.value:<16}{distribution.completion_fraction:>10.4f}"
            f"{distribution.quantiles[0.50]:>12.4f}"
            f"{distribution.quantiles[0.95]:>12.4f}"
            f"{distribution.tail['max_s']:>12.4f}"
            f"{distribution.tail['max_over_median']:>10.4f}"
            f"{float(np.median(distribution.overhead_ratios)):>10.4f}"
            f"{float(np.median(distribution.transmissions)):>8.0f}"
        )
    return lines


def closed_form_agreement(
    prediction: AnalyticDelivery,
    distribution: DeliveryDistribution,
    standard_error_s: float,
) -> list[str]:
    """The no-outage check: simulation mean against the exact expectation.

    The tolerance is the sampling standard error of the simulated mean, not a
    number chosen to pass. The reported figure is the discrepancy in units of
    that error, so a reader can see whether the agreement is as good as the
    sample allows.
    """
    if prediction.expected_delivery_time_s is None:
        raise ValueError(
            f"mode {prediction.mode.value} has no closed-form expected delivery time"
        )
    exact = prediction.expected_delivery_time_s
    measured = distribution.mean_s
    difference = measured - exact
    sigmas = abs(difference) / standard_error_s if standard_error_s > 0 else float("inf")
    return [
        *rule(f"simulation against the closed form, {prediction.mode.value}"),
        f"closed-form E[T]        : {exact:.9f} s",
        f"simulated mean          : {measured:.9f} s over {distribution.runs} runs",
        f"standard error          : {standard_error_s:.9f} s",
        f"difference              : {difference:+.9f} s",
        f"difference / std error  : {sigmas:.4f}",
        f"relative difference     : {difference / exact:+.9e}",
    ]
