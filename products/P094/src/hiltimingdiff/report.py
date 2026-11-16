"""Assembly and rendering of the headline, with the scope stated beside it.

One paragraph is written down once here and appears verbatim in ``README.md``,
in ``validation/VALIDATION.md`` and at the head of every validation output that
reports a hardware figure. SPEC §6 requires the unpinned clock to be stated
**with** the headline rather than relegated to a limitations list, and the only
way to make that true of every surface is to have one string and quote it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .baselines import (
    PREREGISTERED_SPREAD,
    PREREGISTERED_TRANSFER_ERROR,
    Baseline1,
    Baseline2,
    Baseline3,
    baseline_1,
    baseline_2,
    baseline_3,
)
from .bootstrap import RESAMPLES, RESAMPLING_UNIT
from .divergence import Divergence, PassUnits, pass_units
from .drift import Confounding, DriftComparison, confounding, drift_comparison
from .frequency import FrequencyCovariation, frequency_covariation
from .gate import GateCensus, gate_census
from .simulate import DEFAULT_PASSES, SimulationResult, calibrate_ns_per_op, measure_simulation
from .sources import SOURCE_MD5
from .traces import Campaign, load_campaign

#: The measurement model, written once. Quoted verbatim wherever a hardware
#: figure is reported.
MEASUREMENT_MODEL = """\
Hardware-mode figures come from the committed campaign in data/traces/, run on
a Jetson Orin Nano: nine grid cells, 60 passes of 1500 frames each, 90000 timed
frames per cell, 810000 frames in all, with the cell order randomised under seed
20261115. Each of the six stages is bracketed individually by
time.perf_counter_ns, whose resolution the campaign measured at 1.000 ns and
whose empty-bracket cost it measured per cell at 107.365 to 115.204 ns. A pass
summary is computed over EVERY timed frame of that pass; data/traces/stages.csv
is subsampled every third frame and any figure taken from it says so. EVERY
HARDWARE FIGURE HERE IS A FIGURE AT AN UNPINNED, SCHEDUTIL-GOVERNED CLOCK IN
MAXN_SUPER: passwordless sudo was unavailable on the board, so neither the
governor nor the power mode could be pinned, per-core frequency spanned 729600
to 1728000 kHz across the campaign, and the share of per-stage variance that
co-varies with core frequency is measured and reported rather than assumed away.
The board also drifted: a fixed 160x160 float64 matmul reference, sharing no code
with the pipeline, measured 141844.5 ns before the first cell and 152517.0 ns
after the last, a ratio of 1.0752408447278534, and every divergence ratio is
reported raw AND drift-adjusted with the raw figure as the headline.
Simulation-mode figures are measured in the analysis container, now, by the same
instrumented driver over the same committed pipeline file, and move between runs
because they are wall-clock readings.
"""

#: The one-line scope statement that accompanies the headline everywhere.
SCOPE_LINE = (
    "Measured at an unpinned schedutil-governed clock in MAXN_SUPER on a Jetson "
    "Orin Nano; the board drifted 7.52 per cent over the campaign."
)

#: Research-grade statement, required in every surface of this product.
RESEARCH_GRADE = (
    "Research-grade. Not flight-qualified, not certified, and not approved for "
    "operational aerospace use."
)


@dataclass(frozen=True)
class Verdict:
    """A pre-registered expectation and how it landed."""

    name: str
    statement: str
    threshold: float
    measured: float
    direction: str
    held: bool

    @property
    def label(self) -> str:
        """``CONFIRMED`` or ``REFUTED``."""
        return "CONFIRMED" if self.held else "REFUTED"

    def line(self) -> str:
        """One line naming the expectation, the threshold and the measurement."""
        return (
            f"{self.name}: {self.label} -- pre-registered {self.direction} "
            f"{self.threshold:g}, measured {self.measured:.6g}"
        )


def spread_verdict(divergence_result: Divergence) -> Verdict:
    """SPEC §7's primary expectation: the divergence spread exceeds 3."""
    measured = divergence_result.spread
    return Verdict(
        name="SPEC §7 primary (divergence spread)",
        statement=(
            "The declared operation-count model will be stage-dependently wrong, "
            "not uniformly wrong, and the divergence spread max(R)/min(R) will "
            "exceed 3."
        ),
        threshold=PREREGISTERED_SPREAD,
        measured=measured,
        direction=">",
        held=bool(measured > PREREGISTERED_SPREAD),
    )


def transfer_verdict(baseline2: Baseline2) -> Verdict:
    """SPEC §7's secondary expectation: held-out transfer error exceeds 0.5."""
    measured = baseline2.declared.mean_relative_error
    return Verdict(
        name="SPEC §7 secondary (baseline 2 held-out transfer error)",
        statement=(
            "Baseline 2's single global scaling factor will transfer poorly to "
            "held-out stages, with a mean relative error above 0.5."
        ),
        threshold=PREREGISTERED_TRANSFER_ERROR,
        measured=measured,
        direction=">",
        held=bool(measured > PREREGISTERED_TRANSFER_ERROR),
    )


@dataclass(frozen=True)
class Headline:
    """Everything the product reports, assembled once."""

    campaign: Campaign
    units: PassUnits
    simulation: SimulationResult
    ns_per_op: float
    divergence: Divergence
    baseline1: Baseline1
    baseline2: Baseline2
    baseline3: Baseline3
    drift: DriftComparison
    confounding: Confounding
    frequency: FrequencyCovariation
    gate: GateCensus
    spread_verdict: Verdict
    transfer_verdict: Verdict
    source_md5: dict[str, str]

    @property
    def n_hardware_frames(self) -> int:
        """Timed frames in the hardware campaign."""
        return self.gate.n_frames_total


def build_headline(
    *,
    passes: int = DEFAULT_PASSES,
    resamples: int = RESAMPLES,
    traces_directory: str | None = None,
    progress: bool = False,
) -> Headline:
    """Load the campaign, measure simulation mode here, and compute everything.

    Parameters
    ----------
    passes:
        Simulation-mode passes per cell.
    resamples:
        Bootstrap resamples for every interval.
    traces_directory:
        Override for the measured campaign directory.
    progress:
        Pass ``True`` to have simulation mode write one line per cell to stderr.
    """
    campaign = load_campaign(traces_directory)
    units = pass_units(campaign)
    simulation = measure_simulation(campaign, passes=passes, progress=progress)
    ns_per_op = calibrate_ns_per_op(simulation)
    comparison = drift_comparison(campaign, units, ns_per_op, resamples=resamples)
    baseline2 = baseline_2(units, simulation, resamples=resamples)
    return Headline(
        campaign=campaign,
        units=units,
        simulation=simulation,
        ns_per_op=ns_per_op,
        divergence=comparison.raw,
        baseline1=baseline_1(units, ns_per_op),
        baseline2=baseline2,
        baseline3=baseline_3(units, resamples=resamples),
        drift=comparison,
        confounding=confounding(campaign, units),
        frequency=frequency_covariation(campaign, units),
        gate=gate_census(campaign),
        spread_verdict=spread_verdict(comparison.raw),
        transfer_verdict=transfer_verdict(baseline2),
        source_md5=dict(SOURCE_MD5),
    )


def headline_lines(headline: Headline) -> list[str]:
    """The headline as a list of report lines, scope stated with it."""
    divergence_result = headline.divergence
    lines = [
        "HEADLINE -- P094 HilTimingDiff",
        SCOPE_LINE,
        "",
        f"1. Per-stage divergence ratio R[s], {RESAMPLING_UNIT}-unit "
        f"{divergence_result.confidence:.0%} bootstrap interval, "
        f"ns_per_op = {headline.ns_per_op:.6f} ns/op:",
    ]
    for stage in divergence_result.stages:
        interval = divergence_result.ratio_interval[stage]
        lines.append(
            f"     {stage:<11s} R = {interval.format(6)}   "
            f"({divergence_result.ns_per_declared_op[stage]:.6f} ns per declared op)"
        )
    lines += [
        "",
        f"2. Divergence spread max(R)/min(R) = "
        f"{divergence_result.spread_interval.format(4)}",
        f"   {headline.spread_verdict.line()}",
        "",
        f"3. Rank disagreement: Kendall tau-b = "
        f"{divergence_result.tau_b_interval.format(6)}",
        f"   permutation (measured position -> declared position) = "
        f"{divergence_result.permutation}",
        f"   declared order (cheapest first): {' < '.join(divergence_result.declared_order)}",
        f"   measured order (cheapest first): {' < '.join(divergence_result.measured_order)}",
        "",
        f"4. argmax(R) = {divergence_result.argmax} "
        f"(R = {divergence_result.ratio[divergence_result.argmax]:.6g}, "
        f"{divergence_result.argmax_share:.4f} of measured frame time); "
        f"argmin(R) = {divergence_result.argmin} "
        f"(R = {divergence_result.ratio[divergence_result.argmin]:.6g}, "
        f"{divergence_result.argmin_share:.4f} of measured frame time)",
        "",
        f"5. Baseline 2 held-out transfer error = "
        f"{headline.baseline2.declared_interval.format(6)} on held-out stages "
        f"{', '.join(headline.baseline2.declared.holdout)}",
        f"   {headline.transfer_verdict.line()}",
        "",
        RESEARCH_GRADE,
    ]
    return lines
