"""Report line builders, so that one quantity is rendered one way everywhere.

Every function here returns a list of strings and prints nothing. Library code
in this product does no I/O: the CLI in :mod:`powerenvelope.__main__`, the
validation scripts and the examples decide where a line goes.
"""

from __future__ import annotations

from collections.abc import Iterable

from .baselines import (
    BetweenRunSpread,
    DutyCycleModel,
    EnergyPerInference,
    FloorPower,
    FloorRegression,
    GroundTruth,
    RailAccounting,
    RunOrderRow,
)
from .decomposition import Decomposition
from .predictor import Score
from .schema import REJECT_REASONS
from .trace import Campaign
from .windows import WindowSet

#: The measurement model in one paragraph, verbatim in every place it appears:
#: ``README.md``, ``validation/VALIDATION.md``, the output file of every
#: validation script, and the docstring of :mod:`powerenvelope.trace`.
MEASUREMENT_MODEL = (
    "MEASUREMENT MODEL. The platform rail is VDD_IN, measured by the on-board "
    "INA3221 as bus voltage (mV) times shunt current (mA); P(t) = V(t)*I(t). "
    "Energy over a window is the TRAPEZOIDAL integral of P(t) over the "
    "sampler's CLOCK_MONOTONIC timestamps. A cycle is 2.0 s: the pipeline runs "
    "back-to-back inferences for duty*2.0 s and sleeps the rest; a run is 38 "
    "cycles. An inference is one pass over one frame, one row of the decisions "
    "CSV. A USEFUL DECISION is an inference that passes the gate -- a candidate "
    "exists, its box does not touch the border, its area is within the declared "
    "bounds, and its score is at least tau -- so the acceptance rate is counted "
    "from the committed CSV and never asserted. ENERGY PER USEFUL DECISION is "
    "the integral of P(t) over the WHOLE run, sleep windows included, divided by "
    "the number of accepted decisions in that run."
)


def campaign_lines(campaign: Campaign) -> list[str]:
    """Identity of a loaded campaign: origin, board, environment, cells."""
    manifest = campaign.manifest
    environment = manifest.get("env", {})
    ina = manifest.get("ina3221", {})
    lines = [
        campaign.provenance.banner(),
        f"synthetic manifest flag  : {bool(manifest.get('synthetic', False))}",
        f"kernel/uname             : {environment.get('uname', 'MEASURED_PENDING')}",
        f"cores (affinity)         : {environment.get('ncpu_affinity', 'MEASURED_PENDING')}",
        f"governor                 : {environment.get('governor', 'MEASURED_PENDING')}",
        "cpuinfo_max_freq         : "
        f"{environment.get('cpuinfo_max_freq_khz', 'MEASURED_PENDING')} kHz",
        f"loadavg at start         : {environment.get('loadavg_at_start', 'MEASURED_PENDING')}",
        f"INA3221 update_interval  : {ina.get('update_interval', 'MEASURED_PENDING')}",
        f"INA3221 samples averaged : {ina.get('samples', 'MEASURED_PENDING')}",
        f"cycle_s                  : {campaign.cycle_s}",
        f"cycles per run           : {manifest.get('cycles_per_run', 'MEASURED_PENDING')}",
        f"sampler period           : {campaign.sample_period_s} s",
        f"gate constants           : grad_thr={manifest.get('grad_thr')} "
        f"area_ref={manifest.get('area_ref')} area_min={manifest.get('area_min')} "
        f"area_max={manifest.get('area_max')}",
        f"tau rule                 : {manifest.get('tau_rule', 'MEASURED_PENDING')}",
        f"campaign complete marker : {campaign.complete}",
        f"cells loaded             : {', '.join(campaign.provenance.cells)}",
    ]
    return lines


def cell_table(campaign: Campaign) -> list[str]:
    """One line per cell: configuration and counted decision outcomes."""
    header = (
        f"  {'cell':<22}{'dim':>5}{'duty':>7}{'tau':>10}{'inf':>7}{'acc':>7}"
        f"{'a':>9}{'mean_mW':>11}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    from .energy import whole_run_energy

    for label, cell in sorted(campaign.cells.items()):
        mean_mw = whole_run_energy(cell.power).mean_power_mw
        if cell.decisions is None:
            lines.append(
                f"  {label:<22}{cell.frame_dim:>5}{cell.duty:>7.2f}{cell.tau:>10.4f}"
                f"{'-':>7}{'-':>7}{'-':>9}{mean_mw:>11.3f}"
            )
            continue
        lines.append(
            f"  {label:<22}{cell.frame_dim:>5}{cell.duty:>7.2f}{cell.tau:>10.4f}"
            f"{cell.decisions.inferences:>7}{cell.decisions.accepted_count:>7}"
            f"{cell.decisions.acceptance_rate:>9.4f}{mean_mw:>11.3f}"
        )
    return lines


def floor_lines(floor: FloorPower) -> list[str]:
    """The measured background floor, named as a background and not an idle."""
    return [
        f"floor cell                : {floor.cell}",
        f"mean power                : {floor.mean_mw:.3f} mW",
        f"sample sd                 : {floor.sd_mw:.3f} mW",
        f"min / max                 : {floor.min_mw:.3f} / {floor.max_mw:.3f} mW",
        f"samples / duration        : {floor.n_samples} / {floor.duration_s:.3f} s",
        f"current quantum as power  : {floor.quantisation_mw:.3f} mW",
        f"foreign CPU busy fraction : {floor.foreign_busy_mean:.6f} (measured, not assumed absent)",
        f"die temperature           : {floor.tj_mean_c:.3f} degC",
        "THIS IS A BACKGROUND FLOOR ON A SHARED MACHINE, not an idle floor. The board",
        "had logged-in users, a desktop browser and a load average above 1 at campaign",
        "time. The contamination is the number above, not a caveat.",
    ]


def baseline1_lines(baseline: EnergyPerInference) -> list[str]:
    """Baseline 1 for one cell and one variant."""
    return [
        f"variant                       : {baseline.variant}",
        f"mean active power (time-wtd)  : {baseline.mean_active_power_mw:.3f} mW",
        f"mean active power (sample)    : {baseline.sample_mean_active_power_mw:.3f} mW",
        f"mean inference latency        : {baseline.mean_latency_s:.9f} s",
        f"ENERGY PER INFERENCE          : {baseline.energy_mj:.9f} mJ",
        f"inferences                    : {baseline.n_inferences}",
        f"active duration               : {baseline.active_duration_s:.6f} s",
        f"interstitial samples / gaps   : {baseline.interstitial.n_samples} / "
        f"{baseline.interstitial.n_gaps} (widest inter-inference gap "
        f"{baseline.interstitial.max_gap_s:.6f} s)",
        f"sleep-window samples in span  : {baseline.n_sleep_samples}",
    ]


def baseline2_lines(model: DutyCycleModel) -> list[str]:
    """Baseline 2 for one cell, components first."""
    return [
        f"cell                          : {model.cell}",
        f"duty / cycle                  : {model.duty:.2f} / {model.cycle_s:.1f} s",
        f"t_active / t_sleep            : {model.t_active_s:.3f} / {model.t_sleep_s:.3f} s",
        f"P_floor (measured)            : {model.p_floor_mw:.3f} mW",
        f"P_active (measured steady)    : {model.p_active_mw:.3f} mW",
        f"E_ramp per wake (measured)    : {model.e_ramp_mj:.6f} mJ over "
        f"{model.n_wakes} wake(s)"
        + ("" if model.ramp_separable else "   [NOT SEPARABLE at this duty; see below]"),
        f"E_ramp charged per cycle      : {model.e_ramp_per_cycle_mj:.6f} mJ "
        "(per WAKE, scaled by wakes/cycles)",
        f"E_sleep / E_active            : {model.e_sleep_mj:.6f} / {model.e_active_mj:.6f} mJ",
        f"E_cycle (three-term)          : {model.e_cycle_mj:.6f} mJ",
        f"E_cycle (two-term identity)   : {model.two_term_e_cycle_mj:.6f} mJ "
        f"(relative residual {model.identity_residual:+.3e})",
        f"inferences per cycle (counted): {model.n_inf_per_cycle:.4f}",
        f"acceptance rate (counted)     : {model.acceptance_rate:.6f}",
        f"ENERGY PER USEFUL DECISION    : {model.energy_per_useful_decision_mj:.6f} mJ",
    ] + (
        []
        if model.ramp_separable
        else [
            "  The active phase at this duty is shorter than three settling intervals, so",
            "  a ramp and a steady level cannot be separated inside it. E_ramp is reported",
            "  as zero and P_active as the whole-window mean, which leaves E_cycle correct",
            "  and only its DECOMPOSITION unavailable. Nothing was estimated.",
        ]
    )


def baseline3_lines(truth: GroundTruth) -> list[str]:
    """Baseline 3 for one cell, with the reject-reason histogram."""
    reasons = ", ".join(f"{key}={value}" for key, value in sorted(truth.reason_counts.items()))
    return [
        f"cell                          : {truth.cell}",
        f"E_total (whole run)           : {truth.energy_mj:.6f} mJ",
        f"duration / mean power         : {truth.duration_s:.3f} s / {truth.mean_power_mw:.3f} mW",
        f"inferences / accepted         : {truth.inferences} / {truth.accepted}",
        f"acceptance rate (counted)     : {truth.acceptance_rate:.6f}",
        f"reject reasons                : {reasons}",
        f"ENERGY PER USEFUL DECISION    : {truth.energy_per_useful_decision_mj:.6f} mJ",
        f"energy per inference, whole   : {truth.energy_per_inference_whole_run_mj:.6f} mJ",
        f"current quantum as power      : {truth.quantisation_mw:.3f} mW "
        "(40 mW at the 5 V nominal)",
    ]


def spread_lines(spread: BetweenRunSpread) -> list[str]:
    """The between-run interval, with the values it came from."""
    values = ", ".join(f"{value:.4f}" for value in spread.values_mj)
    return [
        f"repeats                       : {', '.join(spread.cells)}",
        f"values                        : {values} mJ per useful decision",
        f"mean                          : {spread.mean_mj:.6f} mJ",
        f"sample sd (ddof=1)            : {spread.sd_mj:.6f} mJ",
        f"range                         : {spread.range_mj:.6f} mJ",
        f"t critical ({spread.confidence:.0%}, n={spread.n})    : {spread.t_critical:.4f}",
        f"HALF-WIDTH                    : {spread.half_width_mj:.6f} mJ",
        f"interval                      : [{spread.low_mj:.6f}, {spread.high_mj:.6f}] mJ",
        "A within-run bootstrap over samples would be WRONG here: the INA3221 averages",
        "512 conversions in hardware, so neighbouring samples are not independent, and",
        "the quantity whose uncertainty matters is run-to-run, not sample-to-sample.",
    ]


def reject_census(campaign: Campaign) -> dict[str, int]:
    """Counts per reject reason, summed over every cell that ran the pipeline."""
    census: dict[str, int] = {}
    for cell in campaign.pipeline_cells():
        for reason, count in cell.require_decisions().reason_counts().items():
            census[reason] = census.get(reason, 0) + count
    return census


def census_lines(campaign: Campaign) -> list[str]:
    """The reject-reason census, with the branches that never fired named.

    A reason with zero occurrences is printed as zero rather than omitted. Two
    of the gate's five branches were never reached on this campaign, which
    means the measured acceptance rate is a property of a gate whose empty-
    candidate and border paths were never exercised by measurement. That is a
    weakness of the measurement, not of the gate, and it is said here.
    """
    census = reject_census(campaign)
    total = sum(census.values()) or 1
    lines = [f"  {'reason':<16}{'count':>10}{'share':>10}"]
    lines.append("  " + "-" * 34)
    for reason in REJECT_REASONS:
        count = census.get(reason, 0)
        lines.append(f"  {reason:<16}{count:>10}{count / total:>10.6f}")
    lines.append(f"  {'TOTAL':<16}{sum(census.values()):>10}{1.0:>10.6f}")
    never = [reason for reason in REJECT_REASONS if census.get(reason, 0) == 0]
    if never:
        lines.extend(
            [
                "",
                f"  NEVER FIRED on this campaign: {', '.join(never)}. Those branches of "
                "the gate",
                "  are exercised only by the synthetic generator. The measured acceptance "
                "rate is",
                "  therefore a property of a gate whose "
                + " and ".join(never)
                + " paths were never",
                "  reached by measurement. No occurrence was synthesised to fill the "
                "table.",
            ]
        )
    return lines


def floor_regression_lines(regression: FloorRegression) -> list[str]:
    """The floor's measured dependence on the board's foreign CPU occupancy."""
    return [
        f"seconds in the fit            : {regression.n_seconds}",
        f"occupancy range               : {regression.busy_min:.6f} to {regression.busy_max:.6f}"
        f" (mean {regression.busy_mean:.6f})",
        f"per-second power              : mean {regression.power_mean_mw:.3f} mW, "
        f"sd {regression.power_sd_mw:.3f} mW",
        f"slope                         : {regression.slope_mw_per_unit_busy:.3f} mW per "
        "unit occupancy",
        f"intercept at zero occupancy   : {regression.intercept_mw:.3f} mW  "
        "[EXTRAPOLATED below the observed range]",
        f"Pearson r                     : {regression.pearson_r:.6f}  "
        f"(r^2 = {regression.r_squared:.6f})",
        f"implied foreign contribution  : {regression.foreign_mw:.3f} mW = "
        f"{100.0 * regression.foreign_fraction:.3f} per cent of the measured floor",
        "In the floor cell the pipeline is not running, so every busy cycle counted in it",
        "is foreign work. The correlation is WEAK and that is part of the result: most of",
        "the floor's second-to-second variation is not explained by the occupancy counter,",
        "so the intercept is an extrapolation and is used only as one row of the published",
        "sensitivity table, never as the headline floor.",
    ]


def run_order_lines(rows: list[RunOrderRow]) -> list[str]:
    """The campaign in execution order, so the drift is visible rather than described."""
    lines = [
        f"  {'cell':<22}{'plan':>5}{'elapsed_s':>11}{'busy':>9}{'tj_C':>8}"
        f"{'load1':>8}{'clk_kHz':>10}{'mean_mW':>10}"
    ]
    lines.append("  " + "-" * (len(lines[0]) - 2))
    for row in rows:
        lines.append(
            f"  {row.cell:<22}{row.order:>5}{row.elapsed_s:>11.1f}{row.busy_mean:>9.4f}"
            f"{row.tj_mean_c:>8.2f}{row.load1_mean:>8.3f}{row.clk_mean_khz:>10.0f}"
            f"{row.mean_power_mw:>10.2f}"
        )
    return lines


def rail_lines(rail: RailAccounting) -> list[str]:
    """The measured rail-accounting ratio fit."""
    return [
        f"cells in the fit              : {len(rail.cells)}",
        f"load range                    : {rail.load_min_mw:.3f} to {rail.load_max_mw:.3f} mW",
        f"ratio range                   : {min(rail.ratio):.6f} to {max(rail.ratio):.6f}",
        f"slope                         : {rail.slope_per_mw:.6e} per mW",
        f"intercept (EXTRAPOLATED)      : {rail.intercept:.6f}",
        f"R^2                           : {rail.r_squared:.6f}",
        f"largest absolute residual     : {rail.residual_max:.6f}",
        "This is a RAIL-ACCOUNTING RATIO, not a converter efficiency. It is the share of",
        "the measured input-rail energy that the two instrumented downstream channels",
        "account for. It omits every rail the INA3221 does not instrument and it includes",
        "the regulators' own losses on the paths it does see. It is not a datasheet curve",
        "and nothing here should be read as one.",
    ]


def headline_lines(decomposition: Decomposition) -> list[str]:
    """The headline factor and its four factors."""
    return decomposition.lines()


def window_lines(windows: WindowSet) -> list[str]:
    """The learned predictor's dataset, and what was dropped from it."""
    return [
        f"windows                       : {len(windows)}",
        f"window length                 : {windows.window_s:.1f} s",
        f"descriptors ({len(windows.feature_names)})"
        + " " * max(1, 30 - len(f"descriptors ({len(windows.feature_names)})"))
        + ": "
        + ", ".join(windows.feature_names),
        "target                        : window mean ACTIVE rail power, mW",
        f"target range                  : {windows.targets.min():.3f} to "
        f"{windows.targets.max():.3f} mW",
        f"windows dropped               : {len(windows.dropped)}",
        f"foreign occupancy imputed in  : {windows.foreign_imputed} window(s)"
        f" from the floor cell's {windows.floor_foreign_busy:.6f}",
        "A window whose duty leaves no pipeline-free second has no per-window foreign",
        "occupancy to measure. The imputation is flagged in the design matrix as",
        "env_foreign_imputed and is reported in MODEL_CARD.md as an imputation.",
    ]


def score_table(scores: Iterable[Score]) -> list[str]:
    """A table of predictor scores, one row per predictor per split."""
    header = (
        f"  {'predictor / split':<44}{'n':>5}{'MAE_mW':>11}"
        f"{'RMSE_mW':>11}{'bias_mW':>11}{'cover':>8}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for item in scores:
        coverage = "    n/a" if item.coverage != item.coverage else f"{item.coverage:>8.3f}"
        lines.append(
            f"  {item.label:<44}{item.n_test:>5}{item.mae_mw:>11.4f}"
            f"{item.rmse_mw:>11.4f}{item.bias_mw:>11.4f}{coverage}"
        )
    return lines
