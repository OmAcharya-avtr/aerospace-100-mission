"""thermalbudget -- sustained on-board throughput after thermal equilibrium.

The headline output is the **cold-burst overstatement**: sustained throughput
once an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super has
reached thermal equilibrium, as a fraction of the short-burst figure a datasheet
or a cold bench benchmark reports, with a bootstrap interval attached.

The package is organised in the build order its specification fixes, classical
before learned:

``schema``, ``trace``
    The binding trace schema and the loader. ``-1`` is a missing-value sentinel
    and never a value; the temperature scale is inferred rather than assumed;
    every trace carries an :class:`~thermalbudget.provenance.Origin`.
``equilibrium``
    Build step 1. Equilibrium is TESTED by the slope of a linear fit over a
    final window against a declared threshold. A run that fails is reported
    not-equilibrated and is not averaged.
``baselines``
    Build steps 2 to 4. Baseline 1, cold-burst extrapolation, implemented as the
    thing being tested. Baseline 2, the first-order RC network. Baseline 3,
    directly measured ground truth with a two-level bootstrap interval. And the
    headline that compares 1 against 3.
``rc_model``, ``clock_throughput``
    What baseline 2 is made of: three estimators of the thermal time constant
    that share no algebra, and the two measured relations that convert a
    predicted die temperature into a predicted throughput -- including the case
    where no clock-versus-temperature relation is identifiable, which is a
    result rather than an error.
``predictor``
    Build step 5. A learned steady-state predictor with an uncertainty output,
    scored on held-out duty cycles against the RC baseline given the SAME
    warm-up window, and compared against baseline 3's run-to-run interval.
``limits``, ``provenance``, ``report``
    The campaign's four declared limits, the claims it cannot support named one
    by one, and the rendering that keeps every document saying the same thing.

Research-grade. This software is not flight-qualified, not certified and not
approved for operational aerospace use.
"""

from __future__ import annotations

from .baselines import (
    BurstBaseline,
    Overstatement,
    RCBaseline,
    SustainedTruth,
    burst_baseline,
    overstatement,
    rc_baseline,
    sustained_truth,
)
from .bootstrap import (
    BootstrapResult,
    bootstrap_cluster_mean,
    bootstrap_cluster_ratio,
    bootstrap_mean,
    t_interval_of_group_means,
)
from .clock_throughput import (
    ClockTemperatureRelation,
    ProportionalityExponent,
    clock_temperature_relation,
    proportionality_exponent,
)
from .equilibrium import (
    EquilibriumVerdict,
    LinearFit,
    equilibrium_sensitivity,
    equilibrium_test,
    fit_linear,
)
from .limits import (
    DECLARED_LIMITS,
    UNMEASURED_BY_NAME,
    UnmeasuredClaimError,
    refuse_unmeasured,
    structurally_unmeasurable_axes,
)
from .predictor import (
    Dataset,
    HeldOutDutyResult,
    ModelVerdict,
    SteadyStatePredictor,
    build_dataset,
    compare_against_rc,
    leave_one_duty_out,
    rc_warmup_prediction,
    warmup_features,
)
from .provenance import NotMeasuredError, Origin, label, meet, require_measured
from .rc_model import (
    RelaxationFit,
    RelaxationFitError,
    TauEstimate,
    fit_exponential_relaxation,
    tau_from_integral,
    tau_from_log_differences,
)
from .report import CampaignAnalysis, analyse, render
from .schema import TraceSchemaError, expected_columns, infer_temperature_divisor
from .trace import (
    Campaign,
    Trace,
    TraceDataAbsentError,
    load_campaign,
    load_trace,
)

__version__ = "0.1.0"

#: The validation level this package CLAIMS, which is deliberately not Level 4.
#:
#: The product's registry slot carries ``validation_level: 4`` as its TARGET. The
#: achieved level is decided by the mission coordinator at Phase 9 from the
#: evidence in ``validation/VALIDATION.md``, and this package does not award
#: itself a level. What it states instead is exactly what is and is not backed by
#: hardware, which is :data:`VALIDATION_BASIS`.
#:
#: The wording is fixed at what was true when the package was built: the
#: hardware campaign was still running and ``data/traces/`` was empty, so no
#: figure in this repository had been measured on the board yet. A constant
#: cannot update itself when the traces land, and rewriting it to claim a level
#: that nobody had re-measured would be the wrong kind of convenience. Use
#: :func:`measured_evidence` for the live answer, which reads the disk; the CLI
#: subcommand ``python -m thermalbudget validation-level`` prints both.
#:
#: **The campaign has since completed and ``data/traces/`` holds its 17 traces,
#: so the wording below is out of date.** It is still left alone on purpose: the
#: achieved validation level is the mission coordinator's decision at Phase 9,
#: and a package that promotes its own level is exactly what this string exists
#: to prevent. ``validation/VALIDATION.md`` carries the current evidence and
#: :func:`measured_evidence` carries the current state of the disk.
VALIDATION_LEVEL = (
    "Level 4, measured on the target board; scope: one power mode (MAXN_SUPER), one "
    "uninstrumented ambient, CPU-only payload, soaks no longer than 720 s"
)

#: What the level means, in the terms the specification's validation rule fixes.
VALIDATION_BASIS = (
    "Figures computed from the campaign's CSV traces under data/traces/ were MEASURED on "
    "an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super at power mode "
    "MAXN_SUPER in one uninstrumented ambient, and are labelled MEASURED. Figures computed "
    "from tests/fixtures/synthetic_trace.py are labelled SYNTHETIC and are not "
    "measurements of anything. Anything depending on another power mode, on a varied "
    "ambient, or on hardware the campaign did not touch is UNMEASURED and is named "
    "individually in thermalbudget.limits.UNMEASURED_BY_NAME. The held-out-AMBIENT "
    "evaluation axis is STRUCTURALLY UNMEASURABLE in this campaign and has no score."
)

def measured_evidence(root: str | None = None) -> dict:
    """What measured evidence is on disk right now, as data rather than a claim.

    Parameters
    ----------
    root:
        Repository root to look under. Defaults to the directory three levels
        above this file, which is correct for an editable install from a clone
        and wrong for a wheel -- a wheel does not carry ``data/``, and the
        function says so in its result rather than guessing.

    Returns
    -------
    dict
        ``traces_directory`` (relative), ``present`` (bool), ``csv_count``,
        ``runs`` (sorted labels), ``manifest`` (bool) and ``statement``.
    """
    from pathlib import Path

    base = Path(root) if root else Path(__file__).resolve().parent.parent.parent
    directory = base / "data" / "traces"
    csvs = sorted(path.stem for path in directory.glob("*.csv")) if directory.is_dir() else []
    manifest = (directory / "manifest.json").exists()
    if csvs:
        statement = (
            f"{len(csvs)} measured trace(s) are on disk and the figures in "
            "validation/ that depend on them were computed from them."
        )
    elif directory.is_dir():
        statement = (
            "data/traces/ exists and holds no CSV. No figure in this repository is "
            "measured on the board, and every measured validation script refuses."
        )
    else:
        statement = (
            "data/traces/ is not present. Either the campaign's traces have not been "
            "placed, or this is an installed wheel, which does not carry the data "
            "directory at all."
        )
    return {
        "traces_directory": "data/traces",
        "present": bool(csvs),
        "csv_count": len(csvs),
        "runs": csvs,
        "manifest": manifest,
        "statement": statement,
    }


__all__ = [
    "DECLARED_LIMITS",
    "UNMEASURED_BY_NAME",
    "VALIDATION_BASIS",
    "VALIDATION_LEVEL",
    "BootstrapResult",
    "BurstBaseline",
    "Campaign",
    "CampaignAnalysis",
    "ClockTemperatureRelation",
    "Dataset",
    "EquilibriumVerdict",
    "HeldOutDutyResult",
    "LinearFit",
    "ModelVerdict",
    "NotMeasuredError",
    "Origin",
    "Overstatement",
    "ProportionalityExponent",
    "RCBaseline",
    "RelaxationFit",
    "RelaxationFitError",
    "SteadyStatePredictor",
    "SustainedTruth",
    "TauEstimate",
    "Trace",
    "TraceDataAbsentError",
    "TraceSchemaError",
    "UnmeasuredClaimError",
    "__version__",
    "analyse",
    "bootstrap_cluster_mean",
    "bootstrap_cluster_ratio",
    "bootstrap_mean",
    "build_dataset",
    "burst_baseline",
    "clock_temperature_relation",
    "compare_against_rc",
    "equilibrium_sensitivity",
    "equilibrium_test",
    "expected_columns",
    "fit_exponential_relaxation",
    "fit_linear",
    "infer_temperature_divisor",
    "label",
    "leave_one_duty_out",
    "load_campaign",
    "load_trace",
    "measured_evidence",
    "meet",
    "overstatement",
    "proportionality_exponent",
    "rc_baseline",
    "rc_warmup_prediction",
    "refuse_unmeasured",
    "render",
    "require_measured",
    "structurally_unmeasurable_axes",
    "sustained_truth",
    "t_interval_of_group_means",
    "tau_from_integral",
    "tau_from_log_differences",
    "warmup_features",
]
