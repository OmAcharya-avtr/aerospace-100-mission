"""Seeded scenario generator and the training table built from it.

No flight data is used anywhere in this package. Every number any learned
component here produces is a statement about the generator below, and the
generator's equations are in this docstring so that a reader can decide how far
to trust it.

Sampling, per scenario, from a seeded :class:`numpy.random.Generator`
---------------------------------------------------------------------
``pdu_count``            ``round(10 ** U(1.6, 2.7))``, so 40 to 501 units.
                         Sampled directly rather than derived from a file size,
                         because it is the quantity that sets the cost of a
                         simulated run and bounding it bounds the compute
                         budget.
``pdu_payload_bytes``    uniform over ``{256, 512, 1024}``.
``file_bytes``           ``(pdu_count - 1) * payload + U{1, payload}``, so the
                         final unit is short on all but a measure-zero set of
                         draws. A generator that always produced an exact
                         multiple would never exercise the ragged-tail
                         accounting.
``forward_rate_bps``     ``10 ** U(4.7, 6.3)``, 50 kbit/s to 2 Mbit/s.
``asymmetry``            ``10 ** U(0.7, 2.3)``, 5 to 200; the return rate is
                         the forward rate divided by it.
``one_way_light_time_s`` ``10 ** U(-1.4, 0.6)``, 0.04 s to 4 s, spanning low
                         Earth orbit to beyond the Moon.
``pdu_loss_probability`` ``U(0, 0.25)``.
``mode``                 uniform over deferred-NAK and immediate-NAK.

In the ``correlated_outage`` regime, additionally
-------------------------------------------------
The outage and contact time scales are drawn **relative to the clean-pass time
of the file on this link**, ``C = 8 * W / forward_rate``. Drawing them from
absolute second ranges produced a regime that was not a regime: outages and
contact gaps an order of magnitude shorter or longer than the transfer either
never bite or make every run fail, and the first version of this generator
completed 100 per cent of runs in both regimes with delivery-time
distributions that overlapped. Relative scales are a declared design choice of
the generator, and they are the reason the two regimes differ.

``mean_down_s``          ``C * 10 ** U(-0.6, 0.9)``, 0.25 to 8 clean passes.
``outage_duty``          ``U(0.55, 0.95)``; ``mean_up_s = mean_down_s *
                         duty / (1 - duty)``.
``window_count``         uniform over ``{4, 5, ..., 10}``.
``window_duration_s``    ``C * 10 ** U(-0.2, 0.8)``, 0.63 to 6.3 clean passes.
``contact_duty``         ``U(0.25, 0.90)``; the window period is the duration
                         divided by it.

In the ``no_outage`` regime the outage process is absent and contact is
continuous, which is exactly the regime in which the closed form of
:mod:`relfiledelivery.analytic` is exact. The two regimes exist so that the
learned predictor can be scored separately in the regime where the analytic
model holds and the regime where it does not, which is the comparison the
specification asks for.

Target
------
One simulated transfer per scenario, and its realised delivery time. One run,
not an average: the learned component is asked for a *calibrated interval*, so
its training target has to carry the run-to-run spread as well as the
scenario-to-scenario spread. Scenarios whose run did not complete inside the
declared horizon carry ``completed = False`` and are excluded from the
regression, which means every learned prediction here is conditional on
completion. That is a real limitation and it is recorded in MODEL_CARD.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analytic import clean_pass_seconds, turnaround_seconds
from .link import LinkConditions, OutageModel, uniform_windows
from .modes import TransferMode
from .payload import FileTransferSpec
from .simulate import simulate_transfer

REGIMES: tuple[str, ...] = ("no_outage", "correlated_outage")

PREDICTOR_MODES: tuple[TransferMode, ...] = (
    TransferMode.DEFERRED_NAK,
    TransferMode.IMMEDIATE_NAK,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "log10_pdu_count",
    "log10_pdu_payload_bytes",
    "log10_file_bytes",
    "pdu_loss_probability",
    "log10_forward_rate_bps",
    "log10_asymmetry",
    "log10_one_way_light_time_s",
    "outage_duty",
    "log10_mean_down_s",
    "contact_duty",
    "availability",
    "log10_clean_pass_s",
    "log10_turnaround_s",
    "mode_is_immediate_nak",
)

TARGET_COLUMN = "delivery_time_s"


@dataclass(frozen=True)
class ScenarioRanges:
    """The sampling ranges, as declared data rather than as literals in code."""

    log10_pdu_count: tuple[float, float] = (1.6, 2.7)
    payload_choices: tuple[int, ...] = (256, 512, 1024)
    pdu_header_bytes: int = 12
    log10_forward_rate_bps: tuple[float, float] = (4.7, 6.3)
    log10_asymmetry: tuple[float, float] = (0.7, 2.3)
    log10_light_time_s: tuple[float, float] = (-1.4, 0.6)
    loss_probability: tuple[float, float] = (0.0, 0.25)
    log10_mean_down_per_clean_pass: tuple[float, float] = (-0.6, 0.9)
    outage_duty: tuple[float, float] = (0.55, 0.95)
    window_count_choices: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)
    log10_window_duration_per_clean_pass: tuple[float, float] = (-0.2, 0.8)
    contact_duty: tuple[float, float] = (0.25, 0.90)
    modes: tuple[TransferMode, ...] = field(default=PREDICTOR_MODES)


@dataclass(frozen=True)
class Scenario:
    """One declared file, link and mode, with the regime it came from."""

    spec: FileTransferSpec
    link: LinkConditions
    mode: TransferMode
    regime: str
    seed: int

    def features(self) -> dict[str, float]:
        """The feature row for this scenario.

        Contains file and link quantities and three cheap derived physical
        quantities -- the clean-pass time, the turnaround time and the
        availability. It deliberately does **not** contain the analytic
        delivery prediction: feeding the closed form in as a feature would turn
        the learned component into a residual corrector and make the comparison
        against the closed form circular.
        """
        link = self.link
        outage = link.outage
        mean_down = outage.mean_down_s if not outage.is_outage_free else 0.0
        return {
            "log10_pdu_count": float(np.log10(self.spec.pdu_count)),
            "log10_pdu_payload_bytes": float(np.log10(self.spec.pdu_payload_bytes)),
            "log10_file_bytes": float(np.log10(self.spec.file_bytes)),
            "pdu_loss_probability": float(link.pdu_loss_probability),
            "log10_forward_rate_bps": float(np.log10(link.forward_rate_bps)),
            "log10_asymmetry": float(np.log10(link.asymmetry)),
            "log10_one_way_light_time_s": float(
                np.log10(max(link.one_way_light_time_s, 1e-6))
            ),
            "outage_duty": float(outage.duty_cycle),
            "log10_mean_down_s": float(np.log10(mean_down + 1.0)),
            "contact_duty": float(link.contact_duty_cycle),
            "availability": float(link.availability),
            "log10_clean_pass_s": float(np.log10(clean_pass_seconds(self.spec, link))),
            "log10_turnaround_s": float(
                np.log10(turnaround_seconds(link.without_outages()))
            ),
            "mode_is_immediate_nak": float(self.mode is TransferMode.IMMEDIATE_NAK),
        }


def _loguniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(10.0 ** rng.uniform(*bounds))


def generate_scenario(
    seed: int, regime: str, ranges: ScenarioRanges | None = None
) -> Scenario:
    """One scenario from ``seed`` and ``regime``. Deterministic in both."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}; got {regime!r}")
    ranges = ranges or ScenarioRanges()
    rng = np.random.default_rng([int(seed), REGIMES.index(regime)])

    pdu_count = int(round(_loguniform(rng, ranges.log10_pdu_count)))
    pdu_count = max(2, pdu_count)
    payload = int(rng.choice(ranges.payload_choices))
    file_bytes = (pdu_count - 1) * payload + int(rng.integers(1, payload + 1))
    spec = FileTransferSpec(file_bytes, payload, ranges.pdu_header_bytes)

    forward = _loguniform(rng, ranges.log10_forward_rate_bps)
    asymmetry = _loguniform(rng, ranges.log10_asymmetry)
    light = _loguniform(rng, ranges.log10_light_time_s)
    loss = float(rng.uniform(*ranges.loss_probability))
    mode = ranges.modes[int(rng.integers(0, len(ranges.modes)))]

    if regime == "no_outage":
        link = LinkConditions(
            forward_rate_bps=forward,
            return_rate_bps=forward / asymmetry,
            one_way_light_time_s=light,
            pdu_loss_probability=loss,
            outage=OutageModel.none(),
            windows=(),
        )
    else:
        # The clean-pass time sets the time scale of the whole scenario, so the
        # outage and contact scales are drawn relative to it. See the module
        # docstring for why an absolute range does not produce a regime.
        clean_pass_s = 8.0 * spec.total_wire_bytes / forward
        mean_down = clean_pass_s * _loguniform(
            rng, ranges.log10_mean_down_per_clean_pass
        )
        duty = float(rng.uniform(*ranges.outage_duty))
        mean_up = mean_down * duty / (1.0 - duty)
        count = int(rng.choice(ranges.window_count_choices))
        duration = clean_pass_s * _loguniform(
            rng, ranges.log10_window_duration_per_clean_pass
        )
        contact_duty = float(rng.uniform(*ranges.contact_duty))
        link = LinkConditions(
            forward_rate_bps=forward,
            return_rate_bps=forward / asymmetry,
            one_way_light_time_s=light,
            pdu_loss_probability=loss,
            outage=OutageModel(mean_up_s=mean_up, mean_down_s=mean_down),
            windows=uniform_windows(count, duration, duration / contact_duty),
        )
    return Scenario(spec=spec, link=link, mode=mode, regime=regime, seed=int(seed))


def generate_table(
    rows: int,
    seed: int,
    regimes: tuple[str, ...] = REGIMES,
    ranges: ScenarioRanges | None = None,
) -> pd.DataFrame:
    """Build a feature/target table of ``rows`` scenarios per regime.

    One simulated run per scenario. Returns every row, completed or not, with a
    ``completed`` column; :func:`fit_split` drops the incomplete ones. The
    returned frame also carries the raw scenario quantities needed to rebuild
    the analytic prediction without re-simulating, so the benchmark never has
    to re-run the simulator to score a baseline.
    """
    if rows < 1:
        raise ValueError(f"rows must be at least 1; got {rows}")
    records: list[dict[str, float | str | bool]] = []
    for regime in regimes:
        for index in range(rows):
            scenario_seed = seed + index
            scenario = generate_scenario(scenario_seed, regime, ranges)
            result = simulate_transfer(
                scenario.spec, scenario.link, scenario.mode, scenario_seed * 7919 + 13
            )
            row: dict[str, float | str | bool] = dict(scenario.features())
            row.update(
                {
                    "regime": regime,
                    "mode": scenario.mode.value,
                    "seed": scenario_seed,
                    "completed": bool(result.delivered),
                    TARGET_COLUMN: (
                        float(result.delivery_time_s) if result.delivered else np.nan
                    ),
                    "forward_transmissions": float(result.forward_transmissions),
                    "overhead_ratio": float(result.overhead_ratio),
                    "rounds": float(result.rounds),
                    "pdu_count": float(scenario.spec.pdu_count),
                    "file_bytes": float(scenario.spec.file_bytes),
                    "pdu_payload_bytes": float(scenario.spec.pdu_payload_bytes),
                    "forward_rate_bps": float(scenario.link.forward_rate_bps),
                    "return_rate_bps": float(scenario.link.return_rate_bps),
                    "one_way_light_time_s": float(scenario.link.one_way_light_time_s),
                    "eof_bytes": float(scenario.link.eof_bytes),
                    "nak_bytes": float(scenario.link.nak_bytes),
                }
            )
            records.append(row)
    return pd.DataFrame.from_records(records)


def fit_split(table: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for regression: completed runs with a finite target."""
    usable = table[table["completed"] & np.isfinite(table[TARGET_COLUMN])]
    return usable.reset_index(drop=True)


def feature_matrix(table: pd.DataFrame) -> np.ndarray:
    """``(rows, len(FEATURE_COLUMNS))`` float array in declared column order."""
    missing = [name for name in FEATURE_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"table is missing feature columns: {missing}")
    return np.asarray(table[list(FEATURE_COLUMNS)], dtype=float)


def targets(table: pd.DataFrame) -> np.ndarray:
    """Delivery times in seconds as a float array."""
    return np.asarray(table[TARGET_COLUMN], dtype=float)


def scenario_from_row(row: pd.Series) -> Scenario:
    """Rebuild the :class:`Scenario` a table row came from.

    Uses the stored seed and regime rather than the stored columns, so the
    rebuild is exact and cannot drift from the generator.
    """
    return generate_scenario(int(row["seed"]), str(row["regime"]))
