"""How much of the measured per-stage variance co-varies with core frequency.

SPEC §6 requires this **measured, not asserted**, and requires the unpinned
clock stated with the headline rather than in a limitations list. Every figure
this product reports about the board is a figure at an unpinned,
``schedutil``-governed clock in ``MAXN_SUPER``, with ``sudo -n true`` failing on
the board so neither the governor nor the power mode could be changed by the
session that measured it.

``env.csv`` samples the six ``scaling_cur_freq`` values, four thermal zones and
the load average before each cell, at ``pass0``, ``pass10``, ``pass20``,
``pass30``, ``pass40`` and ``pass50`` within it, and after it. The six
within-cell samples are each taken immediately before the pass they are named
for, so each one pairs with that pass's own per-stage summary row: 6 samples x 9
cells = 54 paired points per stage.

Two co-variation figures per stage, and the second is the one that means
something
---------------------------------------------------------------------------

* **pooled** — ``r_squared`` of the per-pass stage median on mean core
  frequency across all 54 points. This number is contaminated: frame dimension
  moves both the stage time and, through the governor's load response, the
  frequency, so a pooled association can be dimension and not clock.
* **within-cell** — the same regression after subtracting each cell's own mean
  from both variables. Frame dimension is constant within a cell, so this is
  the share of the *residual* per-stage variance that moves with the clock.

Both are reported. Neither is asserted in advance.

Units: frequency in kHz as the kernel reports it, times in nanoseconds,
``r_squared`` dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .divergence import PassUnits
from .drift import LinearFit, linear_fit
from .traces import Campaign

#: The within-cell env sample points of ``env.csv`` and the pass index each one
#: immediately precedes. Literals, because the campaign's sampling cadence is
#: ``p % max(1, n_passes // 6) == 0`` with 60 passes, and a different cadence
#: would pair a sample with the wrong pass.
PASS_SAMPLE_POINTS: dict[str, int] = {
    "pass0": 0,
    "pass10": 10,
    "pass20": 20,
    "pass30": 30,
    "pass40": 40,
    "pass50": 50,
}


@dataclass(frozen=True)
class StageCovariation:
    """Frequency co-variation of one stage, pooled and within-cell."""

    stage: str
    pooled: LinearFit
    within_cell: LinearFit
    pooled_pearson: float
    within_cell_pearson: float
    n_points: int

    @property
    def pooled_variance_share(self) -> float:
        """Share of pooled per-stage variance explained by mean core frequency."""
        return self.pooled.r_squared

    @property
    def within_cell_variance_share(self) -> float:
        """Share of within-cell per-stage variance explained by frequency."""
        return self.within_cell.r_squared


@dataclass(frozen=True)
class FrequencyCovariation:
    """Per-stage frequency co-variation plus the clock's own spread."""

    stages: tuple[str, ...]
    paired: pd.DataFrame
    per_stage: dict[str, StageCovariation]
    frequency_columns: tuple[str, ...]
    frequency_min_khz: float
    frequency_max_khz: float
    frequency_mean_khz: float
    frequency_relative_spread: float
    governor: str
    power_mode: str
    sudo_passwordless: bool

    @property
    def max_within_cell_share(self) -> float:
        """Largest within-cell variance share over the six stages."""
        return max(item.within_cell_variance_share for item in self.per_stage.values())

    @property
    def max_within_cell_stage(self) -> str:
        """Stage with the largest within-cell variance share."""
        return max(
            self.per_stage.values(), key=lambda item: item.within_cell_variance_share
        ).stage

    @property
    def max_pooled_share(self) -> float:
        """Largest pooled variance share over the six stages."""
        return max(item.pooled_variance_share for item in self.per_stage.values())

    def frame(self) -> pd.DataFrame:
        """Per-stage co-variation table in pipeline order."""
        return pd.DataFrame(
            [
                {
                    "stage": stage,
                    "pooled_r_squared": self.per_stage[stage].pooled_variance_share,
                    "pooled_pearson": self.per_stage[stage].pooled_pearson,
                    "within_cell_r_squared": self.per_stage[
                        stage
                    ].within_cell_variance_share,
                    "within_cell_pearson": self.per_stage[stage].within_cell_pearson,
                    "n_points": self.per_stage[stage].n_points,
                }
                for stage in self.stages
            ]
        )


def pair_frequency_with_passes(campaign: Campaign, units: PassUnits) -> pd.DataFrame:
    """Join each within-cell env sample to the pass it immediately precedes.

    Returns one row per ``(cell, pass_idx)`` that has an env sample, with the
    mean and per-core frequencies beside that pass's six stage medians.

    Raises
    ------
    ValueError
        If no env sample pairs with a pass, which would mean the sampling
        cadence is not the one :data:`PASS_SAMPLE_POINTS` pins.
    """
    env = campaign.env
    columns = campaign.frequency_columns
    if not columns:
        raise ValueError("env.csv carries no freq_cpu* column; nothing to co-vary with")
    samples = env.loc[env["point"].isin(PASS_SAMPLE_POINTS)].copy()
    samples["pass_idx"] = samples["point"].map(PASS_SAMPLE_POINTS).astype(int)
    samples = samples.rename(columns={"run_id": "cell"})
    samples["freq_mean_khz"] = samples[list(columns)].mean(axis=1)

    hardware = pd.DataFrame(units.hw, columns=list(units.stages))
    hardware["cell"] = list(units.cells)
    hardware["pass_idx"] = units.pass_index

    keep = ["cell", "pass_idx", "freq_mean_khz", *columns, "load1", "load5"]
    joined = hardware.merge(samples[keep], on=["cell", "pass_idx"], how="inner")
    if joined.empty:
        raise ValueError(
            "no env sample pairs with a pass summary. env.csv points present: "
            f"{', '.join(sorted(env['point'].drop_duplicates()))}; expected some of "
            f"{', '.join(sorted(PASS_SAMPLE_POINTS))}."
        )
    return joined


def frequency_covariation(campaign: Campaign, units: PassUnits) -> FrequencyCovariation:
    """Measure the frequency co-variation SPEC §6 requires.

    Pooled and within-cell regressions of each stage's per-pass median on mean
    core frequency. Nothing here is asserted: if the clock did not move with the
    stage times, these numbers come out near zero and that is the measurement.
    """
    paired = pair_frequency_with_passes(campaign, units)
    columns = campaign.frequency_columns
    per_stage: dict[str, StageCovariation] = {}
    for stage in units.stages:
        frequency = paired["freq_mean_khz"].to_numpy(dtype=np.float64)
        timing = paired[stage].to_numpy(dtype=np.float64)
        pooled = linear_fit(
            frequency, timing, x_label="freq_mean_khz", y_label=f"{stage}_median_ns"
        )
        centred = paired.copy()
        centred["freq_centred"] = frequency - centred.groupby("cell")[
            "freq_mean_khz"
        ].transform("mean")
        centred["time_centred"] = timing - centred.groupby("cell")[stage].transform("mean")
        within = linear_fit(
            centred["freq_centred"].to_numpy(dtype=np.float64),
            centred["time_centred"].to_numpy(dtype=np.float64),
            x_label="freq_mean_khz_within_cell",
            y_label=f"{stage}_median_ns_within_cell",
        )
        per_stage[stage] = StageCovariation(
            stage=stage,
            pooled=pooled,
            within_cell=within,
            pooled_pearson=float(np.corrcoef(frequency, timing)[0, 1]),
            within_cell_pearson=float(
                np.corrcoef(
                    centred["freq_centred"].to_numpy(dtype=np.float64),
                    centred["time_centred"].to_numpy(dtype=np.float64),
                )[0, 1]
            ),
            n_points=int(len(paired)),
        )
    all_frequencies = campaign.env[list(columns)].to_numpy(dtype=np.float64).ravel()
    all_frequencies = all_frequencies[all_frequencies > 0.0]
    return FrequencyCovariation(
        stages=units.stages,
        paired=paired,
        per_stage=per_stage,
        frequency_columns=tuple(columns),
        frequency_min_khz=float(all_frequencies.min()),
        frequency_max_khz=float(all_frequencies.max()),
        frequency_mean_khz=float(all_frequencies.mean()),
        frequency_relative_spread=float(
            (all_frequencies.max() - all_frequencies.min()) / all_frequencies.mean()
        ),
        governor=str(campaign.meta["governor"]),
        power_mode=str(campaign.meta["nvpmodel"]),
        sudo_passwordless=bool(campaign.meta["sudo_passwordless"]),
    )
