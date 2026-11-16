"""How much of the measured per-stage variance co-varies with core frequency.

SPEC §6 requires this MEASURED, not asserted, and requires the unpinned clock
stated with the headline rather than in a limitations list. The measurement
comes out small, and the honest reading of a small number here is stated with
it: ``env.csv`` samples ``scaling_cur_freq`` six times per cell as a
point-in-time read, not as a time-average over the 1500 frames of the pass it
precedes, so what is bounded is what THIS SAMPLING CADENCE CAN SEE and not what
the unpinned clock did.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import frequency_covariation, load_campaign, pass_units  # noqa: E402
from hiltimingdiff.frequency import PASS_SAMPLE_POINTS  # noqa: E402

#: Pinned pooled and within-cell variance shares, to full double precision.
PINNED_POOLED = {
    "ingest": 0.0005050494884748513,
    "background": 0.002018092883406042,
    "threshold": 0.00428664980281479,
    "centroid": 0.005091334602689734,
    "gate": 0.005286479936121302,
    "track": 0.008537090338330722,
}
PINNED_WITHIN_CELL = {
    "ingest": 0.00019269536848354196,
    "background": 0.0038628918194912876,
    "threshold": 0.004570282291641248,
    "centroid": 0.001027382118275355,
    "gate": 0.0005143720856498613,
    "track": 0.0012054823553220295,
}


def main() -> int:
    tee = _bootstrap.Tee("validate_frequency_covariation_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- the unpinned clock, measured, SPEC §6")

    campaign = load_campaign()
    units = pass_units(campaign)
    covariation = frequency_covariation(campaign, units)

    tee.rule("the operating condition, as the board recorded it")
    tee(f"  governor                 : {covariation.governor}")
    tee(f"  power mode               : {covariation.power_mode}")
    tee(f"  passwordless sudo        : {covariation.sudo_passwordless}")
    tee(
        f"  per-core frequency       : {covariation.frequency_min_khz:.0f} to "
        f"{covariation.frequency_max_khz:.0f} kHz"
    )
    tee(f"  mean over all samples    : {covariation.frequency_mean_khz:.6f} kHz")
    tee(f"  relative spread          : {covariation.frequency_relative_spread:.9f}")
    tee(f"  cores sampled            : {len(covariation.frequency_columns)}")
    tee("")
    tee.check(
        "the governor was schedutil and was not pinned",
        covariation.governor == "schedutil",
    )
    tee.check(
        "passwordless sudo was unavailable, so the clock COULD NOT be pinned",
        covariation.sudo_passwordless is False,
        "this is a restriction on the result, recorded with the headline and not "
        "in a limitations list",
    )
    tee.check(
        "the power mode was MAXN_SUPER",
        "MAXN_SUPER" in covariation.power_mode,
    )
    tee.finding(
        "SPEC §6 recorded per-core scaling_cur_freq spanning 729600 to 1190400 kHz at "
        "idle BEFORE the campaign. During the campaign the clock reached "
        f"{covariation.frequency_max_khz:.0f} kHz, a wider span than the pre-campaign "
        "idle reading, so the specification's figure understates the range the "
        "measurements were taken over. The measured range is what is published."
    )

    tee.rule("per-core frequency, by sample point")
    env = campaign.env
    columns = list(covariation.frequency_columns)
    for point in ("before", "pass0", "pass30", "after"):
        rows = env.loc[env["point"] == point, columns]
        if rows.empty:
            continue
        values = rows.to_numpy(dtype=np.float64)
        tee(
            f"  {point:<8s} n={len(rows):>3d}  min {values.min():>9.0f}  "
            f"mean {values.mean():>11.1f}  max {values.max():>9.0f} kHz"
        )
    tee("")
    tee("  per core over every sample:")
    for name in columns:
        series = env[name].to_numpy(dtype=np.float64)
        series = series[series > 0.0]
        tee(
            f"    {name:<10s} min {series.min():>9.0f}  mean {series.mean():>11.1f}  "
            f"max {series.max():>9.0f} kHz"
        )

    tee.rule("the pairing: six within-cell env samples joined to the passes they precede")
    tee(f"  sample points used : {', '.join(sorted(PASS_SAMPLE_POINTS))}")
    tee(f"  paired points      : {len(covariation.paired)}  (6 samples x 9 cells)")
    tee("")
    tee.check(
        "the pairing yields 54 points",
        len(covariation.paired) == 54,
    )
    tee.check(
        "every cell contributes six paired points",
        set(covariation.paired.groupby("cell").size().unique()) == {6},
    )

    tee.rule("MEASURED co-variation, pooled and within-cell")
    tee(
        "  pooled is contaminated by frame dimension, which moves both the stage time "
        "and, through"
    )
    tee(
        "  the governor's load response, the frequency. within-cell subtracts each "
        "cell's own mean"
    )
    tee("  from both variables, and frame dimension is constant within a cell.")
    tee("")
    tee(
        "  stage        pooled r2     pooled r   within-cell r2   within-cell r   points"
    )
    for stage in covariation.stages:
        item = covariation.per_stage[stage]
        tee(
            f"  {stage:<11s} {item.pooled_variance_share:>10.9f} "
            f"{item.pooled_pearson:>+11.6f} "
            f"{item.within_cell_variance_share:>16.9f} "
            f"{item.within_cell_pearson:>+15.6f} {item.n_points:>8d}"
        )
    tee("")
    tee(
        f"  largest pooled share      : {covariation.max_pooled_share:.9f}"
    )
    tee(
        f"  largest within-cell share : {covariation.max_within_cell_share:.9f} "
        f"({covariation.max_within_cell_stage})"
    )
    tee("")
    for stage in covariation.stages:
        item = covariation.per_stage[stage]
        tee.check(
            f"pooled share for {stage} matches the pinned value",
            abs(item.pooled_variance_share - PINNED_POOLED[stage])
            <= 1e-9 * max(PINNED_POOLED[stage], 1e-12),
            f"pinned {PINNED_POOLED[stage]!r}, measured {item.pooled_variance_share!r}",
        )
        tee.check(
            f"within-cell share for {stage} matches the pinned value",
            abs(item.within_cell_variance_share - PINNED_WITHIN_CELL[stage])
            <= 1e-9 * max(PINNED_WITHIN_CELL[stage], 1e-12),
            f"pinned {PINNED_WITHIN_CELL[stage]!r}, measured "
            f"{item.within_cell_variance_share!r}",
        )
    tee.check(
        "every measured share is below one per cent of the per-stage variance",
        covariation.max_pooled_share < 0.01
        and covariation.max_within_cell_share < 0.01,
        f"largest pooled {covariation.max_pooled_share:.6f}, largest within-cell "
        f"{covariation.max_within_cell_share:.6f}",
    )
    tee.finding(
        "A MEASURED NULL, with its limit stated. Less than one per cent of the "
        "per-stage variance that this sampling can see moves with mean core frequency: "
        f"the largest pooled share is {covariation.max_pooled_share:.6f} (track) and "
        f"the largest within-cell share is {covariation.max_within_cell_share:.6f} "
        f"({covariation.max_within_cell_stage}). THIS IS NOT A CLAIM THAT THE UNPINNED "
        "CLOCK DID NOT MATTER. The clock is sampled six times per cell as a "
        "point-in-time read of scaling_cur_freq, not as a time-average over the 1500 "
        "frames of the pass it precedes, and each paired stage figure is a median over "
        "those 1500 frames. A per-frame association with the instantaneous clock is "
        "invisible to this cadence by construction. What is bounded is the association "
        "between a pass's median and a single frequency reading taken just before it."
    )
    tee.finding(
        "Two of the six within-cell associations are NEGATIVE in sign (centroid "
        f"{covariation.per_stage['centroid'].within_cell_pearson:+.6f}, gate "
        f"{covariation.per_stage['gate'].within_cell_pearson:+.6f}), which is the "
        "wrong direction for a clock effect: a higher clock should make a stage "
        "faster, so a positive correlation between frequency and time is already "
        "surprising and a negative one is what a clock effect would look like. With "
        "r_squared below 0.006 on 54 points neither sign is distinguishable from "
        "noise, and no direction is claimed."
    )

    tee.rule("thermal context, recorded because it is what drove the governor")
    for name in (
        "temp_cpu_thermal",
        "temp_gpu_thermal",
        "temp_soc0_thermal",
        "temp_tj_thermal",
    ):
        series = env[name].to_numpy(dtype=np.float64)
        series = series[series > 0.0]
        if series.size == 0:
            continue
        tee(
            f"  {name:<20s} min {series.min() / 1000.0:>6.2f}  mean "
            f"{series.mean() / 1000.0:>6.2f}  max {series.max() / 1000.0:>6.2f} degC"
        )
    load = env["load1"].to_numpy(dtype=np.float64)
    tee(
        f"  load1                min {load.min():.2f}  mean {load.mean():.2f}  "
        f"max {load.max():.2f}"
    )

    tee.rule("negative controls")
    # Anchored on the TRUE paired table. A stage column replaced by a strictly
    # increasing function of the frequency must show an r_squared near 1.
    paired = covariation.paired.copy()
    frequency = paired["freq_mean_khz"].to_numpy(dtype=np.float64)
    from hiltimingdiff import linear_fit

    synthetic = linear_fit(frequency, 3.0 * frequency + 11.0, x_label="f", y_label="y")
    tee.control(
        "a stage time that IS a function of the frequency is detected",
        synthetic.r_squared > 0.999,
        must_fire=True,
        detail=f"r_squared {synthetic.r_squared:.9f}: the regression can see a real "
        "association, so a small measured share is a measurement and not a blind spot "
        "in the estimator",
    )
    generator = np.random.default_rng(6060)
    noise = generator.normal(0.0, 1.0, size=frequency.size)
    noise_fit = linear_fit(frequency, noise, x_label="f", y_label="noise")
    tee.control(
        "pure noise shows a large association",
        noise_fit.r_squared > 0.1,
        must_fire=False,
        detail=f"r_squared {noise_fit.r_squared:.9f} on {frequency.size} points, which "
        "is the scale of association this cadence cannot distinguish from zero",
    )
    tee(
        f"\n  NOTE: the noise control's r_squared, {noise_fit.r_squared:.6f}, is the "
        "same order as"
    )
    tee(
        "  the measured shares. On 54 points an r_squared of a few thousandths is "
        "indistinguishable"
    )
    tee("  from zero, and that is the honest resolution of this measurement.")

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
