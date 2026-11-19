"""Plain-text rendering of the product's results. No library code prints.

Every function here returns a string. Nothing in :mod:`passyield` calls
``print``; the command-line interface in :mod:`passyield.__main__`, the
examples and the validation scripts are the only things that write to a
stream, and they all render through this module so that a number appears in
one format everywhere it appears at all.
"""

from __future__ import annotations

from .baselines import AccountedYield
from .decompose import Decomposition
from .distribution import ShortfallDistribution

#: Quantiles the product reports by default. The headline is the distribution,
#: so the report is a row of quantiles and not a single number.
REPORTED_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def format_decomposition(decomposition: Decomposition) -> str:
    """One pass's shortfall and its four terms, Mbit."""
    item = decomposition

    def share(value: float) -> str:
        """A term's percentage of the shortfall, or ``n/a`` when it is zero."""
        if item.shortfall_bits == 0.0:
            return "    n/a"
        return f"{100.0 * value / item.shortfall_bits:7.3f}"

    lines = [
        f"pass decomposition -- station {item.station}",
        f"  window duration, s          {item.window_duration_s:12.3f}",
        f"  active duration, s          {item.active_duration_s:12.3f}",
        f"  nominal (booked), Mbit      {item.nominal_bits:12.3f}",
        f"  window capacity, Mbit       {item.capacity_bits:12.3f}",
        f"  realised, Mbit              {item.realised_bits:12.3f}",
        f"  realised / capacity         {item.realised_fraction:12.6f}",
        "",
        f"  SHORTFALL, Mbit             {item.shortfall_bits:12.3f}   share, per cent",
        f"    geometry                  {item.geometry_bits:12.3f}   {share(item.geometry_bits)}",
        f"    acquisition               {item.acquisition_bits:12.3f}"
        f"   {share(item.acquisition_bits)}",
        f"    re-sync                   {item.resync_bits:12.3f}   {share(item.resync_bits)}",
        f"    idle                      {item.idle_bits:12.3f}   {share(item.idle_bits)}",
        f"    residual (reported)       {item.residual_bits:12.3e}   {share(item.residual_bits)}",
        "",
        f"  acquisition truncated       {item.acquisition_truncated!r}",
        f"  largest mask clamp, deg     {item.max_mask_clamp_deg:12.3e}",
        f"  Simpson nodes per interval  {item.nodes_per_interval:12d}",
        f"  Simpson nodes per window    {item.nodes_per_window:12d}",
    ]
    return "\n".join(lines)


def format_distribution(
    distribution: ShortfallDistribution, levels: tuple[float, ...] = REPORTED_QUANTILES
) -> str:
    """A shortfall distribution's quantiles, Mbit. The headline output."""
    lines = [
        f"shortfall distribution -- method {distribution.method}",
        f"  grid step, Mbit             {distribution.grid_step_mbit:12.4f}",
        f"  quantile discretisation     {distribution.quantile_discretisation_mbit():12.4f}",
        f"  truncated mass              {distribution.truncated_mass:12.6e}",
        f"  resampling unit             {distribution.resampling_unit!r}",
        f"  mean, Mbit                  {distribution.mean_mbit():12.3f}",
        f"  standard deviation, Mbit    {distribution.standard_deviation_mbit():12.3f}",
        "",
        "  quantile      shortfall, Mbit",
    ]
    for level in levels:
        lines.append(f"  {level:8.3f}      {distribution.quantile(level):15.3f}")
    return "\n".join(lines)


def format_accounted(accounted: AccountedYield) -> str:
    """Baseline 3's campaign totals, shares and bootstrap intervals."""
    lines = [
        "accounted yield (baseline 3, ground truth)",
        f"  resampling unit             {accounted.resampling_unit!r}",
        f"  passes                      {accounted.passes:12d}",
        f"  bootstrap resamples         {accounted.resamples:12d}",
        f"  interval coverage           {accounted.confidence:12.3f}",
        "",
        f"  campaign nominal, Mbit      {accounted.nominal_mbit:16.1f}",
        f"  campaign capacity, Mbit     {accounted.capacity_mbit:16.1f}",
        f"  campaign realised, Mbit     {accounted.realised_mbit:16.1f}",
        f"  campaign shortfall, Mbit    {accounted.shortfall_mbit:16.1f}",
        "",
        f"  mean shortfall per pass     {accounted.mean_shortfall_mbit:16.3f} Mbit",
        f"    {accounted.confidence:.0%} bootstrap interval     "
        f"[{accounted.mean_shortfall_ci_mbit[0]:.3f}, {accounted.mean_shortfall_ci_mbit[1]:.3f}]",
        f"  realised / capacity         {accounted.realised_fraction:16.6f}",
        f"    {accounted.confidence:.0%} bootstrap interval     "
        f"[{accounted.realised_fraction_ci[0]:.6f}, {accounted.realised_fraction_ci[1]:.6f}]",
        "",
        "  share of total shortfall, per cent",
    ]
    for name in ("geometry", "acquisition", "resync", "idle"):
        lines.append(f"    {name:<24s}  {100.0 * accounted.shortfall_share[name]:10.4f}")
    lines.append(f"    {'residual (reported)':<24s}  {100.0 * accounted.residual_share:10.3e}")
    return "\n".join(lines)


def format_comparison(scores: dict[str, dict[str, float]], title: str) -> str:
    """A scoreboard: one row per predictor, one column per score.

    Columns: CRPS and MAE in Mbit, the PIT calibration error and the 80 per
    cent interval coverage, both dimensionless. Lower is better for the first
    three; for coverage the target is 0.800 and either direction away from it
    is a defect.
    """
    lines = [
        title,
        f"  {'predictor':<28s}{'CRPS, Mbit':>13s}{'MAE, Mbit':>13s}"
        f"{'PIT cal err':>13s}{'cov80':>9s}",
    ]
    for name, row in scores.items():
        lines.append(
            f"  {name:<28s}{row['crps']:>13.2f}{row['mae']:>13.2f}"
            f"{row['pit_calibration_error']:>13.4f}{row['coverage80']:>9.4f}"
        )
    return "\n".join(lines)
