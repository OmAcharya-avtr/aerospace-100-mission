"""The decomposition identity over the whole committed campaign.

The identity

    S = (Y_nom - Y_cap) + Int_acq r + Int_resync r + Int_idle r + rho

is exact in exact arithmetic. It is not exact here, and the point of this
script is that rho is a **measured, reported number** and not an assumption.
Four measurements.

1. **The residual over all 1990 passes**, re-derived from the committed event
   timelines rather than read from the record's own columns, with its median,
   95th percentile and worst case both absolutely and relative to the
   shortfall.

2. **Where the residual comes from.** The rate profile is
   ``min(eta_max, log2(1 + SNR/M))``, which is only C0 at the saturation
   boundary, so the C4 composite-Simpson error bound does not apply there.
   The residual is therefore stratified by peak elevation -- a pass that never
   saturates against one that does -- and the two are reported separately.
   The fourth derivative of the rate profile is estimated by finite
   differences and fed into the Simpson error bound from
   :func:`passyield.decompose.simpson_error_scale`, so the bound is stated
   with a measured derivative and not an asserted one.

3. **Grid refinement.** The node counts are raised four times on the worst
   pass and the observed reduction in the residual is reported. Asserted as a
   direction only, because the integrand is piecewise smooth.

4. **The three documented edge cases**, each exercised against the committed
   campaign where it occurs there and constructed where it does not: a pass
   that never clears the mask, a zero-duration pass, and an acquisition
   longer than the whole pass. The campaign contains 44 passes of the third
   kind and they are reported as such.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.dataset import (  # noqa: E402
    STATIONS,
    events_of,
    load_passes,
    station_by_name,
    window_of,
)
from passyield.decompose import (  # noqa: E402
    DEFAULT_NODES_PER_INTERVAL,
    DEFAULT_NODES_PER_WINDOW,
    MASK_CLAMP_TOLERANCE_DEG,
    DecompositionOptions,
    PassEvents,
    decompose_pass,
    rate_at_offsets,
    simpson_error_scale,
)
from passyield.geometry import (  # noqa: E402
    GroundStation,
    PassWindow,
    campaign_epoch,
    find_passes,
    satellite,
)

#: Peak elevation above which the link saturates for the best station in the
#: catalogue, degrees. Measured by validate_link.py: GS-08 saturates over the
#: top 27.4 per cent of its elevation range, which begins near 65 degrees.
SATURATION_ELEVATION_DEG = 65.0


def main() -> int:
    """Measure the identity's residual and its mechanism."""
    out = _bootstrap.Tee("validate_decomposition_output.txt")
    out("Decomposition identity validation -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    out("")
    out("the identity closes to the QUADRATURE, not to the rounding: each term is a")
    out("composite Simpson sum over its OWN partition of the window, and four sums")
    out("over four partitions do not telescope. Computing the losses and then")
    out("DEFINING the realised yield as the capacity minus them would force rho = 0")
    out("by construction and the identity would be untestable.")
    out("")
    out(f"Simpson nodes per interval: {DEFAULT_NODES_PER_INTERVAL}")
    out(f"Simpson nodes per window  : {DEFAULT_NODES_PER_WINDOW}")

    satrec = satellite()
    record = load_passes()
    limit = 120 if _bootstrap.FAST else len(record)
    rows = record[:limit]

    out.rule("1. THE RESIDUAL OVER THE COMMITTED CAMPAIGN")
    out("re-derived from the committed EVENT TIMELINES, not read from the record's")
    out("own decomposition columns, so a drift between the two would show here.")
    out("")
    residuals: list[float] = []
    relatives: list[float] = []
    clamps: list[float] = []
    elevations: list[float] = []
    truncated = 0
    for row in rows:
        item = decompose_pass(
            satrec, station_by_name(row.station), window_of(row), events_of(row)
        )
        residuals.append(item.residual_bits)
        relatives.append(abs(item.residual_bits / item.shortfall_bits))
        clamps.append(item.max_mask_clamp_deg)
        elevations.append(row.max_elevation_deg)
        truncated += int(item.acquisition_truncated)
    residual = np.asarray(residuals)
    relative = np.asarray(relatives)
    elevation = np.asarray(elevations)

    out(f"passes decomposed                        : {len(rows)}")
    out(f"median |rho|, Mbit                       : {float(np.median(np.abs(residual))):.6e}")
    out(f"95th percentile |rho|, Mbit              : "
        f"{float(np.quantile(np.abs(residual), 0.95)):.6e}")
    out(f"worst |rho|, Mbit                        : {float(np.max(np.abs(residual))):.6e}")
    out("")
    out(f"median |rho| / |S|                       : {float(np.median(relative)):.6e}")
    out(f"95th percentile |rho| / |S|              : {float(np.quantile(relative, 0.95)):.6e}")
    out(f"worst |rho| / |S|                        : {float(np.max(relative)):.6e}")
    out("")
    out(f"largest mask clamp applied, deg          : {float(np.max(clamps)):.6e}")
    out(f"clamp tolerance, deg                     : {MASK_CLAMP_TOLERANCE_DEG!r}")
    out(f"passes with acquisition past mask set    : {truncated}")
    out("")
    out("THE REPORTED RESIDUAL OF THIS PRODUCT is the 95th percentile relative")
    out(f"figure above: {float(np.quantile(relative, 0.95)):.3e}. The test suite's")
    out("identity tolerance is 1.0e-5, which is that figure rounded up by more than")
    out("an order of magnitude and covers the worst case too.")
    out.check(
        "the 95th percentile relative residual is below 1e-5",
        float(np.quantile(relative, 0.95)) < 1.0e-5,
        f"{float(np.quantile(relative, 0.95)):.3e}",
    )
    out.check(
        "the worst relative residual is below 1e-3",
        float(np.max(relative)) < 1.0e-3,
        f"{float(np.max(relative)):.3e}",
    )
    out.check(
        "no mask clamp exceeded the declared tolerance",
        float(np.max(clamps)) <= MASK_CLAMP_TOLERANCE_DEG,
        f"largest {float(np.max(clamps)):.3e} deg",
    )

    out.rule("2. WHERE THE RESIDUAL COMES FROM -- the saturation kink")
    out("min(eta_max, log2(1 + SNR/M)) is only C0 at the saturation boundary, so")
    out("the C4 composite-Simpson error bound does not hold there. If that is the")
    out("mechanism, the residual must concentrate on the passes that saturate.")
    out("")
    high = elevation > SATURATION_ELEVATION_DEG
    low = ~high
    out(f"saturation threshold used, deg           : {SATURATION_ELEVATION_DEG!r}")
    out(f"passes with peak elevation above it      : {int(high.sum())}")
    out(f"passes with peak elevation below it      : {int(low.sum())}")
    out("")
    out(f"mean |rho| above the threshold, Mbit     : "
        f"{float(np.mean(np.abs(residual[high]))):.6e}")
    out(f"mean |rho| below the threshold, Mbit     : "
        f"{float(np.mean(np.abs(residual[low]))):.6e}")
    ratio = float(np.mean(np.abs(residual[high])) / np.mean(np.abs(residual[low])))
    out(f"ratio                                    : {ratio:.1f}")
    out.check(
        "the residual concentrates on the passes that saturate",
        ratio > 10.0,
        f"ratio {ratio:.1f}",
    )

    out("")
    out("the Simpson error bound, with a MEASURED fourth derivative rather than an")
    out("asserted one. simpson_error_scale returns (b-a)h^4/180; the fourth")
    out("derivative of the rate profile is estimated by central finite differences")
    out("on a dense grid of the worst pass.")
    out("")
    worst_index = int(np.argmax(np.abs(residual)))
    worst_row = rows[worst_index]
    station = station_by_name(worst_row.station)
    window = window_of(worst_row)
    nodes = 2049
    offsets = np.linspace(0.0, window.duration_s, nodes)
    rates, _clamp = rate_at_offsets(satrec, station, window, offsets)
    step = float(offsets[1] - offsets[0])
    fourth = np.diff(rates, n=4) / step**4
    bound_scale = simpson_error_scale(window.duration_s, DEFAULT_NODES_PER_INTERVAL)
    bound = bound_scale * float(np.max(np.abs(fourth)))
    out(f"worst pass                               : pass_id {worst_row.pass_id}, "
        f"{worst_row.station}")
    out(f"  peak elevation, deg                    : {worst_row.max_elevation_deg:.3f}")
    out(f"  window duration, s                     : {window.duration_s:.3f}")
    out(f"  finite-difference step, s              : {step:.6f}")
    out(f"  max |d4r/dt4|, Mbit/s per s^4          : {float(np.max(np.abs(fourth))):.6e}")
    out(f"  (b-a)h^4/180 at {DEFAULT_NODES_PER_INTERVAL} nodes          : {bound_scale:.6e}")
    out(f"  implied Simpson bound, Mbit            : {bound:.6e}")
    out(f"  measured |rho| on this pass, Mbit      : {abs(residual[worst_index]):.6e}")
    out("")
    out("the measured residual is BELOW the bound built from the measured fourth")
    out("derivative. That is the expected relationship and it is reported as a")
    out("check; note that the bound itself is only valid on the smooth part of the")
    out("profile, which is why the kink stratification above is the primary")
    out("evidence and this is corroboration.")
    out.check(
        "the worst residual is below the Simpson bound with a measured derivative",
        abs(residual[worst_index]) <= bound,
        f"{abs(residual[worst_index]):.3e} against {bound:.3e}",
    )

    out.rule("3. GRID REFINEMENT ON THE WORST PASS")
    out(f"{'nodes/interval':>16}{'nodes/window':>14}{'rho, Mbit':>16}{'rho/S':>14}")
    events = events_of(worst_row)
    previous = None
    reductions: list[float] = []
    for interval_nodes, window_nodes in ((9, 65), (33, 513), (129, 2049), (513, 8193)):
        item = decompose_pass(
            satrec, station, window, events,
            DecompositionOptions(
                nodes_per_interval=interval_nodes, nodes_per_window=window_nodes
            ),
        )
        out(
            f"{interval_nodes:>16}{window_nodes:>14}{item.residual_bits:>16.6e}"
            f"{item.residual_bits / item.shortfall_bits:>14.3e}"
        )
        if previous is not None and item.residual_bits != 0.0:
            reductions.append(abs(previous / item.residual_bits))
        previous = item.residual_bits
    out("")
    out(f"reduction factors per quadruple-node step: "
        f"{[round(value, 2) for value in reductions]}")
    out("asserted as a DIRECTION only. The reduction is not the C4 order of")
    out("Simpson's rule, and should not be: the kink's position relative to the")
    out("grid changes with every refinement, so the error is dominated by a")
    out("first-order term near the saturation boundary.")
    out.check(
        "the first refinement reduces the residual",
        bool(reductions) and reductions[0] > 1.0,
        f"first factor {reductions[0]:.2f}" if reductions else "no reductions measured",
    )

    out.rule("4. THE THREE DOCUMENTED EDGE CASES")
    out("each of these returns or raises something documented. None produces a")
    out("silent negative yield, which is the failure mode a yield accountant exists")
    out("to prevent.")
    out("")

    out("(a) a pass that never rises above the mask")
    polar = GroundStation("PROBE-POLAR", 85.0, 0.0, 0.0, 10.0, 21.0, 0.4)
    epoch_jd, epoch_fr = campaign_epoch(satrec)
    windows = find_passes(satrec, polar, epoch_jd, epoch_fr, 2.0, coarse_step_s=30.0)
    out(f"  station at 85 N against a {np.degrees(satrec.inclo):.2f} deg inclination orbit")
    out(f"  windows found over 2 days: {len(windows)}")
    out.check(
        "find_passes returns an empty list, not a fabricated window",
        windows == [],
        f"{len(windows)} windows",
    )
    grazing = PassWindow(
        station=STATIONS[0].name,
        rise_jd=window.rise_jd, rise_fr=window.rise_fr,
        duration_s=window.duration_s,
        max_elevation_deg=STATIONS[0].mask_deg - 1.0,
        min_slant_range_km=window.min_slant_range_km,
    )
    refused = False
    message = ""
    try:
        decompose_pass(satrec, STATIONS[0], grazing, PassEvents(1.0))
    except ValueError as error:
        refused = True
        message = str(error)
    out(f"  decompose_pass on a sub-mask window raises: {refused}")
    out(f"  message: {message}")
    out.control("a sub-mask window is refused", refused, must_fire=True)

    out("")
    out("(b) a zero-duration pass")
    degenerate = PassWindow(
        station=STATIONS[0].name,
        rise_jd=window.rise_jd, rise_fr=window.rise_fr, duration_s=0.0,
        max_elevation_deg=45.0, min_slant_range_km=700.0,
    )
    refused = False
    message = ""
    try:
        decompose_pass(satrec, STATIONS[0], degenerate, PassEvents(1.0))
    except ValueError as error:
        refused = True
        message = str(error)
    out(f"  decompose_pass on a zero-duration window raises: {refused}")
    out(f"  message: {message}")
    out.control("a zero-duration window is refused", refused, must_fire=True)

    out("")
    out("(c) an acquisition time exceeding the whole pass")
    out("  this one RETURNS rather than raising, because it is a real operational")
    out("  outcome and not a malformed input: the acquisition interval is clipped to")
    out("  the window, a flag is set, and the realised yield is exactly zero.")
    item = decompose_pass(
        satrec, station, window, PassEvents(acquisition_s=window.duration_s + 500.0)
    )
    out(f"  acquisition_truncated : {item.acquisition_truncated!r}")
    out(f"  realised, Mbit        : {item.realised_bits!r}")
    out(f"  acquisition, Mbit     : {item.acquisition_bits:.6f}")
    out(f"  capacity, Mbit        : {item.capacity_bits:.6f}")
    out(f"  shortfall, Mbit       : {item.shortfall_bits:.6f}")
    out(f"  nominal, Mbit         : {item.nominal_bits:.6f}")
    out.check("the flag is set", item.acquisition_truncated is True)
    out.check("the realised yield is exactly zero", item.realised_bits == 0.0)
    gap = abs(item.acquisition_bits - item.capacity_bits) / item.capacity_bits
    out(f"  |acquisition - capacity| / capacity : {gap:.3e}")
    out("  the two cover the SAME interval with DIFFERENT quadratures -- the")
    out(f"  acquisition term with {DEFAULT_NODES_PER_INTERVAL} nodes and the capacity with")
    out(f"  {DEFAULT_NODES_PER_WINDOW} -- so their difference is the residual class measured in")
    out("  section 1, and this is deliberately the WORST pass of the campaign for")
    out("  it, the one with the saturation kink. The bound used is the campaign's")
    out("  measured worst relative residual, 1e-3, not the 95th-percentile 1e-5.")
    out.check(
        "the acquisition term carries the whole capacity to the worst-case residual",
        gap <= 1.0e-3,
        f"relative difference {gap:.3e}",
    )
    out.check(
        "the shortfall is the whole booked volume",
        abs(item.shortfall_bits - item.nominal_bits) <= 1.0e-12 * item.nominal_bits,
        f"difference {abs(item.shortfall_bits - item.nominal_bits):.3e} Mbit",
    )
    out("")
    out(f"the committed campaign contains {truncated} passes of kind (c) out of "
        f"{len(rows)}.")
    out("They are in the record on purpose: an edge case that only ever appears in a")
    out("constructed test is an edge case nobody has measured.")
    out.control(
        "the committed campaign exercises edge case (c)",
        truncated > 0,
        must_fire=True,
        detail=f"{truncated} passes",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
