"""The README's worked example, executed, so the README cannot drift from it.

Every line of the API usage block in ``README.md`` is executed here and its
printed output is captured into ``validation/worked_example_output.txt``. The
README quotes that file. Two products in consecutive batches of this mission
caught their own wrong README numbers by re-running the quoted commands after
the last code change, so this is the mechanism rather than the intention.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.baselines import (  # noqa: E402
    analytic_shortfall_distribution,
    booked_shortfall_mbit,
    fit_event_model,
)
from passyield.bounds import inside, shortfall_bracket  # noqa: E402
from passyield.dataset import (  # noqa: E402
    events_of,
    load_passes,
    parse_intervals,
    station_by_name,
    window_of,
)
from passyield.decompose import decompose_pass  # noqa: E402
from passyield.geometry import satellite  # noqa: E402
from passyield.link import DEFAULT_BUDGET  # noqa: E402

#: The pass the worked example uses. Declared, so the README's numbers have a
#: single source.
PASS_ID = 5


def main() -> int:
    """Run the worked example and capture its output."""
    out = _bootstrap.Tee("worked_example_output.txt")
    out("Worked example -- P099 PassYield")
    out("")
    out("this is the API block quoted in README.md, executed. The README quotes")
    out("THIS FILE, so the two cannot drift.")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")

    out.rule("THE CODE")
    out("""from passyield.baselines import analytic_shortfall_distribution, fit_event_model
from passyield.bounds import inside, shortfall_bracket
from passyield.dataset import events_of, load_passes, station_by_name, window_of
from passyield.decompose import decompose_pass
from passyield.geometry import satellite
from passyield.link import DEFAULT_BUDGET

record = load_passes()                       # the committed per-pass record
row = next(r for r in record if r.pass_id == 5)
satrec = satellite()                         # SGP4 on a published element set
station = station_by_name(row.station)
window = window_of(row)

# Baseline 3: what the record accounts for, decomposed.
item = decompose_pass(satrec, station, window, events_of(row))

# Baseline 2: the analytic law of the shortfall, as a DISTRIBUTION.
model = fit_event_model(...)                 # measured laws, see below
law = analytic_shortfall_distribution(
    satrec, station, window, model, item.nominal_bits, item.capacity_bits
)

# An independent bracket, from the pass duration and the rate ceiling alone.
bracket = shortfall_bracket(
    item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
)""")

    record = load_passes()
    row = next(entry for entry in record if entry.pass_id == PASS_ID)
    satrec = satellite()
    station = station_by_name(row.station)
    window = window_of(row)
    item = decompose_pass(satrec, station, window, events_of(row))
    model = fit_event_model(
        acquisition_s=[entry.acquisition_s for entry in record],
        outage_counts=[entry.outage_count for entry in record],
        window_durations_s=[entry.duration_s for entry in record],
        outage_durations_s=[
            duration for entry in record
            for _start, duration in parse_intervals(entry.resync_intervals)
        ],
        idle_totals_s=[entry.idle_total_s for entry in record],
    )
    law = analytic_shortfall_distribution(
        satrec, station, window, model, item.nominal_bits, item.capacity_bits
    )
    bracket = shortfall_bracket(
        item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
    )

    out.rule("THE OUTPUT")
    out(f"passes in the committed record        : {len(record)}")
    out(f"pass_id                               : {row.pass_id}")
    out(f"station                               : {row.station}")
    out(f"window duration, s                    : {item.window_duration_s:.3f}")
    out(f"peak elevation, deg                   : {row.max_elevation_deg:.3f}")
    out("")
    out(f"booked volume (baseline 1), Mbit      : {item.nominal_bits:.3f}")
    out(f"window capacity, Mbit                 : {item.capacity_bits:.3f}")
    out(f"accounted realised volume, Mbit       : {item.realised_bits:.3f}")
    out(f"accounted shortfall, Mbit             : {item.shortfall_bits:.3f}")
    out("")
    out(f"baseline 1's predicted shortfall, Mbit: {booked_shortfall_mbit()!r}")
    out("")
    out("the decomposition, Mbit:")
    out(f"  geometry                            : {item.geometry_bits:.3f}")
    out(f"  acquisition                         : {item.acquisition_bits:.3f}")
    out(f"  re-sync                             : {item.resync_bits:.3f}")
    out(f"  idle                                : {item.idle_bits:.3f}")
    out(f"  residual (REPORTED, not assumed)    : {item.residual_bits:.6e}")
    out(f"  sum of the four terms               : {item.decomposed_bits:.3f}")
    out("")
    out("the analytic law of the shortfall -- the headline output is this, not a")
    out("point number:")
    out(f"  method                              : {law.method}")
    out(f"  grid step, Mbit                     : {law.grid_step_mbit:.4f}")
    out(f"  quantile discretisation, Mbit       : {law.quantile_discretisation_mbit():.4f}")
    out(f"  truncated mass                      : {law.truncated_mass:.6e}")
    out(f"  mean, Mbit                          : {law.mean_mbit():.3f}")
    out("")
    for level in (0.05, 0.25, 0.50, 0.75, 0.95):
        out(f"  quantile {level:.2f}, Mbit                  : {law.quantile(level):.3f}")
    out("")
    out(f"independent bracket, Mbit             : [{bracket[0]:.3f}, {bracket[1]:.3f}]")
    out(f"the accounted shortfall is inside it  : {inside(item.shortfall_bits, bracket)}")
    out(f"the median of the law is inside it    : {inside(law.quantile(0.5), bracket)}")
    out("")
    out("what a scheduler would do with this: book the 0.25 quantile of the")
    out("shortfall against this station rather than the booked volume, which is")
    out(f"{law.quantile(0.25):.0f} Mbit of expected loss rather than the "
        f"{booked_shortfall_mbit():.0f} Mbit baseline 1 asserts.")

    out.rule("CHECKS")
    out.check(
        "the four terms total the shortfall within the reported residual",
        abs(item.decomposed_bits - item.shortfall_bits)
        <= 1.0e-5 * abs(item.shortfall_bits),
        f"residual {item.residual_bits:.3e} Mbit",
    )
    out.check(
        "the accounted shortfall lies inside the independent bracket",
        inside(item.shortfall_bits, bracket),
    )
    out.check(
        "the analytic law's median lies inside the independent bracket",
        inside(law.quantile(0.5), bracket),
    )
    out.check(
        "the realised yield is non-negative and at most the capacity",
        0.0 <= item.realised_bits <= item.capacity_bits,
        f"{item.realised_bits:.3f} of {item.capacity_bits:.3f} Mbit",
    )
    out.check(
        "the law reports its own truncated mass",
        law.truncated_mass >= 0.0,
        f"{law.truncated_mass:.6e}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
