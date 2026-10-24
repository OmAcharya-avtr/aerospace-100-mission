"""Regenerate the three shipped YAML cases from the built-in scenarios.

The case files in ``examples/cases/`` are not hand-written state vectors. They
are the output of ``pointahead.scenario``, serialised by
``pointahead.cases.dump_case``, with a header comment prepended. Running this
script must leave them byte for byte unchanged; if it does not, either a
scenario definition or the loader has moved and the committed cases are stale.

The script writes the files and reports whether each one changed, so a reader
can tell the difference between "regenerated identically" and "regenerated
differently" without reading a diff.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, header, relative

ROOT = bootstrap()

from pointahead.budget import BudgetInputs  # noqa: E402
from pointahead.cases import Case, dump_case, load_case  # noqa: E402
from pointahead.scenario import (  # noqa: E402
    DEFAULT_BEAM,
    geo_to_leo_pair,
    leo_crossing_pair,
    leo_trailing_pair,
)

HEADERS = {
    "leo_crossing": (
        "Two 600 km circular orbits on crossing planes, 60 degrees apart in\n"
        "inclination, the remote terminal 0.3 rad of arc along its own orbit.\n"
        "This is the geometry in which the relative acceleration is exactly\n"
        "radial, so the first-order ephemeris staleness term vanishes and the\n"
        "jerk term carries the whole line."
    ),
    "leo_trailing": (
        "Two 600 km circular orbits in the same plane, 0.3 rad of arc apart.\n"
        "The transverse relative velocity is small, so this is the geometry\n"
        "with the smallest point-ahead angle of the three."
    ),
    "geo_relay_to_leo": (
        "A geostationary-altitude relay looking at a 51.6 degree inclined\n"
        "600 km orbiter. The range is about seventeen times the crossing pair,\n"
        "so the light time is seventeen times longer."
    ),
}

BUILDERS = {
    "leo_crossing": leo_crossing_pair,
    "leo_trailing": leo_trailing_pair,
    "geo_relay_to_leo": geo_to_leo_pair,
}


def main() -> int:
    tee = Tee("regenerate_cases_output.txt")
    header(tee, "regenerate the shipped YAML cases from pointahead.scenario")
    directory = ROOT / "examples" / "cases"
    directory.mkdir(parents=True, exist_ok=True)

    inputs = BudgetInputs(
        sigma_velocity_per_axis_mps=0.05,
        ephemeris_age_s=30.0,
        light_time_iterations=1,
        beam=DEFAULT_BEAM,
        aberration_route="momentum_boost",
        angle_method="closed_form",
        loss_budget_db=1.0,
    )

    for name, builder in BUILDERS.items():
        path = directory / f"{name}.yaml"
        before = path.read_text() if path.exists() else None
        case = Case(name=name, link=builder(), inputs=inputs, source=f"{name}.yaml")
        comment = "\n".join(f"# {line}" for line in HEADERS[name].splitlines())
        text = (
            f"{comment}\n"
            "# Regenerate with: python validation/regenerate_cases.py\n"
            f"{dump_case(case)}"
        )
        path.write_text(text)
        state = (
            "created"
            if before is None
            else ("unchanged" if before == text else "CHANGED")
        )
        tee("")
        tee(f"  {relative(path)}: {state}")
        reloaded = load_case(path)
        tee(f"    label              : {reloaded.name}")
        tee(f"    range              : {reloaded.link.range_m / 1.0e3:.6f} km")
        tee(f"    trajectory model   : {reloaded.link.trajectory_model}")
        tee(f"    frame              : {reloaded.link.frame}")
        tee.require(
            f"{name} round-trips through the loader",
            abs(reloaded.link.range_m - case.link.range_m) < 1.0e-6,
            f"range difference {abs(reloaded.link.range_m - case.link.range_m):.3e} m",
        )
        tee.require(
            f"{name} regenerated identically", state in ("created", "unchanged"), state
        )

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
