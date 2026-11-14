"""The loader contract, checked against a SYNTHETIC campaign written on disk.

INPUT ORIGIN: SYNTHETIC. Every number below comes from the committed generator
in ``src/powerenvelope/synthetic.py`` and is a check of the loader's behaviour,
not a measurement of any board. The measured campaign is checked by
``validate_measured_*.py``, which refuse and exit non-zero when it is absent.

What is checked here:

1. a complete campaign loads, with every planned cell and all three traces;
2. the power column is the product of the two columns it is supposed to be;
3. the acceptance rate is RECONSTRUCTIBLE by counting and is never asserted;
4. every malformed header is refused, in each of the three files, including a
   header with the right names in the wrong order;
5. every structural corruption is refused: a ragged row, a non-numeric field, a
   decreasing timestamp, an ``accepted`` value that is not 0 or 1, an
   ``accepted`` value disagreeing with its reason, an inverted inference
   interval, an undeclared reject reason, and a power column that was not
   written as ``V * I / 1000``;
6. an absent directory, an absent manifest, an absent run record and an absent
   required cell are each refused by name;
7. a directory without the completion marker is flagged incomplete.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.schema import (  # noqa: E402
    DECISION_COLUMNS,
    MONITOR_COLUMNS,
    POWER_COLUMNS,
    SchemaError,
)
from powerenvelope.synthetic import write_campaign  # noqa: E402
from powerenvelope.trace import (  # noqa: E402
    PLANNED_CELLS,
    TraceDataAbsentError,
    campaign_is_complete,
    load_campaign,
    load_decisions,
    load_monitor,
    load_power,
)


def _refuses(call, exception=SchemaError) -> tuple[bool, str]:
    try:
        call()
    except exception as error:
        return True, str(error).splitlines()[0][:110]
    except Exception as error:  # noqa: BLE001 -- a different error is still a miss
        return False, f"raised {type(error).__name__} instead: {error}"
    return False, "did not raise"


def _edit(path: Path, line_index: int, field: int, value: str) -> None:
    lines = path.read_text().splitlines()
    fields = lines[line_index].split(",")
    fields[field] = value
    lines[line_index] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")


def _first_accept(path: Path) -> int:
    for index, line in enumerate(path.read_text().splitlines()[1:], start=1):
        if line.endswith(",accept"):
            return index
    raise RuntimeError("the synthetic fixture has no accepted decision")


def main() -> int:
    out = _bootstrap.Tee("validate_loader_contract_output.txt")
    out.rule("LOADER CONTRACT -- what is accepted, and every way it refuses")
    out("INPUT ORIGIN: SYNTHETIC, from src/powerenvelope/synthetic.py. Not a measurement.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.measurement_model())

    scratch = Path(tempfile.mkdtemp(prefix="powerenvelope-loader-"))
    directory = write_campaign(scratch / "good", seed=_bootstrap.SEED_FIXTURE)

    out.rule("1  a complete synthetic campaign loads")
    campaign = load_campaign(directory, directory_label="synthetic")
    out(f"  cells loaded: {len(campaign)}")
    out.check(
        "every planned cell is present",
        set(campaign.cells) == set(PLANNED_CELLS),
        f"{len(campaign)} of {len(PLANNED_CELLS)}",
    )
    out.check(
        "the floor cell has no decisions trace",
        campaign.require("floor").decisions is None,
        "it ran no pipeline, so it has no useful decisions",
    )
    out.check(
        "every pipeline cell has all three traces",
        all(
            len(cell.power) > 0 and len(cell.monitor) > 0 and cell.decisions is not None
            for cell in campaign.pipeline_cells()
        ),
        f"{len(campaign.pipeline_cells())} pipeline cell(s)",
    )
    out.check(
        "the completion marker is detected",
        campaign_is_complete(directory) and campaign.complete,
        "CAMPAIGN_COMPLETE present",
    )

    out.rule("2  the power column is the product of the two columns it must be")
    worst = 0.0
    for cell in campaign.cells.values():
        residual = np.abs(
            cell.power.vdd_in_mv * cell.power.vdd_in_ma / 1000.0 - cell.power.vdd_in_mw
        )
        worst = max(worst, float(residual.max()))
    out.check(
        "vdd_in_mw equals vdd_in_mv * vdd_in_ma / 1000 within the %.3f rounding",
        worst <= 1e-3,
        f"largest residual {worst:.3e} mW over {len(campaign)} cell(s)",
    )

    out.rule("3  the acceptance rate is reconstructed by counting")
    cell = campaign.require(_bootstrap.HEADLINE_CELL)
    decisions = cell.require_decisions()
    counts = decisions.reason_counts()
    out(f"  reject reasons: {counts}")
    out.check(
        "the reason histogram sums to the inference count",
        sum(counts.values()) == decisions.inferences,
        f"{sum(counts.values())} == {decisions.inferences}",
    )
    out.check(
        "accepted equals the count of rows whose reason is 'accept'",
        counts.get("accept", 0) == decisions.accepted_count,
        f"{counts.get('accept', 0)} == {decisions.accepted_count}",
    )
    out.check(
        "the rate is the quotient of the two counted numbers",
        abs(decisions.acceptance_rate - decisions.accepted_count / decisions.inferences)
        <= 0.0,
        f"a = {decisions.acceptance_rate:.6f}",
    )

    out.rule("4  every malformed header is refused")
    header_cases: list[tuple[str, tuple[str, ...], object, str]] = []
    for label, columns, loader, filename in (
        ("power", POWER_COLUMNS, load_power, "power_w64_d050.csv"),
        ("monitor", MONITOR_COLUMNS, load_monitor, "monitor_w64_d050.csv"),
        ("decisions", DECISION_COLUMNS, load_decisions, "decisions_w64_d050.csv"),
    ):
        header_cases.append((f"{label}: a renamed column", columns, loader, filename))
        header_cases.append((f"{label}: a missing column", columns, loader, filename))
        header_cases.append((f"{label}: an extra column", columns, loader, filename))
        header_cases.append((f"{label}: the right names reordered", columns, loader, filename))
    for name, columns, loader, filename in header_cases:
        target = write_campaign(
            scratch / f"h{abs(hash(name)) % 10**8}", seed=_bootstrap.SEED_FIXTURE
        ) / filename
        body = target.read_text().splitlines()[1:]
        if name.endswith("a renamed column"):
            header = (*columns[:-1], "renamed")
        elif name.endswith("a missing column"):
            header = columns[:-1]
        elif name.endswith("an extra column"):
            header = (*columns, "surprise")
        else:
            header = (columns[1], columns[0], *columns[2:])
        target.write_text(",".join(header) + "\n" + "\n".join(body) + "\n")
        fired, detail = _refuses(lambda target=target, loader=loader: loader(target))
        out.control(name, fired, must_fire=True, detail=detail)

    out.rule("5  every structural corruption is refused")
    corruptions: list[tuple[str, str, int, int, str, object]] = [
        ("a non-numeric power field", "power_w64_d050.csv", 3, 2, "nope", load_power),
        ("a decreasing timestamp", "power_w64_d050.csv", 5, 0, "1.0", load_power),
        ("an accepted value of 2", "decisions_w64_d050.csv", 1, 8, "2", load_decisions),
        ("an undeclared reject reason", "decisions_w64_d050.csv", 1, 9, "exploded", load_decisions),
    ]
    for name, filename, line_index, field, value, loader in corruptions:
        target = write_campaign(
            scratch / f"c{abs(hash(name)) % 10**8}", seed=_bootstrap.SEED_FIXTURE
        ) / filename
        _edit(target, line_index, field, value)
        fired, detail = _refuses(lambda target=target, loader=loader: loader(target))
        out.control(name, fired, must_fire=True, detail=detail)

    target = write_campaign(scratch / "cpow", seed=_bootstrap.SEED_FIXTURE) / "power_w64_d050.csv"
    original = float(target.read_text().splitlines()[4].split(",")[7])
    _edit(target, 4, 7, "%.3f" % (original + 1.0))
    fired, detail = _refuses(lambda: load_power(target))
    out.control(
        "a power column that is not V*I/1000", fired, must_fire=True, detail=detail
    )

    target = (
        write_campaign(scratch / "cacc", seed=_bootstrap.SEED_FIXTURE)
        / "decisions_w64_d050.csv"
    )
    _edit(target, _first_accept(target), 8, "0")
    fired, detail = _refuses(lambda: load_decisions(target))
    out.control(
        "accepted disagreeing with its reason", fired, must_fire=True, detail=detail
    )

    target = (
        write_campaign(scratch / "cint", seed=_bootstrap.SEED_FIXTURE)
        / "decisions_w64_d050.csv"
    )
    start = float(target.read_text().splitlines()[1].split(",")[0])
    _edit(target, 1, 1, "%.6f" % (start - 1.0))
    fired, detail = _refuses(lambda: load_decisions(target))
    out.control("an inverted inference interval", fired, must_fire=True, detail=detail)

    target = write_campaign(scratch / "crag", seed=_bootstrap.SEED_FIXTURE) / "power_w64_d050.csv"
    lines = target.read_text().splitlines()
    lines[3] = lines[3] + ",99"
    target.write_text("\n".join(lines) + "\n")
    fired, detail = _refuses(lambda: load_power(target))
    out.control("a ragged row", fired, must_fire=True, detail=detail)

    target = write_campaign(scratch / "chdr", seed=_bootstrap.SEED_FIXTURE) / "power_w64_d050.csv"
    target.write_text(target.read_text().splitlines()[0] + "\n")
    fired, detail = _refuses(lambda: load_power(target))
    out.control("a file with a header and no rows", fired, must_fire=True, detail=detail)

    out.rule("6  an absent directory, manifest, run record or cell is refused by name")
    fired, detail = _refuses(
        lambda: load_campaign(scratch / "not-there"), TraceDataAbsentError
    )
    out.control("an absent directory", fired, must_fire=True, detail=detail)

    missing_manifest = write_campaign(scratch / "nomanifest", seed=_bootstrap.SEED_FIXTURE)
    (missing_manifest / "manifest.json").unlink()
    fired, detail = _refuses(
        lambda: load_campaign(missing_manifest), TraceDataAbsentError
    )
    out.control("an absent manifest", fired, must_fire=True, detail=detail)

    no_record = write_campaign(scratch / "norecord", seed=_bootstrap.SEED_FIXTURE)
    manifest = json.loads((no_record / "manifest.json").read_text())
    manifest["runs"] = [run for run in manifest["runs"] if run["cell"] != "w48_d050"]
    (no_record / "manifest.json").write_text(json.dumps(manifest))
    fired, detail = _refuses(lambda: load_campaign(no_record), TraceDataAbsentError)
    out.control("a manifest with no run record for a cell", fired, must_fire=True, detail=detail)

    fired, detail = _refuses(
        lambda: load_campaign(directory, require=("w64_d999",)), TraceDataAbsentError
    )
    out.control("a required cell that is absent", fired, must_fire=True, detail=detail)

    out.rule("7  a complete campaign must NOT be refused")
    fired, detail = _refuses(
        lambda: load_campaign(directory, require=PLANNED_CELLS), TraceDataAbsentError
    )
    out.control(
        "the unmodified synthetic campaign", fired, must_fire=False, detail=detail
    )
    partial = write_campaign(scratch / "partial", seed=_bootstrap.SEED_FIXTURE, complete=False)
    out.check(
        "a directory without the marker is flagged incomplete",
        not campaign_is_complete(partial) and not load_campaign(partial).complete,
        "the campaign writes CAMPAIGN_COMPLETE last of all",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
