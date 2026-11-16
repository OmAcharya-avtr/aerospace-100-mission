"""What the loader REFUSES, exercised hermetically against temporary directories.

A loader that quietly produced a plausible number when the campaign was absent
would make a simulated figure indistinguishable from a measurement. There is no
synthetic fallback in this product, by design, so every refusal path has to be
exercised rather than assumed: each case below builds a deliberately broken
copy of the committed campaign in a temporary directory outside the repository
and requires the named refusal.

``HILTIMINGDIFF_TRACES`` moves WHICH directory is required. It never makes a
measurement optional and it never selects a substitute, and that distinction is
checked here too.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import load_campaign  # noqa: E402
from hiltimingdiff.sources import SourceIdentityError, load_board_module  # noqa: E402
from hiltimingdiff.traces import (  # noqa: E402
    REQUIRED_FILES,
    TRACES_ENVIRONMENT_VARIABLE,
    TraceDataAbsentError,
    campaign_is_complete,
)


def _copy(campaign, scratch: Path, name: str) -> Path:
    target = scratch / name
    shutil.copytree(campaign.directory, target)
    return target


def main() -> int:
    tee = _bootstrap.Tee("validate_loader_contract_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- loader refusal contract")

    campaign = load_campaign()
    scratch = Path(tempfile.mkdtemp(prefix="hiltimingdiff-loader-"))
    tee.rule("the happy path, so the refusals mean something")
    tee(f"  loaded {campaign.label}/ with {len(campaign.runs)} cells")
    tee.check("the committed campaign loads", len(campaign.runs) == 9)
    tee.check("the completion token is present", campaign_is_complete(campaign.directory))

    tee.rule("refusal: the directory does not exist")
    try:
        load_campaign(scratch / "absent")
    except TraceDataAbsentError as error:
        message = str(error)
        tee(f"  {message[:200]}...")
        tee.check("refused with MEASURED CAMPAIGN ABSENT", "MEASURED CAMPAIGN ABSENT" in message)
        tee.check(
            "the refusal names every required file",
            all(name in message for name in REQUIRED_FILES),
        )
        tee.check(
            "the refusal says nothing was substituted",
            "no synthetic" in message or "nothing was substituted" in message.lower(),
        )
    else:
        tee.check("refused when the directory does not exist", False, "it did not")

    tee.rule("refusal: the directory exists and is empty")
    empty = scratch / "empty"
    empty.mkdir()
    try:
        load_campaign(empty)
    except TraceDataAbsentError as error:
        tee(f"  {str(error)[:200]}...")
        tee.check("refused with MEASURED CAMPAIGN INCOMPLETE", "INCOMPLETE" in str(error))
    else:
        tee.check("refused when the directory is empty", False, "it did not")

    tee.rule("refusal: one required file at a time is removed")
    for name in REQUIRED_FILES:
        target = _copy(campaign, scratch, f"missing-{name.replace('.', '-')}")
        (target / name).unlink()
        try:
            load_campaign(target)
        except TraceDataAbsentError as error:
            tee.check(
                f"refused when {name} is absent, and named it",
                name in str(error),
                str(error).splitlines()[0][:120],
            )
        else:
            tee.check(f"refused when {name} is absent", False, "it did not")

    tee.rule("refusal: the campaign did not complete")
    truncated = _copy(campaign, scratch, "no-token")
    (truncated / "campaign.log").write_text("[FULL] started and was interrupted\n")
    try:
        load_campaign(truncated)
    except TraceDataAbsentError as error:
        tee.check("refused with DID NOT COMPLETE", "DID NOT COMPLETE" in str(error))
        tee.check(
            "the refusal names the token the campaign writes last",
            "CAMPAIGN_COMPLETE" in str(error),
        )
    else:
        tee.check("refused when the completion token is absent", False, "it did not")

    tee.rule("refusal: a cell is missing from runs.csv")
    without_cell = _copy(campaign, scratch, "missing-cell")
    runs = pd.read_csv(without_cell / "runs.csv")
    runs.loc[runs["cell"] != "w96_dense"].to_csv(without_cell / "runs.csv", index=False)
    try:
        load_campaign(without_cell, require_rows=False)
    except TraceDataAbsentError as error:
        tee.check("refused with MISSING CELLS", "MISSING CELLS" in str(error))
        tee.check("the refusal names the absent cell", "w96_dense" in str(error))
    else:
        tee.check("refused when a cell is missing", False, "it did not")

    tee.rule("refusal: a raw array is missing")
    without_raw = _copy(campaign, scratch, "missing-raw")
    (without_raw / "raw" / "w64_dense.npy").unlink()
    try:
        load_campaign(without_raw, require_rows=False)
    except TraceDataAbsentError as error:
        tee.check("refused with MISSING RAW ARRAY", "MISSING RAW ARRAY" in str(error))
        tee.check("the refusal names the absent array", "w64_dense" in str(error))
    else:
        tee.check("refused when a raw array is missing", False, "it did not")

    tee.rule("refusal: a table is silently truncated")
    for name, expected in (("stages_pass.csv", 3240), ("stages.csv", 270000)):
        shortened = _copy(campaign, scratch, f"short-{name.replace('.', '-')}")
        table = pd.read_csv(shortened / name)
        table.iloc[: len(table) // 2].to_csv(shortened / name, index=False)
        try:
            load_campaign(shortened)
        except TraceDataAbsentError as error:
            tee.check(
                f"refused when {name} is halved, naming both counts",
                "ROW COUNTS CHANGED" in str(error) and str(expected) in str(error),
                str(error).splitlines()[0][:140],
            )
        else:
            tee.check(f"refused when {name} is halved", False, "it did not")

    tee.rule("the environment variable moves WHICH directory, never WHETHER one is needed")
    import os

    previous = os.environ.get(TRACES_ENVIRONMENT_VARIABLE)
    os.environ[TRACES_ENVIRONMENT_VARIABLE] = str(scratch / "elsewhere")
    try:
        load_campaign()
    except TraceDataAbsentError as error:
        tee.check(
            "with the variable set to an absent directory, the loader still refuses",
            "elsewhere" in str(error),
            "the variable did not make the measurement optional",
        )
    else:
        tee.check("the variable does not make a measurement optional", False, "it did")
    finally:
        if previous is None:
            os.environ.pop(TRACES_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[TRACES_ENVIRONMENT_VARIABLE] = previous

    tee.rule("the source-identity refusal")
    try:
        load_board_module("campaign")
    except SourceIdentityError as error:
        tee.check(
            "the campaign driver is hashed but not importable",
            "not one of the importable" in str(error),
        )
    else:
        tee.check("the campaign driver is refused for import", False, "it was imported")

    tee.rule("negative controls")
    intact = _copy(campaign, scratch, "intact")
    try:
        load_campaign(intact)
    except TraceDataAbsentError as error:
        tee.control(
            "an untouched copy is refused",
            True,
            must_fire=False,
            detail=f"refused with {str(error).splitlines()[0][:80]}",
        )
    else:
        tee.control(
            "an untouched copy is refused",
            False,
            must_fire=False,
            detail="loaded, as it must",
        )
    one_row_short = _copy(campaign, scratch, "one-row-short")
    table = pd.read_csv(one_row_short / "stages_pass.csv")
    table.iloc[:-1].to_csv(one_row_short / "stages_pass.csv", index=False)
    try:
        load_campaign(one_row_short)
    except TraceDataAbsentError:
        tee.control(
            "removing ONE row of 3240 is detected",
            True,
            must_fire=True,
            detail="the row-count pins are literals, so a single missing row is a "
            "refusal and not a slightly different median",
        )
    else:
        tee.control("removing ONE row of 3240 is detected", False, must_fire=True)

    shutil.rmtree(scratch, ignore_errors=True)
    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
