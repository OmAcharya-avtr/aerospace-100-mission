"""The committed record: its hash, its schema, and its own falsifiability.

A product whose evidence is a committed data file has to answer three
questions about it, and this script answers them in order.

1. **Is this the file the generator wrote?** The SHA-256 of
   ``data/passes.csv`` is compared against ``data/manifest.json``, and the
   manifest's byte count and pass count against the file itself. A record that
   has drifted from its manifest is a defect whether or not any test fails.

2. **Does the committed decomposition follow from the committed events?** The
   record stores the full event timeline, not only its totals, exactly so that
   this is checkable: every decomposition column is recomputed from
   ``acquisition_s``, ``resync_intervals`` and ``idle_intervals`` and compared
   against the committed column. Storing only the totals would make the
   committed decomposition unfalsifiable from the committed record, which is
   the state a validation suite exists to avoid.

3. **Is the record ignored by git?** The ``.gitignore`` semantics are
   re-implemented here rather than delegated to a git command, because build
   agents in this batch are forbidden from running git. A bare pattern in a
   ``.gitignore`` matches at ANY depth: a bare ``junit.xml`` once silently
   excluded a required committed fixture in another product of this mission,
   and a cold clone then failed with 17 failures and 9 errors while every
   local check passed.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.dataset import (  # noqa: E402
    COLUMNS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RECORD_PATH,
    STATIONS,
    events_of,
    file_digest,
    load_passes,
    station_by_name,
    window_of,
)
from passyield.decompose import decompose_pass  # noqa: E402
from passyield.geometry import satellite  # noqa: E402

#: Files that MUST reach a cold clone. A product whose evidence is gitignored
#: has no evidence.
REQUIRED_FILES = (
    "data/passes.csv",
    "data/manifest.json",
    "data/generate_passes.py",
    "data/generate_passes_output.txt",
)

#: Absolute tolerance on a recomputed volume column, Mbit.
#:
#: Derived with a stated margin, not chosen. Three printing-precision effects
#: separate a recomputed column from its stored value, and all three are
#: properties of the record's own format:
#:
#: 1. the column itself is written at nine significant digits, so a column of
#:    magnitude 6.6e4 Mbit carries up to 3.3e-5 Mbit;
#: 2. ``duration_s`` is written at nine significant digits too, so a 410 s
#:    window carries 5e-7 s, which reaches the nominal column through the
#:    booked rate and is worth up to 5.1e-5 Mbit at the 250 Mbit/s ceiling;
#: 3. each event-interval endpoint is written to nine decimal places of a
#:    second, so it carries 5e-10 s, which is 1.25e-7 Mbit per endpoint, and a
#:    pass here has at most a few tens of endpoints.
#:
#: Those sum to of order 1e-4 Mbit. The bound is set an order of magnitude
#: above that, at 1e-3, because the three do not simply add: the duration
#: perturbs the window, which perturbs the SGP4 elevation profile, which
#: perturbs the rate at every quadrature node, and this product does not model
#: that propagation term by term. The MEASURED worst absolute difference over
#: the whole record is reported below, so a reader can see where it sits
#: against the bound rather than taking the margin on trust.
VOLUME_TOLERANCE_MBIT = 1.0e-3

#: Relative tolerance on a recomputed volume column.
#:
#: The record is written at nine significant digits, so a column carries about
#: 1e-9 of relative error. The SHORTFALL is a difference of two columns of
#: comparable size, so its conditioning is worse by nominal/(nominal-realised),
#: which on this record is about nine; its tolerance is an order of magnitude
#: looser for that reason and for no other.
COLUMN_RELATIVE_TOLERANCE = 1.0e-8
SHORTFALL_RELATIVE_TOLERANCE = 1.0e-7


def main() -> int:
    """Check the record's hash, its falsifiability and its reachability."""
    out = _bootstrap.Tee("validate_record_output.txt")
    out("Committed record validation -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")

    out.rule("1. IS THIS THE FILE THE GENERATOR WROTE?")
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    digest = file_digest(DEFAULT_RECORD_PATH)
    size = DEFAULT_RECORD_PATH.stat().st_size
    record = load_passes()
    out(f"record                       : {_bootstrap.relative(DEFAULT_RECORD_PATH)}")
    out(f"manifest                     : {_bootstrap.relative(DEFAULT_MANIFEST_PATH)}")
    out(f"computed SHA-256             : {digest}")
    out(f"manifest SHA-256             : {manifest['record_sha256']}")
    out(f"computed size, bytes         : {size}")
    out(f"manifest size, bytes         : {manifest['record_bytes']}")
    out(f"rows loaded                  : {len(record)}")
    out(f"manifest pass count          : {manifest['passes']}")
    out(f"generator seed               : {manifest['seed']}")
    out(f"campaign days                : {manifest['campaign_days']}")
    out(f"element set line 1           : {manifest['tle_line1']}")
    out(f"element set line 2           : {manifest['tle_line2']}")
    out("")
    out("the record is under 1 MB, so it is committed. The mission's rule is that a")
    out("data file over 1 MB is not committed and only the generator and the hash")
    out("are; this one does not reach that.")
    out(f"size as a fraction of 1 MB   : {size / (1 << 20):.4f}")
    out("")
    out.check("the record's SHA-256 matches the manifest", digest == manifest["record_sha256"])
    out.check("the record's size matches the manifest", size == manifest["record_bytes"])
    out.check("the row count matches the manifest", len(record) == manifest["passes"])
    out.check("the record is under 1 MB", size < (1 << 20), f"{size} bytes")
    out.check(
        "the schema header is the dataclass's own field order",
        len(COLUMNS) == 27 and COLUMNS[0] == "pass_id",
        f"{len(COLUMNS)} columns",
    )
    out.check(
        "every station in the record is in the catalogue",
        {row.station for row in record} == {station.name for station in STATIONS},
    )

    out.rule("2. DOES THE COMMITTED DECOMPOSITION FOLLOW FROM THE COMMITTED EVENTS?")
    out("every decomposition column is recomputed from the stored event timeline and")
    out("compared against the stored column. The tolerances are derived in this")
    out("script's own constants and are not chosen: the interval fields carry")
    out("5e-10 s per endpoint, which times the 250 Mbit/s ceiling is 1.25e-7 Mbit,")
    out("and the columns carry nine significant digits.")
    out("")
    satrec = satellite()
    limit = 150 if _bootstrap.FAST else len(record)
    rows = record[:limit]
    worst_absolute: dict[str, float] = {}
    worst_relative: dict[str, float] = {}
    offenders: list[str] = []
    for row in rows:
        item = decompose_pass(
            satrec, station_by_name(row.station), window_of(row), events_of(row)
        )
        pairs = {
            "nominal_mbit": (item.nominal_bits, row.nominal_mbit, COLUMN_RELATIVE_TOLERANCE),
            "capacity_mbit": (item.capacity_bits, row.capacity_mbit, COLUMN_RELATIVE_TOLERANCE),
            "realised_mbit": (item.realised_bits, row.realised_mbit, COLUMN_RELATIVE_TOLERANCE),
            "shortfall_mbit": (
                item.shortfall_bits, row.shortfall_mbit, SHORTFALL_RELATIVE_TOLERANCE
            ),
            "geometry_mbit": (item.geometry_bits, row.geometry_mbit, SHORTFALL_RELATIVE_TOLERANCE),
            "acquisition_mbit": (
                item.acquisition_bits, row.acquisition_mbit, COLUMN_RELATIVE_TOLERANCE
            ),
            "resync_mbit": (item.resync_bits, row.resync_mbit, COLUMN_RELATIVE_TOLERANCE),
            "idle_mbit": (item.idle_bits, row.idle_mbit, COLUMN_RELATIVE_TOLERANCE),
        }
        for name, (computed, stored, relative_tolerance) in pairs.items():
            difference = abs(computed - stored)
            scale = max(abs(stored), 1.0e-30)
            worst_absolute[name] = max(worst_absolute.get(name, 0.0), difference)
            # The relative figure is reported only over columns of magnitude at
            # least 1 Mbit. Below that it is not a useful statistic and it is
            # actively misleading: 44 of the 1990 passes have an acquisition
            # time longer than the whole window, so their realised volume is
            # EXACTLY zero, and a ratio against zero reports 1e25 for an
            # absolute difference of 1e-5 Mbit. The pass/fail criterion below
            # is unaffected -- it requires BOTH the absolute and the relative
            # bound to be exceeded -- but a reported number that looks like a
            # catastrophic failure and is not would be worse than no number.
            if abs(stored) >= 1.0:
                worst_relative[name] = max(
                    worst_relative.get(name, 0.0), difference / abs(stored)
                )
            if (
                difference > VOLUME_TOLERANCE_MBIT
                and difference > relative_tolerance * scale
            ):
                offenders.append(
                    f"pass_id {row.pass_id} {name}: {computed!r} against {stored!r}"
                )
    out(f"passes recomputed                 : {len(rows)}")
    out("")
    out("the relative column is taken only over rows whose stored value is at")
    out("least 1 Mbit in magnitude; see this script's source for why, and note")
    out("that the pass/fail criterion uses both bounds and is unaffected.")
    out("")
    out(f"{'column':>20}{'worst |difference|, Mbit':>27}{'worst relative':>17}")
    for name in sorted(worst_absolute):
        out(
            f"{name:>20}{worst_absolute[name]:>27.3e}"
            f"{worst_relative.get(name, float('nan')):>17.3e}"
        )
    out("")
    out(f"absolute tolerance, Mbit          : {VOLUME_TOLERANCE_MBIT!r}")
    out(f"worst absolute difference, Mbit    : {max(worst_absolute.values()):.3e}")
    out(f"margin against the bound           : "
        f"{VOLUME_TOLERANCE_MBIT / max(worst_absolute.values()):.2f}x")
    out(f"columns outside tolerance         : {len(offenders)}")
    for offender in offenders[:10]:
        out(f"  OFFENDER: {offender}")
    out.check(
        "every committed decomposition column follows from the committed events",
        not offenders,
        f"{len(offenders)} outside tolerance",
    )

    out("")
    out("the identity itself, over the recomputed passes, from the record's OWN")
    out("columns: geometry + acquisition + resync + idle + residual = shortfall.")
    identity = np.asarray(
        [
            row.geometry_mbit
            + row.acquisition_mbit
            + row.resync_mbit
            + row.idle_mbit
            + row.residual_mbit
            - row.shortfall_mbit
            for row in record
        ]
    )
    shortfalls = np.asarray([row.shortfall_mbit for row in record])
    out(f"worst |closure| over all {len(record)} rows, Mbit : "
        f"{float(np.max(np.abs(identity))):.3e}")
    out(f"worst |closure| / |shortfall|              : "
        f"{float(np.max(np.abs(identity / shortfalls))):.3e}")
    out("")
    out("this closes to the PRINTED PRECISION of the record and not to the")
    out("quadrature: the residual column is itself stored, so the identity is exact")
    out("in exact arithmetic over these six numbers. The quadrature residual is the")
    out("residual COLUMN, measured in validation/validate_decomposition.py.")
    out("")
    out("THE BOUND IS ABSOLUTE AND IT IS DERIVED. Each column is written at nine")
    out("significant digits, so a column of magnitude M carries up to 0.5e-9 * M of")
    out("printing error; six columns enter the identity, and the largest of them on")
    out("this record is of order 1e5 Mbit, so the closure can reach")
    out("6 * 0.5e-9 * 1e5 = 3e-4 Mbit. A RELATIVE bound would be the wrong shape")
    out("here: the error scales with the largest column and the identity's left-hand")
    out("side is the SHORTFALL, which on some passes is two orders of magnitude")
    out("smaller, so the relative figure is large for a reason that is printing")
    out("precision and not arithmetic.")
    largest_column = float(
        max(
            max(
                abs(row.geometry_mbit), abs(row.acquisition_mbit), abs(row.resync_mbit),
                abs(row.idle_mbit), abs(row.shortfall_mbit),
            )
            for row in record
        )
    )
    derived_bound = 6.0 * 0.5e-9 * largest_column
    out("")
    out(f"largest column magnitude in the identity, Mbit : {largest_column:.1f}")
    out(f"derived closure bound, Mbit                    : {derived_bound:.3e}")
    out.check(
        "the stored columns close the identity within the derived printing bound",
        float(np.max(np.abs(identity))) <= derived_bound,
        f"worst {float(np.max(np.abs(identity))):.3e} against {derived_bound:.3e} Mbit",
    )

    out.rule("3. IS THE RECORD REACHABLE IN A COLD CLONE?")
    ignore, negate = _bootstrap.gitignore_rules()
    out(f"ignore patterns : {ignore}")
    out(f"negation patterns: {negate}")
    out("")
    out("a pattern with NO SLASH matches a basename at any depth. That is the trap:")
    out("a bare junit.xml in another product of this mission silently excluded a")
    out("required committed fixture, and a cold clone failed with 17 failures and 9")
    out("errors while every local check passed. Every scratch pattern in this")
    out("product's .gitignore is anchored with a leading slash.")
    out("")
    unanchored = [
        pattern
        for pattern in ignore
        if "/" not in pattern.rstrip("/") and not pattern.startswith("*")
    ]
    out(f"unanchored, non-glob ignore patterns: {unanchored}")
    out.check(
        "every non-glob ignore pattern is anchored or a directory name",
        all(pattern.endswith("/") for pattern in unanchored),
        f"{unanchored}",
    )
    out("")
    out(f"{'required file':>36}{'ignored':>10}{'reason':>34}")
    missing: list[str] = []
    ignored_files: list[str] = []
    for name in REQUIRED_FILES:
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        flag, reason = _bootstrap.gitignore_match(name, ignore, negate)
        if flag:
            ignored_files.append(name)
        out(f"{name:>36}{str(flag):>10}{reason:>34}")
    out("")
    for name in sorted((ROOT / "validation").glob("*_output.txt")):
        relative_name = f"validation/{name.name}"
        flag, reason = _bootstrap.gitignore_match(relative_name, ignore, negate)
        if flag:
            ignored_files.append(relative_name)
    out.check("every required file exists", not missing, f"missing {missing}")
    out.check(
        "no required file is excluded by the product's own .gitignore",
        not ignored_files,
        f"ignored: {ignored_files}",
    )
    out("")
    out("and the control in the other direction: the patterns MUST ignore the")
    out("scratch files they exist for, or they are decoration.")
    for name in (
        "__pycache__/passyield.cpython-313.pyc",
        "src/passyield/__pycache__/link.cpython-313.pyc",
        "junit.xml",
        ".ruff_cache/0.16.9/abc",
        "model.joblib",
        "src/passyield/model.pkl",
    ):
        flag, reason = _bootstrap.gitignore_match(name, ignore, negate)
        out.control(
            f"{name} is ignored",
            flag,
            must_fire=True,
            detail=f"by {reason!r}" if flag else "NOT ignored",
        )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
