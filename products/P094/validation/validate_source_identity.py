"""The one property SPEC §1 rests on: the same file ran in both modes.

The md5 of every committed board program is measured from disk and compared
against the sum the coordinator verified against the copy that executed on the
Jetson Orin Nano. The two programs this package IMPORTS and executes are then
imported and their module ``__file__`` is hashed again at the moment of import,
so the check is of the file that is actually timed and not of a file beside it.

The four sums are pinned as literals in this script AND in
``src/hiltimingdiff/sources.py`` AND in ``tests/test_sources.py``, three
independent copies. If any two disagree, the check fails.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import sources  # noqa: E402

#: The coordinator's verified sums, as literals, independent of the package.
COORDINATOR_MD5 = {
    "campaign.py": "8c137fc2c31c48c2e8f16d1ae1f206d3",
    "staged_exec.py": "0f654894ea7dc3c6ec5d3880e9150459",
    "pat_pipeline.py": "80ba9015b54fa8e95eae96b2c93eb9c2",
    "verify_outputs.py": "ac6b6aa2b486ea7d9a227d1598dc8293",
}


def main() -> int:
    tee = _bootstrap.Tee("validate_source_identity_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- committed board program identity")

    tee.rule("md5 of every committed board program, measured from disk")
    measured = sources.measured_md5()
    for name in sorted(COORDINATOR_MD5):
        path = sources.source_path(name)
        tee(
            f"  {name:<20s} {measured[name]}  "
            f"{path.stat().st_size:>6d} bytes  {_bootstrap.relative(path)}"
        )
    tee("")
    for name in sorted(COORDINATOR_MD5):
        tee.check(
            f"{name} md5 matches the board copy",
            measured[name] == COORDINATOR_MD5[name],
            f"recorded {COORDINATOR_MD5[name]}, measured {measured[name]}",
        )

    tee.rule("the package's own table agrees with the coordinator's list")
    tee.check(
        "sources.SOURCE_MD5 equals the coordinator list",
        sources.SOURCE_MD5 == COORDINATOR_MD5,
        "three independent copies of the four sums: this script, "
        "src/hiltimingdiff/sources.py and tests/test_sources.py",
    )
    tee.check(
        "no source identity failures reported by the package",
        sources.source_identity_failures() == {},
    )

    tee.rule("the file that is TIMED is the file that is hashed")
    pipeline = sources.pipeline_module()
    executor = sources.executor_module()
    for label, module, name in (
        ("pipeline", pipeline, "pat_pipeline.py"),
        ("instrument", executor, "staged_exec.py"),
    ):
        on_disk = Path(module.__file__).resolve()
        tee(f"  {label:<11s} module file : {_bootstrap.relative(on_disk)}")
        tee(f"  {label:<11s} md5 at import: {sources.md5_of_file(on_disk)}")
        tee.check(
            f"{label} module is the committed file at data/{name}",
            on_disk == sources.source_path(name).resolve(),
        )
        tee.check(
            f"{label} md5 at import equals the board sum",
            sources.md5_of_file(on_disk) == COORDINATOR_MD5[name],
        )

    tee.rule("the imported source declares the six stages and five gate branches")
    tee(f"  stages         : {', '.join(sources.stage_names())}")
    tee(f"  reject reasons : {', '.join(sources.reject_reasons())}")
    tee.check(
        "six stages in pipeline order",
        sources.stage_names()
        == ("ingest", "background", "threshold", "centroid", "gate", "track"),
    )
    tee.check(
        "four reject reasons, which with accept make the five gate branches",
        sources.reject_reasons()
        == ("no_candidate", "low_score", "area_out", "border"),
    )

    tee.rule("negative controls: a changed byte must be detected")
    import shutil
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="hiltimingdiff-identity-"))
    copy = scratch / "pat_pipeline.py"
    shutil.copy(sources.source_path("pat_pipeline.py"), copy)
    unchanged_md5 = sources.md5_of_file(copy)
    tee.control(
        "an untouched copy hashes identically",
        unchanged_md5 != COORDINATOR_MD5["pat_pipeline.py"],
        must_fire=False,
        detail=f"copy md5 {unchanged_md5}",
    )
    # Anchored on the TRUE file content, not on an output: one byte is appended
    # to a copy of the real file, so the control cannot pass by accident.
    with open(copy, "ab") as handle:
        handle.write(b"\n")
    changed_md5 = sources.md5_of_file(copy)
    tee.control(
        "a one-byte append changes the md5",
        changed_md5 != unchanged_md5,
        must_fire=True,
        detail=f"appended md5 {changed_md5}",
    )
    shutil.rmtree(scratch, ignore_errors=True)

    tee.rule("summary")
    tee(
        "  The pipeline timed in simulation mode in this container is the file that "
        "ran on the board, byte for byte. A per-stage timing difference is therefore "
        "a property of the platform and not of two implementations."
    )
    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
