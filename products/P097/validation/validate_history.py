"""The committed history: its digest, its schema, and what the loader rejects.

Every number in ``README.md``, ``validation/VALIDATION.md`` and
``MODEL_CARD.md`` is a function of ``data/run_history.json``, so this script
pins its SHA-256 and reports its contents. If the digest moves, every document
in this repository is stale and the whole validation set fails here first.

This script does NOT regenerate the history. The generator is
``data/generate_history.py``, which lives outside ``validation/`` deliberately:
the release gate runs ``sorted(glob('validation/*.py'))``, so a generator there
would rewrite the evidence an alphabetically adjacent script pins, and the
gate's own ordering would collide them.

The loader's rejections are exercised here as well as in ``tests/``, because a
reader of the evidence should be able to see that the schema is enforced
without reading the test suite.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.history import (  # noqa: E402
    SCHEMA,
    history_from_mapping,
    history_sha256,
    load_history,
)

#: Rejections exercised here, as ``(label, mutation)``. Each mutation edits a
#: parsed copy of the committed history, so every case is applied to real data.
REJECTIONS = (
    ("an unknown schema string", lambda payload: payload.update({"schema": "other/9"})),
    ("a duration unit other than millisecond",
     lambda payload: payload.update({"duration_unit": "second"})),
    ("a retry budget below 1", lambda payload: payload.update({"retry_budget": 0})),
    ("a negative duration",
     lambda payload: payload["attempts"]["duration_ms"].__setitem__(0, -1)),
    ("a fractional duration",
     lambda payload: payload["attempts"]["duration_ms"].__setitem__(0, 1.5)),
    ("a failed flag that is not 0 or 1",
     lambda payload: payload["attempts"]["failed"].__setitem__(0, 7)),
    ("a session index outside the session table",
     lambda payload: payload["attempts"]["session"].__setitem__(0, 10_000)),
    ("a test index outside the test table",
     lambda payload: payload["attempts"]["test"].__setitem__(0, 10_000)),
    ("an attempt number that is not 1-based",
     lambda payload: payload["attempts"]["attempt"].__setitem__(0, 0)),
    ("attempt arrays of unequal length",
     lambda payload: payload["attempts"].__setitem__(
         "duration_ms", payload["attempts"]["duration_ms"][:-1])),
    ("a duplicate test name",
     lambda payload: payload["tests"][1].__setitem__(
         "name", payload["tests"][0]["name"])),
)


def main() -> int:
    """Pin the digest, report the contents, and exercise the rejections."""
    out = _bootstrap.Tee("validate_history_output.txt")
    out("The committed recorded run history -- P097 flakerank")
    out("")
    out("Every number in README.md, validation/VALIDATION.md and MODEL_CARD.md is")
    out("a function of this one file. Its SHA-256 is pinned below, so a silent")
    out("edit makes the whole validation set fail here first.")
    out("")
    out(f"fast path (FLAKERANK_FAST=1): {_bootstrap.FAST}")

    path = ROOT / _bootstrap.HISTORY
    out.rule("1. THE FILE")
    out("")
    out(f"path   : {_bootstrap.relative(path)}")
    if not path.is_file():
        out.unavailable(
            "the committed history is present",
            "data/run_history.json is absent; run data/generate_history.py",
        )
        return out.finish()
    digest = history_sha256(path)
    size = path.stat().st_size
    out(f"bytes  : {size}")
    out(f"sha256 : {digest}")
    out(f"pinned : {_bootstrap.HISTORY_SHA256}")
    out.check(
        "the committed history has the pinned SHA-256",
        digest == _bootstrap.HISTORY_SHA256,
        f"measured {digest}",
    )
    out.check(
        "the committed history is under the mission's 1 MB limit for data",
        size < 1_000_000,
        f"{size} bytes",
    )

    history = load_history(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out.rule("2. WHAT IS IN IT")
    out("")
    out(f"schema                    : {payload['schema']}")
    out(f"generator                 : {payload['generator']}")
    out(f"declared generator seed   : {payload['seed']}")
    out(f"retry budget              : {history.retry_budget} attempts per invocation")
    out(f"duration unit             : {payload['duration_unit']}")
    out(f"regime change at session  : {payload['regime_change_session']}")
    out(f"tests                     : {history.n_tests}")
    out(f"sessions                  : {history.n_sessions}")
    out(f"attempts                  : {history.n_attempts}")
    failures = int(np.count_nonzero(history.attempt_failed))
    out(f"failed attempts           : {failures}")
    out(f"pooled per-attempt rate   : {failures / history.n_attempts!r}")
    total_ms = int(history.attempt_duration_ms.sum())
    failed_ms = int(history.attempt_duration_ms[history.attempt_failed.astype(bool)].sum())
    out(f"total attempt time, ms    : {total_ms}")
    out(f"wasted (failed) time, ms  : {failed_ms}")
    out(f"wasted fraction of total  : {failed_ms / total_ms!r}")
    out(f"shortest attempt, ms      : {int(history.attempt_duration_ms.min())}")
    out(f"longest attempt, ms       : {int(history.attempt_duration_ms.max())}")
    out("")
    out("Durations are exact integer milliseconds. The largest total is")
    out(f"{total_ms} ms, far below 2**53 = {2 ** 53}, so every partial sum of these")
    out("integers is exactly representable in double precision and the input")
    out("contributes no representation error to any accumulation.")
    out.check(
        "every duration is a non-negative integer number of milliseconds",
        bool(np.issubdtype(history.attempt_duration_ms.dtype, np.integer))
        and int(history.attempt_duration_ms.min()) >= 0,
        f"dtype {history.attempt_duration_ms.dtype}, "
        f"minimum {int(history.attempt_duration_ms.min())}",
    )
    out.check(
        "the duration total is exactly representable in double precision",
        total_ms < 2**53,
        f"{total_ms} < {2 ** 53}",
    )
    out.check("the schema string is the one the loader declares", payload["schema"] == SCHEMA)

    out.rule("3. THE TIER STRUCTURE AND THE MOTIVATING CASE")
    out("")
    out("The generator's ground-truth parameters are recorded in the file for")
    out("negative controls only. No baseline and no model reads them.")
    out("")
    tiers: dict[str, int] = {}
    for item in payload["tests"]:
        tiers[item["tier"]] = tiers.get(item["tier"], 0) + 1
    for tier, count in sorted(tiers.items()):
        out(f"  {tier:>9}: {count:>3} tests")
    out("")
    index = history.test_names.index("soak/test_validation_probe_1_in_44")
    out("the pinned case, named for the failure this mission actually shipped:")
    out(f"  name              : {history.test_names[index]}")
    out(f"  true p            : {float(history.true_failure_probability[index])!r}")
    out(f"  1/44 exactly      : {1.0 / 44.0!r}")
    summary = next(
        item for item in history.summaries()
        if item.name == "soak/test_validation_probe_1_in_44"
    )
    out(f"  measured p        : {summary.failure_probability!r}")
    out(f"  attempts          : {summary.attempts}")
    out(f"  failed attempts   : {summary.failed_attempts}")
    out(f"  mean attempt, s   : {summary.mean_attempt_seconds!r}")
    out(f"  wasted seconds    : {summary.wasted_seconds!r}")
    out.check(
        "the pinned case's true p is 1/44",
        abs(float(history.true_failure_probability[index]) - 1.0 / 44.0) <= 1e-15,
        f"{float(history.true_failure_probability[index])!r}",
    )
    out("")
    out("The duration band was sampled INDEPENDENTLY of p by the generator, which")
    out("is what produces slow rare flakes and fast frequent ones and therefore")
    out("what makes the two rankings disagree. Measured correlation between the")
    out("generator's true p and its true mean duration, over the 72 tests:")
    finite = np.isfinite(history.true_failure_probability) & np.isfinite(history.true_mean_ms)
    correlation = float(
        np.corrcoef(
            history.true_failure_probability[finite], np.log(history.true_mean_ms[finite])
        )[0, 1]
    )
    out(f"  Pearson correlation of true p against log true mean duration: {correlation!r}")
    out("")
    out("This is a SAMPLE correlation over 72 independent draws, so it is not")
    out("zero; its standard error under independence is about 1/sqrt(72) =")
    out(f"{float(1.0 / np.sqrt(72.0))!r}, and the measured value is "
        f"{abs(correlation) * np.sqrt(72.0):.3f} standard errors from zero.")
    out.check(
        "the generator's p and duration are uncorrelated to within 3 standard errors",
        abs(correlation) * np.sqrt(72.0) <= 3.0,
        f"{abs(correlation) * np.sqrt(72.0):.3f} standard errors",
    )

    out.rule("4. WHAT THE LOADER REJECTS -- every case applied to the real data")
    out("")
    out("Each case edits a parsed copy of the committed history and must raise.")
    out("A loader that accepts malformed evidence silently is worse than none.")
    out("")
    for label, mutate in REJECTIONS:
        broken = json.loads(path.read_text(encoding="utf-8"))
        mutate(broken)
        try:
            history_from_mapping(broken)
        except ValueError as error:
            message = str(error).splitlines()[0][:66]
            out.control(
                f"the loader rejects {label}", True, must_fire=True, detail=message
            )
        else:
            out.control(f"the loader rejects {label}", False, must_fire=True,
                        detail="accepted")
    out("")
    out("CONTROL that must STAY SILENT: the unmodified committed history loads.")
    try:
        history_from_mapping(json.loads(path.read_text(encoding="utf-8")))
        rejected = False
    except ValueError as error:  # pragma: no cover - would be a defect
        rejected = True
        out(f"  unexpected rejection: {error}")
    out.control(
        "the loader rejects the unmodified committed history",
        rejected,
        must_fire=False,
        detail="loaded",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
