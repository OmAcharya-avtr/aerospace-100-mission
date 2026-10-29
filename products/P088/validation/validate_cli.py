"""Every CLI subcommand, run as a subprocess, with its exit status recorded.

The subprocess environment is built by :func:`_bootstrap.subprocess_environment`,
which gives each child a temporary ``HOME`` and ``MPLCONFIGDIR`` outside the
repository and sets ``PYTHONDONTWRITEBYTECODE``. An earlier product in this
mission set ``HOME`` to its own product root, matplotlib wrote a configuration
directory into the repository, and the release gate failed it on a tracked
build artifact and on an absolute path in one go.

The dictionary files the ``diff`` and ``impact`` subcommands need are written
into a temporary directory outside the repository and removed afterwards, so
running this script leaves the tree exactly as it found it. That is checked,
not assumed: the set of files under the repository is compared before and
after.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, relative, subprocess_environment

ROOT = bootstrap()

from tmtcdiff.changes import ChangeClass  # noqa: E402
from tmtcdiff.parse import dumps  # noqa: E402
from tmtcdiff.report import format_table  # noqa: E402
from tmtcdiff.synth import baseline_dictionary, inject_changes, synthetic_corpus  # noqa: E402

#: Files and directories the tree is allowed to gain while this script runs.
#: Only this script's own output file.
EXPECTED_NEW = {"validation/validate_cli_output.txt"}


def _tracked_files() -> set[str]:
    skip = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis"}
    out: set[str] = set()
    for path in ROOT.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            out.add(path.relative_to(ROOT).as_posix())
    return out


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("tmtcdiff -- every CLI subcommand, as a subprocess")
    for line in environment_lines():
        out(line)

    before = _tracked_files()
    scratch = Path(tempfile.mkdtemp(prefix="tmtcdiff-cli-"))
    environment = subprocess_environment()
    try:
        old = baseline_dictionary(40, 11311)
        pair = inject_changes(
            old,
            (
                ChangeClass.RESCALED,
                ChangeClass.UNIT_CHANGED,
                ChangeClass.REENUMERATED,
                ChangeClass.REMOVED,
                ChangeClass.RENAMED,
                ChangeClass.ADDED,
            ),
            11312,
        )
        old_path = scratch / "dictionary_1_0.json"
        new_path = scratch / "dictionary_1_1.json"
        old_path.write_text(dumps(pair.old), encoding="utf-8")
        new_path.write_text(dumps(pair.new), encoding="utf-8")

        corpus_root = scratch / "corpus"
        corpus = synthetic_corpus(
            pair.old, 11313, names=[change.name for change in pair.truth]
        )
        for source in corpus.sources:
            target = corpus_root / source.label
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.text, encoding="utf-8")
        (corpus_root / "notes.bin").write_bytes(b"\x00\x01binary, not a corpus source")

        out("")
        out(f"dictionary files written to a temporary directory outside {relative(ROOT)}")
        out(f"corpus sources written : {len(corpus.sources)} plus one unreadable file")

        invocations: list[tuple[str, list[str], int]] = [
            ("version", ["--version"], 0),
            ("info", ["info"], 0),
            ("taxonomy", ["taxonomy"], 0),
            ("diff", ["diff", str(old_path), str(new_path)], 0),
            (
                "diff --no-renames",
                ["diff", str(old_path), str(new_path), "--no-renames"],
                0,
            ),
            (
                "diff --strategy greedy",
                ["diff", str(old_path), str(new_path), "--strategy", "greedy"],
                0,
            ),
            (
                "diff --fail-on-silent",
                ["diff", str(old_path), str(new_path), "--fail-on-silent"],
                2,
            ),
            (
                "impact",
                [
                    "impact",
                    str(old_path),
                    str(new_path),
                    "--corpus",
                    str(corpus_root),
                    "--max-rows",
                    "5",
                ],
                0,
            ),
            ("demo", ["demo", "--size", "40", "--max-rows", "3"], 0),
            ("sweep", ["sweep", "--sizes", "30", "--thresholds", "0.6", "0.8"], 0),
            ("negative-control", ["negative-control", "--size", "30"], 0),
            ("missing file", ["diff", "no_such_file.json", str(new_path)], 1),
            (
                "bad threshold",
                ["diff", str(old_path), str(new_path), "--threshold", "5.0"],
                1,
            ),
            ("unknown subcommand", ["nonsense"], 2),
        ]

        rows = []
        failures = 0
        transcripts: dict[str, str] = {}
        for label, arguments, expected in invocations:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "tmtcdiff", *arguments],
                cwd=str(scratch),
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            status = completed.returncode
            transcripts[label] = completed.stdout
            ok = status == expected
            if not ok:
                failures += 1
            rows.append(
                [
                    label,
                    expected,
                    status,
                    len(completed.stdout.splitlines()),
                    len(completed.stderr.splitlines()),
                    "ok" if ok else "UNEXPECTED",
                ]
            )
        out.rule("exit status of every invocation")
        out(
            format_table(
                ["invocation", "expected", "actual", "stdout lines", "stderr lines", "verdict"],
                rows,
            )
        )
        out.check(
            "every CLI invocation exits with its expected status",
            failures == 0,
            f"{failures} of {len(invocations)} unexpected",
        )

        out.rule("quoted fragments -- what the README shows, from this run")
        out("python -m tmtcdiff --version")
        out(transcripts["version"].strip())
        out("")
        out("python -m tmtcdiff negative-control --size 30")
        out(transcripts["negative-control"].strip())
        out("")
        out("python -m tmtcdiff diff dictionary_1_0.json dictionary_1_1.json  (first 24 lines)")
        for line in transcripts["diff"].splitlines()[:24]:
            out(line)

        out.rule("the unreadable corpus source appears in the impact output")
        impact_text = transcripts["impact"]
        out.check(
            "the impact report names the unreadable source",
            "notes.bin" in impact_text,
            "found" if "notes.bin" in impact_text else "absent",
        )
        out.check(
            "the impact report prints the coverage verdict",
            "coverage verdict" in impact_text,
        )
        out.check(
            "no CLI output contains a safety or certification claim",
            # Word boundaries, not substrings: the generator mints enumeration
            # labels such as SAFE_MODE and the CLI prints them.
            not any(
                re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
                for text in transcripts.values()
                for word in ("safe", "certified", "flight-qualified", "production-ready")
            ),
        )
        out.check(
            "no CLI output contains the absolute repository path",
            not any(str(ROOT) in text for text in transcripts.values()),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    after = _tracked_files()
    gained = after - before
    lost = before - after
    out.rule("the tree is unchanged apart from this script's own output")
    out(f"files gained : {sorted(gained)}")
    out(f"files lost   : {sorted(lost)}")
    out.check(
        "the run adds nothing to the repository but its own output file",
        gained <= EXPECTED_NEW and not lost,
        f"gained {sorted(gained - EXPECTED_NEW)}, lost {sorted(lost)}",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
