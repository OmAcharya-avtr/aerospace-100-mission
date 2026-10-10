"""Run all five example scripts, commit their stdout, and check README quotes.

The README quotes output from `examples/ece_bias_curve.py`. Rather than trust
that quote, this script runs every example in a clean subprocess, records the
stdout verbatim, and then asserts that each line the README quotes inside a
fenced block appears, character for character, in one of those transcripts.
A quoted line that does not appear is a failing check.

It also confirms that each example wrote the PNG the README embeds, so a
screenshot cannot drift away from the code that made it.
"""

from __future__ import annotations

import os
import subprocess
import sys

from _harness import HERE, Recorder

ROOT = HERE.parent
EXAMPLES = (
    "reliability_diagram.py",
    "ece_bias_curve.py",
    "decomposition_bars.py",
    "recalibration_audit.py",
    "binning_strategy.py",
)
#: Lines the README quotes from an example transcript, identified by a stable
#: prefix. Each must appear verbatim in the captured stdout.
README_QUOTED_PREFIXES = (
    "    5     200   0.000000",
    "   10     200   0.000000",
    "   20     200   0.000000",
    "   50     200   0.000000",
    "  100     200   0.000000",
    " bins       n   true_ECE",
)


def _fenced_lines(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] | None = None
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if cur is None:
                cur = []
            else:
                out.extend(cur)
                cur = None
            continue
        if cur is not None:
            cur.append(line)
    return out


def main() -> int:
    rec = Recorder("validate_examples")
    rec.header("Every example script, re-run, with its stdout recorded")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"

    transcripts: dict[str, str] = {}
    bad_exit = []
    for name in EXAMPLES:
        rec.say(f"$ MPLBACKEND=Agg python examples/{name}")
        res = subprocess.run(
            [sys.executable, str(ROOT / "examples" / name)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            timeout=900,
        )
        transcripts[name] = res.stdout
        # The first line is an absolute path, which differs per checkout; the
        # recorded transcript keeps only the basename so the committed file is
        # reproducible on any machine.
        for line in res.stdout.rstrip().splitlines():
            if line.startswith("wrote "):
                rec.say(f"  wrote screenshots/{line.rsplit('/', 1)[-1]}")
            else:
                rec.say(f"  {line}")
        if res.stderr.strip():
            for line in res.stderr.rstrip().splitlines():
                rec.say(f"  [stderr] {line}")
        rec.say(f"  -> exit status {res.returncode}")
        rec.say()
        if res.returncode != 0:
            bad_exit.append(name)

    rec.check(
        "every example script exits 0",
        reference="subprocess returncode",
        measured=f"{len(EXAMPLES) - len(bad_exit)} of {len(EXAMPLES)} exited 0"
        + ("" if not bad_exit else f"; failed: {bad_exit}"),
        expectation="all five exit 0",
        passed=not bad_exit,
    )

    pngs = sorted(p.name for p in (ROOT / "screenshots").glob("*.png"))
    expected_pngs = sorted(n.replace(".py", ".png") for n in EXAMPLES)
    rec.say(f"screenshots present: {', '.join(pngs)}")
    rec.say()
    rec.check(
        "each example wrote the PNG the README embeds",
        reference="screenshots/<example>.png for each examples/<example>.py",
        measured=f"{len(pngs)} PNGs present: {pngs}",
        expectation=f"exactly {expected_pngs}",
        passed=pngs == expected_pngs,
    )

    joined = "\n".join(transcripts.values())
    quoted = _fenced_lines((ROOT / "README.md").read_text())
    missing = []
    checked = 0
    for prefix in README_QUOTED_PREFIXES:
        readme_lines = [ln for ln in quoted if ln.startswith(prefix)]
        if not readme_lines:
            missing.append(f"{prefix!r} is not quoted in README.md at all")
            continue
        for line in readme_lines:
            checked += 1
            if line not in joined:
                missing.append(line)
    rec.say("README lines quoted from an example transcript:")
    for prefix in README_QUOTED_PREFIXES:
        for line in [ln for ln in quoted if ln.startswith(prefix)]:
            mark = "ok " if line in joined else "BAD"
            rec.say(f"  {mark} {line}")
    rec.say()
    rec.check(
        "every README line quoted from an example appears verbatim in the transcript",
        reference="stdout of the example scripts, captured above",
        measured=f"{checked} quoted lines checked, {len(missing)} not found"
        + ("" if not missing else f": {missing}"),
        expectation="0 not found. A previous batch in this mission published a "
        "README output block for a command it had not run.",
        passed=not missing,
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
