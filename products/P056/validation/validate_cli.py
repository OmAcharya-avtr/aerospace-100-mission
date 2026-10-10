"""Re-run every command quoted in the documentation and record what it prints.

Two jobs:

1. **Execute** each CLI invocation in a clean subprocess and record its stdout,
   stderr and exit status verbatim, so that the output blocks in README.md can
   be checked against something rather than trusted.
2. **Cover** every command quoted anywhere in the documentation. The script
   scans README.md, MODEL_CARD.md, CHANGELOG.md and validation/VALIDATION.md
   for shell commands inside fenced code blocks and asserts that each one is
   either executed below or named in ``NOT_EXECUTED_HERE`` with a reason. A
   quoted command that is neither is a failing check, because a documented
   command nobody runs is how an unrun output block gets published.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from _harness import HERE, Recorder

ROOT = HERE.parent
SRC = str(ROOT / "src")

#: CLI invocations executed below. Arguments only; the interpreter and
#: ``-m calibaudit`` are added by the runner.
CLI_COMMANDS: tuple[list[str], ...] = (
    ["--help"],
    ["--version"],
    ["specs"],
    ["specs", "--json"],
    ["decompose", "--spec", "overconfident", "-n", "4000", "--bins", "10"],
    ["decompose", "--spec", "calibrated", "-n", "2000", "--strategy", "equal_mass"],
    ["decompose", "-n", "100", "--bins", "0"],
    ["ece", "--spec", "calibrated", "-n", "2000", "--bins", "15", "--replicates", "400"],
    ["ece", "--spec", "biased_high", "-n", "2000", "--bins", "15", "--replicates", "400"],
    [
        "ece-bias",
        "--spec", "calibrated",
        "--bins-grid", "5", "10", "20", "50",
        "--samples-grid", "200", "1000", "5000",
        "--replicates", "60",
    ],
    ["ece-bias", "--spec", "calibrated", "--bins-grid", "10", "--samples-grid", "400",
     "--replicates", "20"],
    ["reliability", "--spec", "overconfident", "-n", "4000", "--bins", "12",
     "--bootstrap", "500"],
    ["reliability", "-n", "200", "--level", "1.5"],
    ["recalibrate", "--spec", "overconfident", "-n", "2000", "--bootstrap", "400"],
    ["recalibrate", "--spec", "calibrated", "-n", "200", "--bootstrap", "400",
     "--require-improvement"],
    ["recalibrate", "--spec", "biased_high", "-n", "8000", "--bootstrap", "400",
     "--require-improvement"],
    ["recalibrate", "-n", "2000", "--methods", "raw", "platt", "--ensemble", "100",
     "--bootstrap", "200"],
)

#: Commands quoted in the docs that this script deliberately does not run, and
#: why. Matching is by the first two whitespace-separated tokens.
NOT_EXECUTED_HERE: dict[str, str] = {
    "git clone": "the repository is already present; cloning it here would prove nothing",
    "cd calibaudit": "a shell builtin, part of the clone recipe",
    "python -m": "python -m venv and python -m pytest are environment setup, not "
    "package behaviour; pytest is run by the release gate and its output is "
    "committed as validation/outputs/pytest_output.txt",
    "pip install": "environment setup",
    "curl -s": "the manual, network-dependent snapshot capture documented in "
    "validation/VALIDATION.md section 7; validate_alternatives.py is deliberately "
    "offline so the release gate gets byte-identical output, and api.crossref.org "
    "and some PyPI access are blocked in parts of this environment",
    "pip download": "part of the same manual snapshot capture",
    "unzip -o": "part of the same manual snapshot capture",
    "source .venv/bin/activate": "environment setup",
    "ruff check": "lint, run by the release gate",
    "MPLBACKEND=Agg": "the five example scripts; each writes a PNG and is run by the "
    "release gate, which is also what keeps screenshots/ in step with the code",
    "python validation/validate_cli.py": "this script; running it from inside itself "
    "would not terminate",
    "python validation/validate_examples.py": "run separately by the release gate; it "
    "runs all five example scripts and checks the README lines quoted from them",
    "python validation/validate_environment.py": "run separately by the release gate",
    "python validation/validate_alternatives.py": "run separately by the release gate",
    "python validation/validate_decomposition_identity.py": "run separately",
    "python validation/validate_ece_bias.py": "run separately",
    "python validation/validate_reliability_bands.py": "run separately",
    "python validation/validate_recalibration.py": "run separately",
    "python validation/validate_sklearn_interop.py": "run separately",
    "python validation/worked_example.py": "run separately; its output is committed as "
    "validation/outputs/worked_example_output.txt and quoted in README.md",
    "python examples/reliability_diagram.py": "an example script, run by the gate",
    "python examples/ece_bias_curve.py": "an example script, run by the gate",
    "python examples/decomposition_bars.py": "an example script, run by the gate",
    "python examples/recalibration_audit.py": "an example script, run by the gate",
    "python examples/binning_strategy.py": "an example script, run by the gate",
}

DOCS = ("README.md", "MODEL_CARD.md", "CHANGELOG.md", "validation/VALIDATION.md")
_FENCE = re.compile(r"^\s*```")
_SHELL_HEADS = ("python", "python3", "pip", "git", "cd", "ruff", "source", "curl", "unzip")


def _fenced_blocks(text: str) -> list[str]:
    """Every fenced block's body, whatever its language tag.

    Written as a line scan rather than a regex: a non-greedy regex over a
    document containing both tagged and untagged fences pairs the wrong
    delimiters and silently skips blocks, which is how the first version of
    this script found 6 of the commands instead of all of them.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if _FENCE.match(line):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _quoted_commands() -> list[tuple[str, str]]:
    """Return (document, command) for every shell command in a fenced block."""
    found: list[tuple[str, str]] = []
    for name in DOCS:
        path = ROOT / name
        if not path.exists():
            continue
        for block in _fenced_blocks(path.read_text()):
            for raw in block.splitlines():
                line = raw.strip()
                if line.startswith("$ "):
                    line = line[2:].strip()
                if not line:
                    continue
                tokens = line.split()
                head = tokens[0]
                if head in _SHELL_HEADS:
                    found.append((name, line))
                elif "=" in head and len(tokens) > 1 and tokens[1] in _SHELL_HEADS:
                    # VAR=value command ...
                    found.append((name, line))
    return found


def _is_synopsis(command: str) -> bool:
    """A usage line with optional-argument placeholders, not a runnable command."""
    return "[" in command or "..." in command or "|" in command


def _covered(command: str, executed: set[str]) -> str | None:
    """Return the reason this command needs no execution, or None if it does."""
    tokens = command.split()
    if tokens[:3] == ["python", "-m", "calibaudit"]:
        rest = tokens[3:]
        if _is_synopsis(command):
            # A synopsis is covered when its subcommand is exercised by at
            # least one executed invocation, so a new subcommand cannot be
            # documented without also being run here.
            sub = next((t for t in rest if not t.startswith("-")), None)
            if sub is None:
                return "usage synopsis, no subcommand"
            if any(c.split()[0] == sub for c in executed):
                return f"synopsis; '{sub}' executed below"
            return None
        key = " ".join(rest)
        return "executed below" if key in executed else None
    for prefix, reason in NOT_EXECUTED_HERE.items():
        if command.startswith(prefix):
            return reason
    two = " ".join(tokens[:2])
    return NOT_EXECUTED_HERE.get(two)


def main() -> int:
    rec = Recorder("validate_cli")
    rec.header("Every documented command, re-run - calibaudit 0.1.0")

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["COLUMNS"] = "100"

    executed: set[str] = set()
    failures = 0
    for args in CLI_COMMANDS:
        executed.add(" ".join(args))
        rec.say(f"$ python -m calibaudit {' '.join(args)}")
        res = subprocess.run(
            [sys.executable, "-m", "calibaudit", *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            timeout=900,
        )
        for line in res.stdout.rstrip().splitlines():
            rec.say(f"  {line}")
        for line in res.stderr.rstrip().splitlines():
            rec.say(f"  [stderr] {line}")
        rec.say(f"  -> exit status {res.returncode}")
        rec.say()
        if res.returncode not in (0, 1, 2):
            failures += 1

    rec.check(
        "every executed CLI command exits with a documented status",
        reference="README.md states 0 on success, 1 on a usage or input error, "
        "2 from argparse or from recalibrate --require-improvement",
        measured=f"{len(CLI_COMMANDS)} commands run, {failures} with an undocumented "
        "exit status",
        expectation="0 undocumented statuses",
        passed=failures == 0,
    )

    # --- the nonzero-exit contract, asserted on returncode -----------------
    rec.say("The nonzero-exit contract, asserted on subprocess returncode")
    rec.say()
    contract = [
        (["recalibrate", "--spec", "calibrated", "-n", "200", "--bootstrap", "400",
          "--require-improvement"], 2,
         "nothing beats the raw baseline on an already-calibrated forecast"),
        (["recalibrate", "--spec", "biased_high", "-n", "8000", "--bootstrap", "400",
          "--require-improvement"], 0,
         "Platt scaling beats the raw baseline on a strongly biased forecast"),
        (["decompose", "-n", "100", "--bins", "0"], 1, "invalid bin count"),
        (["reliability", "-n", "200", "--level", "1.5"], 1, "invalid band level"),
        (["decompose", "--spec", "not_a_spec"], 2, "argparse rejects the choice"),
        (["nonsense"], 2, "argparse rejects the subcommand"),
        ([], 2, "argparse requires a subcommand"),
    ]
    wrong = []
    for args, want, why in contract:
        res = subprocess.run(
            [sys.executable, "-m", "calibaudit", *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            timeout=900,
        )
        ok = res.returncode == want
        if not ok:
            wrong.append((args, want, res.returncode))
        rec.say(
            f"  {'ok ' if ok else 'BAD'} exit {res.returncode} (want {want}): "
            f"python -m calibaudit {' '.join(args) or '<no arguments>'}  -- {why}"
        )
    rec.say()
    rec.check(
        "exit statuses match the contract in README.md",
        reference="subprocess returncode, not a caught SystemExit",
        measured=f"{len(contract) - len(wrong)} of {len(contract)} correct"
        + ("" if not wrong else f"; wrong: {wrong}"),
        expectation="all correct",
        passed=not wrong,
    )

    # --- coverage of the documentation --------------------------------------
    rec.say("Coverage: every command quoted in the documentation")
    rec.say()
    quoted = _quoted_commands()
    uncovered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for doc, command in quoted:
        if command in seen:
            continue
        seen.add(command)
        reason = _covered(command, executed)
        if reason is None:
            uncovered.append((doc, command))
            rec.say(f"  NOT COVERED  [{doc}] {command}")
        else:
            rec.say(f"  {reason[:28]:<28} [{doc}] {command}")
    rec.say()
    rec.check(
        "every command quoted in the documentation is executed here or has a "
        "recorded reason not to be",
        reference=f"fenced code blocks in {', '.join(DOCS)}",
        measured=f"{len(seen)} distinct commands quoted, {len(uncovered)} uncovered"
        + ("" if not uncovered else f": {uncovered}"),
        expectation="0 uncovered. A documented command that nobody runs is how an "
        "unrun output block gets published.",
        passed=not uncovered,
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
