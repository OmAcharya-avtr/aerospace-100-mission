"""Audit of this repository's own use of ``pytest.approx``.

``pytest.approx(expected, rel=r)`` carries a DEFAULT ABSOLUTE TOLERANCE of
1e-12 and passes when EITHER bound is satisfied. For any quantity of order 1e-5,
every relative tolerance tighter than 2e-8 is therefore silently replaced by the
absolute default, and the assertion is far weaker than it reads. An assertion
written as ``rel=1e-15`` on a small number is, in practice, ``abs=1e-12``.

This script measures how much of this repository's test suite depends on an
absolute tolerance, in two passes.

**Pass 1, the default-reliance audit.** Every ``pytest.approx`` call that
passes ``rel=`` and does *not* pass ``abs=`` is counted. Those are the calls
whose effective tolerance is the 1e-12 default rather than the number written in
the test. The count is reported per file.

**Pass 2, the absolute-reliance audit.** Every ``pytest.approx`` call with
``rel=`` is rewritten so that its absolute tolerance is exactly zero, replacing
whatever it declared, and the suite is re-run. Assertions that then fail are the
ones that pass *only* through an absolute bound. Most of those are legitimate
and unavoidable: an expected value of exactly 0.0 cannot be approached within
any relative tolerance, since every relative bound around zero is zero. The
output classifies the failures on that basis rather than treating them all as
defects.

Files containing no ``rel=`` call at all are excluded from the copy, because the
rewrite cannot change them and copying them out of the repository breaks the one
test that resolves paths relative to its own location.

Nothing is written outside this repository's own directory except a temporary
copy of the test tree, and the temporary path is normalised out of the committed
output, which may not contain an absolute path.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, relative

ROOT = bootstrap()

APPROX = "pytest.approx("
PLACEHOLDER = "<temporary-copy>"


def approx_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` character spans of every ``pytest.approx(...)`` call.

    ``start`` points at the ``p`` of ``pytest``; ``end`` points one past the
    matching close parenthesis. Nested parentheses and brackets are matched;
    string literals are not interpreted, which is safe here because no test in
    this repository puts an unbalanced parenthesis inside a string argument to
    ``approx``.
    """
    spans: list[tuple[int, int]] = []
    index = text.find(APPROX)
    while index != -1:
        depth = 0
        cursor = index + len(APPROX) - 1
        while cursor < len(text):
            if text[cursor] in "([{":
                depth += 1
            elif text[cursor] in ")]}":
                depth -= 1
                if depth == 0:
                    spans.append((index, cursor + 1))
                    break
            cursor += 1
        index = text.find(APPROX, cursor + 1)
    return spans


def rewrite(text: str, force_abs_zero: bool) -> tuple[str, int, int]:
    """Rewrite the absolute tolerance of every ``rel=`` call.

    With ``force_abs_zero`` false, ``abs=0.0`` is inserted only where no ``abs=``
    is present, which is the default-reliance audit. With it true, an existing
    ``abs=`` is replaced as well, which is the absolute-reliance audit -- but
    only for calls declaring a NON-ZERO relative tolerance. A call written
    ``rel=0.0, abs=x`` is a pure absolute tolerance by design, and forcing its
    absolute bound to zero would turn it into bit-exact floating-point equality,
    which this mission's build guide forbids asserting at all. Those calls are
    counted and left alone.

    Returns ``(new_text, inserted, replaced)``.
    """
    spans = approx_spans(text)
    inserted = 0
    replaced = 0
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        body = text[start:end]
        pieces.append(text[cursor:start])
        if not re.search(r"\brel\s*=", body):
            pieces.append(body)
        elif re.search(r"\babs\s*=", body):
            zero_relative = re.search(r"\brel\s*=\s*0(\.0*)?\s*[,)]", body) is not None
            if force_abs_zero and not zero_relative:
                replaced += 1
                pieces.append(re.sub(r"\babs\s*=\s*[^,)]+", "abs=0.0", body))
            else:
                pieces.append(body)
        else:
            inserted += 1
            pieces.append(body[:-1] + ", abs=0.0)")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), inserted, replaced


def run_suite(copy: Path, temporary: Path) -> tuple[int, str]:
    """Run pytest over ``copy`` and return ``(exit status, normalised output)``."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(copy),
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "--tb=line",
        ],
        cwd=ROOT,
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MPLBACKEND": "Agg",
            "HOME": str(temporary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout + result.stderr).replace(str(temporary), PLACEHOLDER)
    combined = combined.replace(str(ROOT), PLACEHOLDER)
    combined = re.sub(r"(\.\./)+" + re.escape(PLACEHOLDER), PLACEHOLDER, combined)
    return result.returncode, combined


def main() -> None:
    out = Tee("validate_approx_audit_output.txt")
    out.rule("contactsched validate_approx_audit")
    for line in environment_lines():
        out(line)

    tests = ROOT / "tests"
    files = sorted(tests.glob("test_*.py"))

    out.rule("1. How this repository's own assertions are written")
    out(
        f"{'file':<28} {'approx calls':>13} {'with rel=':>10} {'rel+abs':>9} "
        f"{'rel only':>9}"
    )
    out("-" * 74)
    total_calls = 0
    total_rel = 0
    total_explicit = 0
    total_rel_only = 0
    with_rel_files: list[Path] = []
    for path in files:
        text = path.read_text()
        spans = approx_spans(text)
        with_rel = sum(1 for a, b in spans if re.search(r"\brel\s*=", text[a:b]))
        both = sum(
            1
            for a, b in spans
            if re.search(r"\brel\s*=", text[a:b]) and re.search(r"\babs\s*=", text[a:b])
        )
        rel_only = with_rel - both
        total_calls += len(spans)
        total_rel += with_rel
        total_explicit += both
        total_rel_only += rel_only
        if with_rel:
            with_rel_files.append(path)
        out(
            f"{path.name:<28} {len(spans):>13d} {with_rel:>10d} {both:>9d} "
            f"{rel_only:>9d}"
        )
    out("-" * 74)
    out(
        f"{'total':<28} {total_calls:>13d} {total_rel:>10d} {total_explicit:>9d} "
        f"{total_rel_only:>9d}"
    )
    out("")
    out("PASS 1 RESULT, the number the audit was written to produce:")
    out(f"  assertions relying on the 1e-12 default absolute tolerance : {total_rel_only}")
    out("")
    if total_rel_only == 0:
        out("Every rel= call in this suite states its own absolute tolerance, so no")
        out("assertion's effective tolerance differs from the number written in the test.")
        out("Inserting abs=0.0 where it is missing would change nothing, which is")
        out("verified byte for byte below rather than asserted.")
        identical = True
        for path in files:
            new_text, inserted, _replaced = rewrite(path.read_text(), force_abs_zero=False)
            if new_text != path.read_text() or inserted:
                identical = False
        out(f"  insertion pass is a no-op on every file : {identical}")
        if not identical:
            out("  the table above and the rewrite disagree, which is a defect in this script")
    else:
        out("The calls counted above state a relative tolerance that is NOT the tolerance")
        out("in force whenever the compared quantity is small. They are the subject of")
        out("pass 2 below.")

    out.rule("2. Pass 2: absolute tolerance forced to zero, suite re-run")
    out("Every rel= call has its absolute tolerance replaced by exactly zero. Files")
    out("with no rel= call are excluded, because the rewrite cannot change them and")
    out("copying them out of the repository would break the one test that resolves a")
    out("path relative to its own location.")
    out("")
    temporary = Path(tempfile.mkdtemp(prefix="contactsched-approx-audit-"))
    try:
        copy = temporary / "tests"
        copy.mkdir(parents=True)
        shutil.copy2(tests / "conftest.py", copy / "conftest.py")
        inserted_total = 0
        replaced_total = 0
        zero_relative_total = 0
        for path in with_rel_files:
            text = path.read_text()
            zero_relative_total += sum(
                1
                for a, b in approx_spans(text)
                if re.search(r"\brel\s*=\s*0(\.0*)?\s*[,)]", text[a:b])
            )
            new_text, inserted, replaced = rewrite(text, force_abs_zero=True)
            (copy / path.name).write_text(new_text)
            inserted_total += inserted
            replaced_total += replaced
        out(f"files copied               : {len(with_rel_files)}")
        out(f"files excluded (no rel=)   : {len(files) - len(with_rel_files)}")
        out(f"abs= inserted              : {inserted_total}")
        out(f"abs= replaced with 0.0     : {replaced_total}")
        out("")

        out("Control run first: the same copied files with their tolerances UNCHANGED.")
        out("Without this control a failure below could not be attributed to the")
        out("rewrite rather than to the act of copying.")
        control = temporary / "control"
        control.mkdir()
        shutil.copy2(tests / "conftest.py", control / "conftest.py")
        for path in with_rel_files:
            shutil.copy2(path, control / path.name)
        control_status, control_output = run_suite(control, temporary)
        control_tail = [line for line in control_output.splitlines() if "passed" in line
                        or "failed" in line or "error" in line]
        out(f"  control exit status : {control_status}")
        for line in control_tail[-3:]:
            out(f"  {line.strip()}")
        if control_status != 0:
            out("")
            out("  The control did not pass, so the comparison below is against a broken")
            out("  baseline and must be read with that in mind:")
            for line in control_output.splitlines():
                if re.search(r":\d+: ", line):
                    out(f"    {line.strip()}")
        out("")

        status, combined = run_suite(copy, temporary)
        out(f"rewritten exit status : {status}")
        out("")
        out("Output, with every absolute path normalised out so that this committed")
        out("file contains none:")
        out("")
        for line in combined.splitlines():
            if line.strip():
                out(f"  {line}")
        failures = [line.strip() for line in combined.splitlines() if re.search(r":\d+: ", line)]
        out("")
        out("PASS 2 RESULT:")
        out(
            "  assertions with a NON-ZERO declared relative tolerance, re-run with their"
        )
        out(f"  absolute tolerance forced to zero, that failed : {len(failures)}")
        out(f"  calls rewritten for this pass                  : {replaced_total}")
        out(f"  calls left alone as pure absolute tolerances    : {zero_relative_total}")
        out("")
        if status == 0:
            out("Every assertion in this suite that declares a non-zero relative tolerance")
            out("passes on that relative tolerance alone, with no absolute bound whatever.")
            out("That is a stronger statement than pass 1 and it is the one the audit set")
            out("out to test.")
            out("")
            out("The calls left alone are written rel=0.0 with an explicit absolute bound.")
            out("They are deliberately absolute: the quantities they compare are angles and")
            out("times whose expected value is zero or whose meaningful tolerance is a")
            out("physical quantity rather than a fraction. Forcing their absolute bound to")
            out("zero would make them assert bit-exact floating-point equality, which this")
            out("mission's build guide forbids after two products shipped that defect.")
        else:
            out("Those assertions declare a non-zero relative tolerance and nevertheless")
            out("pass only through their absolute bound. Each is listed above in full; none")
            out("is silently loosened. The fix for each is either to state the relative")
            out("tolerance the quantity actually achieves, or to declare rel=0.0 and make")
            out("the absolute bound the explicit contract.")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    out.rule("3. A worked demonstration of the trap")
    out("Why this audit exists, in four lines that can be run by hand:")
    out("")
    import pytest

    small = 1.0e-5
    perturbed = small + 1.0e-13
    passes_with_default = perturbed == pytest.approx(small, rel=1e-15)
    passes_without = perturbed == pytest.approx(small, rel=1e-15, abs=0.0)
    out(f"  value                        : {small:.6e}")
    out(f"  perturbed value              : {perturbed:.6e}")
    out(f"  relative difference          : {(perturbed - small) / small:.6e}")
    out(f"  approx(rel=1e-15)            : {'PASSES' if passes_with_default else 'fails'}")
    out(f"  approx(rel=1e-15, abs=0.0)   : {'PASSES' if passes_without else 'fails'}")
    out("")
    out("The relative difference is 1e-8, ten million times the stated relative")
    out("tolerance, and the assertion passes anyway because 1e-13 is inside the default")
    out("absolute bound of 1e-12.")
    if passes_with_default is not True or passes_without is not False:
        out("")
        out("NOTE: this demonstration did not behave as described, which means the")
        out("installed pytest version has changed the default. The audit above is still")
        out("the measurement; this section is explanatory.")

    out.rule("Result")
    out(f"written to {out.save()}")
    out(f"repository root printed relatively as {relative(ROOT / 'validation')}")


if __name__ == "__main__":
    main()
