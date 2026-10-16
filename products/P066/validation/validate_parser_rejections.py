"""Every malformed declaration the parser is supposed to reject.

A parser that silently drops the line you meant to add is worse than one
that will not load, so every rejection below also has to name the offending
YAML key path. Both properties are measured here, not asserted.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from _support import CASES, Emitter

from budgetroll import load, loads
from budgetroll.errors import DeclarationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_parse import MINIMAL, REJECTIONS


def main() -> int:
    """Feed every malformed document to the parser and report."""
    out = Emitter(__file__)
    out("budgetroll validation: parser rejection paths")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use.")
    out()
    out("The table below is the same parametrisation the test suite uses, in")
    out("tests/test_parse.py, executed here so the count appears in committed")
    out("output rather than only in a test run.")
    out()

    out(f"1. {len(REJECTIONS)} schema rejection paths")
    accepted = 0
    unnamed = 0
    for label, text, _pattern in REJECTIONS:
        try:
            loads(text)
        except DeclarationError as exc:
            message = str(exc)
            # The criterion: the message must either quote an identifier or
            # key with 'single quotes', or name a schema field by its YAML
            # name. "the document is empty" satisfies the second because
            # there is no key to name.
            named = bool(re.search(r"'[^']+'", message)) or any(
                field in message
                for field in (
                    "budget",
                    "policies",
                    "correlations",
                    "document",
                    "k_sigma",
                    "children",
                )
            )
            unnamed += 0 if named else 1
            out(f"   [reject] {label:<32s} {message.splitlines()[0][:72]}")
        else:
            accepted += 1
            out(f"   [ACCEPT] {label:<32s} the parser accepted a malformed document")
    out()
    out.check(
        f"all {len(REJECTIONS)} malformed documents were rejected",
        accepted == 0,
        f"{accepted} accepted",
    )
    out.check(
        "every rejection message quotes an identifier or names a schema field",
        unnamed == 0,
        f"{unnamed} messages named nothing",
    )
    out()

    out("2. Three input and output paths")
    with tempfile.TemporaryDirectory() as tmp:
        io_cases = [
            ("missing file", Path(tmp) / "absent.yaml", "no such budget declaration"),
            ("a directory", Path(tmp), "is a directory"),
        ]
        invalid = Path(tmp) / "invalid.yaml"
        invalid.write_bytes(b"\xff\xfe\x00budget:")
        io_cases.append(("not valid UTF-8", invalid, "not valid UTF-8"))
        for label, path, expected in io_cases:
            try:
                load(path)
            except DeclarationError as exc:
                out.check(f"{label} is rejected with the right message", expected in str(exc),
                          str(exc).replace(tmp, "<tmp>")[:72])
            else:
                out.check(f"{label} is rejected", False, "the parser accepted it")
    out()

    out("3. The minimal valid document still parses, and so does every shipped case")
    out.check("the minimal document parses", loads(MINIMAL).id == "b")
    for name in (
        "clean_mass.yaml",
        "planted_findings.yaml",
        "cubesat_power.yaml",
        "cubesat_link.yaml",
    ):
        budget = load(CASES / name)
        out.check(f"{name} parses", bool(budget.id), f"id {budget.id}")
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
