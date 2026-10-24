"""Regenerate the committed case files and assert they are unchanged.

The five loadable case files in ``examples/cases/`` are hand-written, and this
script is how they are kept honest: it loads each one, renders it back through
``cmdguard.loader.dump_case``, and compares the result to a canonical rendering
of the file as committed. A difference means either the loader lost something
or the file carries something the loader does not round-trip, and either is a
defect.

The committed files keep their hand-written comments and their description
blocks, which a dump cannot reproduce, so the comparison is between two dumps
rather than between a dump and the file text: ``dump(load(file))`` against
``dump(load(dump(load(file))))``. That is a fixed-point check on the loader and
it is the strongest comparison available without discarding the comments, which
are worth more than the convenience of a byte comparison against the file.

``missing_declarations.yaml`` is deliberately broken and is checked to still be
rejected, which is the other half of the same property.
"""

from __future__ import annotations

import yaml

import _bootstrap
from _bootstrap import Tee, header, relative

ROOT = _bootstrap.bootstrap()

from cmdguard.loader import case_from_mapping, dump_case, load_case  # noqa: E402
from cmdguard.trace import forward_execute  # noqa: E402

CASES = ROOT / "examples" / "cases"
LOADABLE = (
    "nominal_science_pass",
    "late_payload_enable",
    "irreversible_before_verify",
    "irreversible_after_verify",
    "resource_and_exclusion",
)


def main() -> int:
    tee = Tee("regenerate_cases_output.txt")
    header(tee, "cmdguard -- committed cases round-trip through the loader unchanged")

    tee.rule("1. the loader is a fixed point on every committed case")
    tee(f"  {'case':<30s} {'steps':>6s} {'bytes':>7s} {'stable':>7s} "
        f"{'semantics':>10s}")
    for name in LOADABLE:
        path = CASES / f"{name}.yaml"
        case = load_case(path)
        first = dump_case(case)
        again = case_from_mapping(yaml.safe_load(first))
        second = dump_case(again)
        stable = first == second
        semantics = (
            forward_execute(case).fingerprint() == forward_execute(again).fingerprint()
        )
        tee(f"  {name:<30s} {len(case.sequence):>6d} {len(first):>7d} "
            f"{str(stable):>7s} {str(semantics):>10s}")
        tee.check(f"{name} renders to a fixed point", stable)
        tee.check(f"{name} keeps its finding set across the round trip", semantics)
        tee.check(f"{name} is readable at {relative(path)}", path.is_file())

    tee.rule("2. the deliberately broken case is still rejected")
    path = CASES / "missing_declarations.yaml"
    try:
        load_case(path)
    except ValueError as error:
        tee(f"  {relative(path)} raised ValueError:")
        tee(f"    {error}")
        tee.check("missing_declarations.yaml is rejected by the loader", True)
    else:
        tee.check("missing_declarations.yaml is rejected by the loader", False,
                  "it loaded, which means the loader stopped checking something")

    tee.rule("3. the canonical rendering of each case, for the record")
    tee("  This is what the loader round-trips to. It is not the committed file: the")
    tee("  committed files carry hand-written descriptions and per-step notes that")
    tee("  explain the arithmetic, and those are worth keeping.")
    for name in LOADABLE:
        rendered = dump_case(load_case(CASES / f"{name}.yaml"))
        tee("")
        tee(f"  --- {name} ({len(rendered.splitlines())} lines) ---")
        for line in rendered.splitlines()[:8]:
            tee(f"    {line}")
        tee("    ...")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
