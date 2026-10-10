"""Command line interface: ``python -m traceaudit``.

Exit statuses
-------------

====  ======================================================================
 0    the audit ran and found nothing blocking
 1    the audit ran and reported at least one blocking finding
 2    the audit could not run: a missing file, unparseable input, or a bad
      configuration
====  ======================================================================

Status 1 is the product behaving correctly, which is why the validation
scripts in this repository invoke the CLI through ``subprocess.run`` and
assert on ``returncode`` rather than letting a non-zero status propagate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .assertions import index_test_files
from .config import TraceConfig
from .findings import CODE_DESCRIPTIONS, audit
from .report import render_json, render_markdown_matrix, render_text
from .requirements import merge_documents, parse_requirements_file
from .testreports import load_report

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used by the tests."""
    parser = argparse.ArgumentParser(
        prog="python -m traceaudit",
        description=(
            "Compute requirements-to-test traceability from a markdown requirements "
            "document and a junit XML or pytest collection report. A clean run means "
            "the bookkeeping closes, not that anything has been verified."
        ),
        epilog="exit 0 = no findings, 1 = findings, 2 = could not run",
    )
    parser.add_argument("--version", action="version", version=f"traceaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="run every check and report findings")
    a.add_argument("--requirements", "-r", required=True, action="append", metavar="MD",
                   help="markdown requirements document; repeatable")
    a.add_argument("--report", "-t", required=True, metavar="FILE",
                   help="junit XML (.xml) or pytest --collect-only -q output")
    a.add_argument("--test-root", metavar="DIR",
                   help="directory of test sources, enabling the TA006 assertion heuristic")
    a.add_argument("--config", "-c", metavar="JSON", help="configuration file")
    a.add_argument("--ignore", action="append", default=[], metavar="CODE",
                   help="report this code but do not let it set the exit status; repeatable")
    a.add_argument("--no-heuristic", action="store_true",
                   help="do not run the assertion heuristic at all")
    a.add_argument("--show-matrix", action="store_true", help="print the full mapping")
    a.add_argument("--json", action="store_true", help="emit JSON instead of text")
    a.add_argument("--markdown-matrix", action="store_true",
                   help="emit only the markdown traceability table")
    a.add_argument("--relative-to", metavar="DIR",
                   help="print paths relative to this directory")

    r = sub.add_parser("requirements", help="list the requirement declarations parsed")
    r.add_argument("--requirements", "-r", required=True, action="append", metavar="MD")
    r.add_argument("--config", "-c", metavar="JSON")
    r.add_argument("--json", action="store_true")

    t = sub.add_parser("tests", help="list the test cases, outcomes and claims parsed")
    t.add_argument("--report", "-t", required=True, metavar="FILE")
    t.add_argument("--config", "-c", metavar="JSON")
    t.add_argument("--json", action="store_true")

    h = sub.add_parser("assertions", help="run the assertion heuristic over a test tree")
    h.add_argument("--test-root", required=True, metavar="DIR")
    h.add_argument("--json", action="store_true")

    sub.add_parser("codes", help="list the finding codes and what they mean")
    sub.add_parser("config-template", help="print the default configuration as JSON")
    return parser


def _load_config(path: str | None, ignore: list[str], no_heuristic: bool) -> TraceConfig:
    cfg = TraceConfig.from_json_file(path) if path else TraceConfig.default()
    data = cfg.to_dict()
    if ignore:
        unknown = sorted(set(ignore) - set(CODE_DESCRIPTIONS))
        if unknown:
            raise ValueError(
                f"unknown finding code(s) to ignore: {', '.join(unknown)}; "
                f"known codes are {', '.join(CODE_DESCRIPTIONS)}"
            )
        data["ignored_codes"] = sorted(set(data["ignored_codes"]) | set(ignore))
    if no_heuristic:
        data["heuristic_assertions"] = False
    return TraceConfig.from_dict(data)


def _cmd_audit(args: argparse.Namespace, out) -> int:
    rel = Path(args.relative_to) if args.relative_to else None
    cfg = _load_config(args.config, args.ignore, args.no_heuristic)
    docs = [
        parse_requirements_file(p, config=cfg, relative_to=rel) for p in args.requirements
    ]
    document = merge_documents(docs)
    report = load_report(args.report, config=cfg, relative_to=rel)
    result = audit(document, report, config=cfg, test_root=args.test_root,
                   relative_to=rel)
    if args.markdown_matrix:
        print(render_markdown_matrix(result), file=out)
    elif args.json:
        print(render_json(result), file=out)
    else:
        print(render_text(result, show_matrix=args.show_matrix), file=out)
    return EXIT_FINDINGS if result.blocking else EXIT_OK


def _cmd_requirements(args: argparse.Namespace, out) -> int:
    cfg = _load_config(args.config, [], False)
    document = merge_documents(
        [parse_requirements_file(p, config=cfg) for p in args.requirements]
    )
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "declarations": [
                        {"id": r.id, "title": r.title, "source": r.source, "line": r.line}
                        for r in document.declarations
                    ],
                    "unique_ids": list(document.ids),
                    "duplicates": {k: len(v) for k, v in document.duplicates.items()},
                },
                indent=2,
            ),
            file=out,
        )
        return EXIT_OK
    print(f"{len(document.declarations)} declaration(s), "
          f"{len(document.ids)} unique id(s)", file=out)
    for req in document.declarations:
        print(f"  {req.id:<10s} {req.location:<32s} {req.title}", file=out)
    for req_id, decls in sorted(document.duplicates.items()):
        print(f"  duplicate: {req_id} declared {len(decls)} times", file=out)
    return EXIT_OK


def _cmd_tests(args: argparse.Namespace, out) -> int:
    cfg = _load_config(args.config, [], False)
    report = load_report(args.report, config=cfg)
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "source": report.source,
                    "kind": report.kind,
                    "declared_totals": dict(report.declared_totals),
                    "outcome_counts": report.outcome_counts,
                    "cases": [
                        {
                            "node_id": c.node_id,
                            "classname": c.classname,
                            "name": c.name,
                            "outcome": c.outcome.value,
                            "file": c.file,
                            "line": c.line,
                            "claims": list(c.claims),
                            "message": c.message,
                        }
                        for c in report.cases
                    ],
                },
                indent=2,
            ),
            file=out,
        )
        return EXIT_OK
    print(f"{report.source} ({report.kind})", file=out)
    print(f"  header totals : {dict(report.declared_totals)}", file=out)
    print(f"  counted       : {report.outcome_counts}", file=out)
    for case in report.cases:
        claims = ",".join(case.claims) or "-"
        print(f"  {case.outcome.value:<8s} {case.node_id:<58s} {claims}", file=out)
    return EXIT_OK


def _cmd_assertions(args: argparse.Namespace, out) -> int:
    index, notes = index_test_files(args.test_root)
    rows = sorted(
        {(v.file, v.line, v.function, v.has_assertion, v.detail) for v in index.values()},
        key=lambda r: (r[0], r[1]),
    )
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "notes": list(notes),
                    "functions": [
                        {
                            "file": f,
                            "line": ln,
                            "function": fn,
                            "has_assertion": ha,
                            "detail": d,
                        }
                        for f, ln, fn, ha, d in rows
                    ],
                },
                indent=2,
            ),
            file=out,
        )
        return EXIT_OK
    print(f"{len(rows)} function(s) inspected under {args.test_root}", file=out)
    print("HEURISTIC: calls are not followed; a test whose assertion lives in a "
          "helper reads as empty.", file=out)
    for f, ln, fn, ha, d in rows:
        flag = "asserts" if ha else "NO ASSERTION"
        print(f"  {flag:<12s} {fn:<42s} {f}:{ln}  ({d})", file=out)
    for note in notes:
        print(f"  note: {note}", file=out)
    return EXIT_OK


def _cmd_codes(out) -> int:
    for code, description in CODE_DESCRIPTIONS.items():
        tag = "  [heuristic]" if code == "TA006" else ""
        print(f"{code}  {description}{tag}", file=out)
    return EXIT_OK


def _cmd_config_template(out) -> int:
    import json

    print(json.dumps(TraceConfig.default().to_dict(), indent=2), file=out)
    return EXIT_OK


def main(argv: list[str] | None = None, out=None) -> int:
    """Entry point.  Returns the exit status rather than calling ``sys.exit``."""
    out = out or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _cmd_audit(args, out)
        if args.command == "requirements":
            return _cmd_requirements(args, out)
        if args.command == "tests":
            return _cmd_tests(args, out)
        if args.command == "assertions":
            return _cmd_assertions(args, out)
        if args.command == "codes":
            return _cmd_codes(out)
        if args.command == "config-template":
            return _cmd_config_template(out)
    except (FileNotFoundError, NotADirectoryError, ValueError, TypeError) as exc:
        print(f"traceaudit: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser.error(f"unhandled command {args.command!r}")
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
