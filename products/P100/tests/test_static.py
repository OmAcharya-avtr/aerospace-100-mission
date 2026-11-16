"""The static screen for the self-polluting-check class."""

from __future__ import annotations

import pytest

from seedaudit import static

SCANNER = '''from pathlib import Path

ROOT = Path(".")


def main() -> None:
    strays = list((ROOT / "validation").glob("*.txt"))
    (ROOT / "validation" / "probe_output.txt").write_text(str(len(strays)))
'''

CLEAN = '''from pathlib import Path

ROOT = Path(".")


def main() -> None:
    data = (ROOT / "data" / "in.csv").read_text()
    (ROOT / "out" / "result.txt").write_text(data)
'''

ABSENCE_ASSERTER = '''from pathlib import Path

ROOT = Path(".")


def main() -> None:
    (ROOT / "cache").mkdir()
    assert not (ROOT / "cache").is_dir()
'''


def test_the_scanner_pattern_is_flagged_and_names_both_source_lines() -> None:
    findings = static.find_self_pollution(SCANNER)
    assert len(findings) == 1
    assert findings[0].subkind == "write_under_glob"
    description = findings[0].describe()
    assert "line 7" in description
    assert "line 8" in description


def test_the_clean_pattern_is_not_flagged() -> None:
    assert static.find_self_pollution(CLEAN) == ()


def test_a_script_creating_what_it_asserts_absent_is_flagged() -> None:
    findings = static.find_self_pollution(ABSENCE_ASSERTER)
    assert findings
    assert findings[0].subkind in ("write_to_read_path", "write_under_read")


def test_a_write_to_the_exact_read_path_is_its_own_subkind() -> None:
    source = 'from pathlib import Path\np = Path("a/b.txt")\np.read_text()\np.write_text("x")\n'
    assert [f.subkind for f in static.find_self_pollution(source)] == ["write_to_read_path"]


def test_the_open_mode_decides_the_direction() -> None:
    for source, kind in (
        ('open("a/b.txt", "w")\n', "write"),
        ('open("a/b.txt", "a")\n', "write"),
        ('open("a/b.txt", "x")\n', "write"),
        ('open("a/b.txt", mode="w")\n', "write"),
        ('open("a/b.txt")\n', "read"),
        ('open("a/b.txt", "rb")\n', "read"),
    ):
        assert static.path_uses(source)[0].kind == kind, source


def test_directory_enumeration_is_recognised_in_every_shape() -> None:
    for source in (
        'from pathlib import Path\nlist(Path("a/b").glob("*.txt"))\n',
        'from pathlib import Path\nlist(Path("a/b").rglob("*"))\n',
        'from pathlib import Path\nlist(Path("a/b").iterdir())\n',
    ):
        globs = [use for use in static.path_uses(source) if use.kind == "glob"]
        assert globs and globs[0].expr == ("a", "b"), source


def test_a_module_level_enumeration_takes_its_first_argument_as_the_root() -> None:
    assert [(u.kind, u.expr) for u in static.path_uses('import os\nos.listdir("a/b")\n')] == [
        ("glob", ("a", "b"))
    ]


def test_the_receiver_is_not_the_path_for_a_first_argument_call() -> None:
    # fig.savefig(path) and numpy.save(path, array) have a receiver that is not
    # a path. Treating it as one reported variable names as directories and
    # buried the real findings underneath.
    assert [(u.kind, u.expr) for u in static.path_uses('fig.savefig("screenshots/x.png")\n')] == [
        ("write", ("screenshots", "x.png"))
    ]
    assert static.path_uses('from pathlib import Path\nPath("a/b").mkdir()\n')[0].kind == "write"


def test_a_copy_records_its_destination_as_a_write_and_its_source_as_a_read() -> None:
    uses = static.path_uses('import shutil\nshutil.copy("a/x", "b/y")\n')
    kinds = {(use.kind, use.expr) for use in uses}
    assert ("write", ("b", "y")) in kinds
    assert ("read", ("a", "x")) in kinds


def test_name_collisions_with_non_path_calls_are_not_treated_as_path_uses() -> None:
    # `str.replace` shares a name with `Path.replace`, and `json.dump` takes an
    # object rather than a path. Flagging either is a name collision, not
    # conservatism, and it buried the real findings.
    assert static.path_uses('text = "a".replace("a", "b")\n') == ()
    assert static.path_uses("import json\njson.dump(obj, handle)\n") == ()


def test_path_expressions_are_reduced_to_segments() -> None:
    assert static.path_uses('open("a/b/c.txt")\n')[0].expr == ("a", "b", "c.txt")
    assert static.path_uses('from pathlib import Path\nopen(Path("a") / "b" / "c.txt")\n')[
        0
    ].expr == ("a", "b", "c.txt")
    assert static.path_uses('open(ROOT / "b.txt")\n')[0].expr == ("$ROOT", "b.txt")
    assert static.path_uses('import os\nopen(os.path.join("a", "b.txt"))\n')[0].expr == (
        "a",
        "b.txt",
    )
    assert static.path_uses('open("./a/b.txt")\n')[0].expr == ("a", "b.txt")


def test_an_f_string_keeps_its_literal_prefix_and_wildcards_the_rest() -> None:
    expr = static.path_uses('open(f"validation/{stem}_output.txt")\n')[0].expr
    assert expr[0] == "validation"
    assert static.WILDCARD in expr


def test_containment_is_prefix_containment_with_wildcards_matching_anything() -> None:
    assert static.is_under(("a", "b"), ("a", "b"))
    assert static.is_under(("a",), ("a", "b", "c"))
    assert not static.is_under(("a", "b"), ("a", "c"))
    assert not static.is_under(("a", "b"), ("a",))
    assert static.is_under((), ("a", "b"))
    assert static.is_under(("a", static.WILDCARD), ("a", "z", "q"))
    assert static.segments_match(static.WILDCARD, "a")
    assert static.segments_match("a", static.WILDCARD)


def test_a_vacuous_path_expression_carries_no_containment_claim() -> None:
    # As a root it contains everything and as a path it lies under everything,
    # so a containment test involving it is true by vacuity. Reporting that as
    # a finding would be the screen manufacturing its own result.
    assert not static.is_informative((static.WILDCARD,))
    assert not static.is_informative(())
    assert static.is_informative(("validation", static.WILDCARD))
    assert static.find_self_pollution(
        'import json\ndata = json.load(handle)\nopen("a/b.txt", "w")\n'
    ) == ()


def test_coverage_counts_what_the_screen_cannot_reason_about() -> None:
    coverage = static.path_use_coverage('open(unknown_expression())\nopen("a/b.txt", "w")\n')
    assert coverage["uses"] == 2
    assert coverage["vacuous"] == 1
    assert coverage["informative"] == 1
    kinds = static.path_use_coverage(
        'from pathlib import Path\n'
        'Path("a").glob("*")\n'
        'open("b/c.txt")\n'
        'open("d/e.txt", "w")\n'
    )
    assert (kinds["globs"], kinds["reads"], kinds["writes"]) == (1, 1, 1)


def test_findings_are_deduplicated_and_uses_are_sorted_by_line() -> None:
    findings = static.find_self_pollution(SCANNER)
    assert len({(f.write.line, f.observe.line, f.subkind) for f in findings}) == len(findings)
    lines = [use.line for use in static.path_uses('open("a.txt")\nopen("b.txt", "w")\n')]
    assert lines == sorted(lines)


def test_input_validation_and_the_empty_rendering() -> None:
    assert static.PathUse(kind="read", expr=(), line=1, call="open").rendered == "."
    with pytest.raises(TypeError, match="must be a str"):
        static.path_uses(None)  # type: ignore[arg-type]
    with pytest.raises(SyntaxError):
        static.path_uses("def (:\n")


def test_regression_the_ruff_cache_pattern_is_flagged() -> None:
    # The mission's real instance: ruff writes .ruff_cache into the product
    # tree, and the audit then reports that tree as dirty.
    source = (
        "from pathlib import Path\n"
        "ROOT = Path('.')\n"
        "subprocess.run(['ruff', 'check'], cwd=str(ROOT))\n"
        "(ROOT / '.ruff_cache').mkdir(exist_ok=True)\n"
        "strays = list(ROOT.iterdir())\n"
    )
    assert any(f.subkind == "write_under_glob" for f in static.find_self_pollution(source))
