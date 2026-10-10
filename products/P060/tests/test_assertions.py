"""The assertion heuristic, rule by rule, plus its hand-labelled error rates."""

from __future__ import annotations

import ast
import json

import pytest

from traceaudit import index_test_files, inspect_file, inspect_function, inspect_source


def _verdict(body: str, name: str = "test_x"):
    source = f"def {name}():\n" + "\n".join(f"    {line}" for line in body.splitlines())
    return inspect_source(source, file="t.py")[name]


@pytest.mark.verifies("REQ-018")
def test_r1_plain_assert_counts():
    verdict = _verdict("assert 2 + 2 == 4")
    assert verdict.has_assertion is True
    assert "R1" in verdict.rules_fired


@pytest.mark.verifies("REQ-018")
@pytest.mark.parametrize("literal", ["True", "1", "'x'", "3.5"])
def test_r2_truthy_constant_assert_does_not_count(literal):
    verdict = _verdict(f"assert {literal}")
    assert verdict.has_assertion is False
    assert verdict.rules_fired == ("R2", "R6")


@pytest.mark.verifies("REQ-018")
def test_r2_falsy_constant_assert_counts_as_an_assertion():
    # assert False is not vacuous: it fails unconditionally, so it is an assertion.
    verdict = _verdict("assert False")
    assert verdict.has_assertion is True
    assert "R1" in verdict.rules_fired


@pytest.mark.verifies("REQ-018")
@pytest.mark.parametrize("ctx", ["pytest.raises(ValueError)", "pytest.warns(UserWarning)",
                                 "pytest.deprecated_call()"])
def test_r3_raises_and_warns_context_managers_count(ctx):
    verdict = _verdict(f"with {ctx}:\n    pass")
    assert verdict.has_assertion is True
    assert "R3" in verdict.rules_fired


@pytest.mark.verifies("REQ-018")
@pytest.mark.parametrize("call", ["self.assertEqual(1, 1)",
                                  "np.testing.assert_allclose(a, b)",
                                  "assert_that(1, equal_to(1))",
                                  "assert_frame_equal(x, y)"])
def test_r4_any_call_whose_tail_starts_with_assert_counts(call):
    verdict = _verdict(call)
    assert verdict.has_assertion is True
    assert "R4" in verdict.rules_fired


@pytest.mark.verifies("REQ-018")
@pytest.mark.parametrize("call", ["pytest.fail('bad')", "self.fail('bad')"])
def test_r5_an_explicit_fail_counts(call):
    verdict = _verdict(call)
    assert verdict.has_assertion is True
    assert "R5" in verdict.rules_fired


@pytest.mark.verifies("REQ-018")
def test_r6_fires_when_nothing_else_does():
    verdict = _verdict("value = compute()\nvalue += 1")
    assert verdict.has_assertion is False
    assert verdict.rules_fired == ("R6",)
    assert "no assert" in verdict.detail


@pytest.mark.verifies("REQ-018")
def test_an_assertion_in_a_nested_def_is_found():
    verdict = _verdict("def check(v):\n    assert v > 0\ncheck(1)")
    assert verdict.has_assertion is True


@pytest.mark.verifies("REQ-018")
def test_an_assertion_in_a_loop_or_branch_is_found():
    assert _verdict("for v in items:\n    assert v > 0").has_assertion is True
    assert _verdict("if flag:\n    assert v > 0").has_assertion is True
    assert _verdict("try:\n    assert v > 0\nexcept Exception:\n    pass").has_assertion is True


@pytest.mark.verifies("REQ-018")
def test_verdict_records_the_function_name_file_and_line():
    verdicts = inspect_source("x = 1\n\n\ndef test_y():\n    assert x\n", file="mod.py")
    verdict = verdicts["test_y"]
    assert verdict.function == "test_y"
    assert verdict.file == "mod.py"
    assert verdict.line == 4


@pytest.mark.verifies("REQ-018")
def test_methods_and_free_functions_are_both_inspected():
    source = (
        "class TestA:\n"
        "    def test_method(self):\n"
        "        assert 1 == 1\n"
        "\n"
        "def test_free():\n"
        "    pass\n"
    )
    verdicts = inspect_source(source, file="m.py")
    assert set(verdicts) == {"test_method", "test_free"}
    assert verdicts["test_method"].has_assertion is True
    assert verdicts["test_free"].has_assertion is False


@pytest.mark.verifies("REQ-018")
def test_async_test_functions_are_inspected():
    verdicts = inspect_source("async def test_async():\n    assert 1 == 1\n", file="m.py")
    assert verdicts["test_async"].has_assertion is True


@pytest.mark.verifies("REQ-018")
def test_inspect_function_can_be_called_on_a_parsed_node():
    tree = ast.parse("def test_z():\n    assert 1 == 1\n")
    node = tree.body[0]
    verdict = inspect_function(node, file="z.py")
    assert verdict.function == "test_z"
    assert verdict.has_assertion is True


@pytest.mark.verifies("REQ-022")
def test_unparseable_python_raises_value_error():
    with pytest.raises(ValueError, match="not parseable Python"):
        inspect_source("def test_x(:\n", file="bad.py")


@pytest.mark.verifies("REQ-022")
def test_inspect_file_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="test source file not found"):
        inspect_file(tmp_path / "absent.py")


@pytest.mark.verifies("REQ-022")
def test_index_test_files_raises_for_a_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="test root is not a directory"):
        index_test_files(tmp_path / "absent")


@pytest.mark.verifies("REQ-018")
def test_index_test_files_keys_by_stem_and_name(tmp_path):
    (tmp_path / "test_a.py").write_text("def test_one():\n    assert 1 == 1\n", encoding="utf-8")
    index, notes = index_test_files(tmp_path)
    assert ("test_a", "test_one") in index
    assert ("", "test_one") in index
    assert notes == ()


@pytest.mark.verifies("REQ-018")
def test_index_test_files_notes_a_name_collision(tmp_path):
    (tmp_path / "test_a.py").write_text("def test_dup():\n    assert 1 == 1\n", encoding="utf-8")
    (tmp_path / "test_b.py").write_text("def test_dup():\n    pass\n", encoding="utf-8")
    index, notes = index_test_files(tmp_path)
    assert ("", "test_dup") not in index
    assert any("occurs in 2 files" in note for note in notes)


@pytest.mark.verifies("REQ-018")
def test_a_helper_name_collision_is_not_noted(tmp_path):
    # Only a test name can be looked up by name alone, so a collision between
    # two module-level helpers is not reported.
    (tmp_path / "test_a.py").write_text("def _helper():\n    assert 1 == 1\n", encoding="utf-8")
    (tmp_path / "test_b.py").write_text("def _helper():\n    pass\n", encoding="utf-8")
    index, notes = index_test_files(tmp_path)
    assert ("", "_helper") not in index
    assert notes == ()


@pytest.mark.verifies("REQ-018")
def test_index_test_files_notes_an_unparseable_file(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_x(:\n", encoding="utf-8")
    index, notes = index_test_files(tmp_path)
    assert index == {}
    assert any("not parseable Python" in note for note in notes)


@pytest.mark.verifies("REQ-018")
def test_the_hand_labelled_corpus_reproduces_the_published_confusion_matrix(heuristic_corpus):
    # Hand labels: 8 functions verify nothing, 15 verify something, total 23.
    # Expected, from reading the rules against the labels before running them:
    #   tp = 5 (empty and flagged), fp = 3 (asserts via a helper, flagged),
    #   fn = 3 (vacuous but syntactically non-constant), tn = 12.
    labels = json.loads(
        (heuristic_corpus / "labels.json").read_text(encoding="utf-8")
    )["labels"]
    index, _ = index_test_files(heuristic_corpus)
    verdicts = {
        name: verdict.has_assertion
        for (stem, name), verdict in index.items()
        if stem and name.startswith("test_")
    }
    assert set(verdicts) == set(labels)
    tp = fp = fn = tn = 0
    for name, label in labels.items():
        flagged = not verdicts[name]
        truly_empty = not label["asserts_something"]
        if flagged and truly_empty:
            tp += 1
        elif flagged and not truly_empty:
            fp += 1
        elif truly_empty:
            fn += 1
        else:
            tn += 1
    assert (tp, fp, fn, tn) == (5, 3, 3, 12)
    assert tp + fp + fn + tn == 23


@pytest.mark.verifies("REQ-018")
def test_the_three_known_false_positives_are_the_delegating_tests(heuristic_corpus):
    index, _ = index_test_files(heuristic_corpus)
    flagged = {
        name for (stem, name), verdict in index.items()
        if stem == "test_assertion_elsewhere" and name.startswith("test_")
        and not verdict.has_assertion
    }
    assert flagged == {
        "test_delegates_to_module_helper",
        "test_delegates_to_fixture",
        "test_delegates_to_verifier_object",
    }


@pytest.mark.verifies("REQ-018")
def test_the_three_known_false_negatives_are_the_vacuous_asserts(heuristic_corpus):
    index, _ = index_test_files(heuristic_corpus)
    missed = {
        name for (stem, name), verdict in index.items()
        if stem == "test_truly_empty" and name.startswith("test_") and verdict.has_assertion
    }
    assert missed == {
        "test_vacuous_self_comparison",
        "test_always_true_predicate",
        "test_assert_short_circuited_by_true",
    }
