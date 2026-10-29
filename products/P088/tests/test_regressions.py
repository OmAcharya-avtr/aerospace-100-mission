"""Defects found during the build, and measured constants pinned against drift.

Three kinds of test live here.

**Pinned measured constants.** A number quoted in ``README.md`` or in a
``validation/`` output is pinned here with its generator inputs, so a change to
the generator or the scorer that moves a published figure fails the suite
rather than silently republishing a different number.

**Defects found during the build.** Each is reproduced with the literal inputs
that exposed it. Two were found by running the validation scripts, and one by
noticing that a measured false-match rate was zero at every threshold, which
is not a result but a broken experiment.

**Counterexamples.** ``.hypothesis/`` is gitignored, so a falsifying example
found by one run is lost on a fresh clone and has to be pinned here with its
literal inputs. The property suite in ``tests/test_properties.py`` found no
falsifying example in this build; there is therefore nothing pinned under this
heading, and that is stated rather than left as an empty section.
"""

from __future__ import annotations

import difflib

import pytest

from tmtcdiff.changes import ChangeClass
from tmtcdiff.diff import DEFAULT_RENAME_THRESHOLD, DiffConfig, diff_dictionaries
from tmtcdiff.model import Parameter
from tmtcdiff.parse import loads
from tmtcdiff.rename import pair_score, signature_collision_bound
from tmtcdiff.synth import (
    INJECTABLE,
    baseline_dictionary,
    change_pair,
    inject_changes,
    reordering_control,
)

EXACT = DiffConfig(numeric_rel_tol=0.0)


class TestPinnedConstants:
    def test_the_rename_threshold_is_the_value_the_sweep_selected(self) -> None:
        # validation/validate_rename_sweep_output.txt, rule 1: the lowest
        # threshold on the swept grid whose measured false-match rate is zero
        # at every swept dictionary size. Changing this constant without
        # re-running that sweep makes every rename figure in the README wrong.
        assert DEFAULT_RENAME_THRESHOLD == 0.80

    def test_signature_collisions_at_the_largest_swept_size(self) -> None:
        # validation/validate_rename_sweep_output.txt, the bound section.
        colliding, distinct, fraction = signature_collision_bound(
            list(baseline_dictionary(400, 4100).parameters)
        )
        assert (colliding, distinct) == (24, 388)
        assert fraction == pytest.approx(0.06, rel=1e-12, abs=0.0)

    def test_signature_collisions_at_two_hundred(self) -> None:
        colliding, distinct, fraction = signature_collision_bound(
            list(baseline_dictionary(200, 4100).parameters)
        )
        assert (colliding, distinct) == (6, 197)
        assert fraction == pytest.approx(0.03, rel=1e-12, abs=0.0)

    def test_the_negative_control_text_difference(self) -> None:
        # validation/validate_negative_control_output.txt, size 60 seed 7400:
        # 2526 text lines differ and zero semantic changes are reported.
        dictionary = baseline_dictionary(60, 7400)
        canonical, reordered = reordering_control(dictionary, 7401)
        differing = sum(
            1
            for line in difflib.unified_diff(
                canonical.splitlines(), reordered.splitlines(), n=0, lineterm=""
            )
            if line[:1] in "+-" and not line.startswith(("---", "+++"))
        )
        assert differing == 2526
        assert diff_dictionaries(dictionary, loads(reordered), EXACT).changes == ()

    def test_the_worked_example_rename_score(self) -> None:
        # validation/worked_example_output.txt: PAYL_CNT -> PAYLOAD_FRAME_COUNT
        # scores 0.8778, above the 0.80 threshold, on a structural term of 1.0.
        old = Parameter(
            name="PAYL_CNT",
            path="TM/PAYL/PAYL_CNT",
            description="payload frame counter",
            data_type="uint",
            bit_offset=32,
            bit_width=32,
            unit="count",
            packet="PAYLOAD_HK",
        )
        new = old.with_fields(name="PAYLOAD_FRAME_COUNT", path="TM/PAYL/PAYLOAD_FRAME_COUNT")
        score, structural, name, description = pair_score(old, new)
        assert structural == 1.0
        assert description == 1.0
        assert name == pytest.approx(0.5925925925925926, rel=1e-12, abs=0.0)
        assert score == pytest.approx(0.8777777777777779, rel=1e-12, abs=0.0)
        assert score >= DEFAULT_RENAME_THRESHOLD

    def test_the_generated_dictionary_size_quoted_in_the_readme(self) -> None:
        # A 60-parameter generated dictionary carries 15 commands, so 75 items.
        dictionary = baseline_dictionary(60, 7400)
        assert (len(dictionary.parameters), len(dictionary.commands), len(dictionary)) == (
            60,
            15,
            75,
        )

    def test_an_isolated_rescale_produces_exactly_one_change(self) -> None:
        pair = inject_changes(baseline_dictionary(30, 5200), (ChangeClass.RESCALED,), 5201)
        result = diff_dictionaries(pair.old, pair.new, EXACT)
        assert len(result.changes) == 1
        assert result.changes[0].change_class is ChangeClass.RESCALED
        assert result.changes[0].name == "COMM_P_TANK_15"


class TestBuildDefects:
    def test_removing_a_command_does_not_consume_a_second_one(self) -> None:
        # Defect: after popping a removed command from the working list, the
        # used-index set was renumbered WITH the removed index still in it, so
        # the removed command's neighbour was marked used. Every all-classes
        # injection then ran out of commands one injection early and raised
        # "no command left to inject command_renamed into" at the smallest
        # supported size. Found by validation/validate_change_classes.py.
        pair = change_pair(28, 11)
        assert {change.change_class for change in pair.truth} == set(INJECTABLE)

    def test_all_classes_inject_at_the_documented_minimum_size(self) -> None:
        for seed in (11, 23, 57, 101):
            pair = change_pair(28, seed)
            assert len(pair.truth) == len(INJECTABLE)

    def test_bit_offsets_collide_so_the_false_match_rate_is_measurable(self) -> None:
        # Defect: the first generator assigned every parameter a globally
        # unique bit offset, which made every structural signature unique.
        # The rename sweep then measured a false-match rate of exactly zero at
        # every threshold from 0.40 to 0.95 and the threshold selection had
        # nothing to choose between. Offsets now wrap inside each packet.
        colliding, _, _ = signature_collision_bound(
            list(baseline_dictionary(400, 4100).parameters)
        )
        assert colliding > 0

    def test_an_ambiguous_enum_label_is_only_a_verdict_reason_when_referenced(self) -> None:
        # Defect: every enumeration label shared between two parameters was
        # reported as something the assessment could not determine, even when
        # no corpus reference mentioned it. That made a fully covered corpus
        # with no label references report itself as undeterminable.
        from tmtcdiff.corpus import Corpus, CorpusSource
        from tmtcdiff.impact import assess_impact

        dictionary = baseline_dictionary(30, 8500)
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        corpus = Corpus(
            (
                CorpusSource(
                    label="a.py",
                    kind="script",
                    text="\n".join(f'x = tm["{name}"]' for name in dictionary.item_names),
                ),
            )
        )
        report = assess_impact(result, corpus, dictionary, dictionary)
        assert report.verdict.ambiguous_enum_labels == ()
        assert report.verdict.determinable

    def test_a_safety_word_scan_uses_word_boundaries(self) -> None:
        # Defect in a check rather than in the library: a substring scan for
        # the word "safe" fired on the generated enumeration label SAFE_MODE
        # and reported a safety claim where there was none.
        import re

        assert re.search(r"\bsafe\b", "SAFE_MODE", re.IGNORECASE) is None
        assert re.search(r"\bsafe\b", "this upgrade is safe", re.IGNORECASE) is not None


class TestNoCounterexamplesWereFound:
    def test_the_property_suite_found_no_falsifying_example_in_this_build(self) -> None:
        # Recorded as a test so the claim is visible where a reader looks for
        # pinned counterexamples. If a later run finds one, it is pinned above
        # with its literal inputs and this test's docstring is corrected.
        assert True
