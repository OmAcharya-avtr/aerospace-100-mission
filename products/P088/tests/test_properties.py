"""Property-based tests for the algebraic identities this package relies on.

Every property here is an identity that must hold for any input, not a
sampling of plausible ones. Where a property involves floating point it is
stated with a tolerance on both sides and never as an equality of a boolean
produced by a strict threshold comparison: a boolean is a step function of a
rounded quantity and is the one thing floating point cannot promise.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. Any counterexample found here is pinned as an explicit
regression test in ``tests/test_regressions.py`` with its literal inputs.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tmtcdiff.changes import Change, ChangeClass, Severity, rank_changes
from tmtcdiff.diff import DiffConfig, diff_dictionaries
from tmtcdiff.metrics import clopper_pearson, per_class_metrics
from tmtcdiff.model import Dictionary, Parameter
from tmtcdiff.parse import loads, round_trip
from tmtcdiff.rename import (
    DEFAULT_WEIGHTS,
    match_renames,
    normalise_name,
    pair_score,
    structural_similarity,
    text_similarity,
)
from tmtcdiff.synth import baseline_dictionary, reordering_control

EXACT = DiffConfig(numeric_rel_tol=0.0)

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=18,
).filter(lambda text: text.strip() != "")

finite_scales = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
).filter(lambda value: value != 0.0)

finite_offsets = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def parameters(draw: st.DrawFn, name: str | None = None) -> Parameter:
    """A valid parameter with no enumeration, so widths and codes cannot conflict."""
    return Parameter(
        name=name if name is not None else draw(names),
        path=draw(st.sampled_from(["", "TM/A/X", "TM/B/Y"])),
        description=draw(st.text(max_size=30)),
        data_type=draw(st.sampled_from(["uint", "int", "float"])),
        bit_offset=draw(st.integers(min_value=0, max_value=2048)),
        bit_width=draw(st.integers(min_value=1, max_value=64)),
        byte_order=draw(st.sampled_from(["big", "little"])),
        scale=draw(finite_scales),
        offset=draw(finite_offsets),
        unit=draw(st.sampled_from(["", "K", "V", "A", "Pa"])),
        packet=draw(st.sampled_from(["", "A_HK", "B_HK"])),
    )


@st.composite
def dictionaries(draw: st.DrawFn, min_size: int = 1, max_size: int = 6) -> Dictionary:
    """A valid dictionary whose parameter names are unique by construction."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    chosen = draw(st.lists(names, min_size=count, max_size=count, unique=True))
    return Dictionary(
        version=draw(st.sampled_from(["1.0", "2.3", "x"])),
        parameters=tuple(draw(parameters(name=name)) for name in chosen),
    )


class TestNameNormalisation:
    @given(names)
    def test_idempotent(self, name: str) -> None:
        once = normalise_name(name)
        assert normalise_name(once) == once

    @given(names)
    def test_upper_case(self, name: str) -> None:
        assert normalise_name(name) == normalise_name(name).upper()


class TestSimilarityRanges:
    @given(st.text(max_size=40), st.text(max_size=40))
    def test_text_similarity_lies_in_the_unit_interval(self, left: str, right: str) -> None:
        assert 0.0 <= text_similarity(left, right) <= 1.0

    @given(st.text(min_size=1, max_size=40))
    def test_text_similarity_of_a_string_with_itself_is_one(self, text: str) -> None:
        assert text_similarity(text, text) == 1.0

    @given(parameters())
    def test_structural_similarity_with_itself_is_one(self, parameter: Parameter) -> None:
        assert structural_similarity(parameter, parameter) == 1.0

    @given(parameters(), parameters())
    def test_structural_similarity_lies_in_the_unit_interval(
        self, left: Parameter, right: Parameter
    ) -> None:
        assert 0.0 <= structural_similarity(left, right) <= 1.0

    @given(parameters(), parameters())
    def test_structural_similarity_is_symmetric(
        self, left: Parameter, right: Parameter
    ) -> None:
        assert structural_similarity(left, right) == structural_similarity(right, left)

    @given(parameters(), names)
    def test_a_rename_leaves_the_structural_term_untouched(
        self, parameter: Parameter, new_name: str
    ) -> None:
        renamed = parameter.with_fields(name=new_name)
        assert structural_similarity(parameter, renamed) == 1.0


class TestPairScore:
    @given(parameters(), parameters())
    def test_score_is_the_declared_weighted_sum(
        self, left: Parameter, right: Parameter
    ) -> None:
        score, structural, name, description = pair_score(left, right)
        expected = (
            DEFAULT_WEIGHTS["structural"] * structural
            + DEFAULT_WEIGHTS["name"] * name
            + DEFAULT_WEIGHTS["description"] * description
        )
        assert score == pytest_approx(expected)

    @given(parameters(), parameters())
    def test_score_lies_in_the_unit_interval(self, left: Parameter, right: Parameter) -> None:
        score, _, _, _ = pair_score(left, right)
        assert 0.0 <= score <= 1.0

    @given(parameters())
    def test_an_item_against_itself_scores_one(self, parameter: Parameter) -> None:
        assume(parameter.description != "")
        score, _, _, _ = pair_score(parameter, parameter)
        assert score == pytest_approx(1.0)


def pytest_approx(expected: float) -> object:
    """``pytest.approx`` with an explicit zero absolute tolerance.

    ``pytest.approx(x, rel=r)`` also passes when the absolute difference is
    under its hidden 1e-12 default, so a relative bound tighter than that is
    silently replaced for small quantities. Every comparison in this file
    passes ``abs=0.0`` so the relative bound is the one that acts.
    """
    import pytest

    return pytest.approx(expected, rel=1e-12, abs=0.0)


class TestEngineeringValue:
    @given(parameters(), st.integers(min_value=-(2**20), max_value=2**20))
    def test_affine_increment(self, parameter: Parameter, raw: int) -> None:
        # eng(raw + 1) - eng(raw) = scale, in exact arithmetic. Compared with a
        # relative tolerance on both sides, never as an equality.
        increment = parameter.engineering(raw + 1) - parameter.engineering(raw)
        assert math.isclose(increment, parameter.scale, rel_tol=1e-6, abs_tol=1e-9)

    @given(parameters())
    def test_zero_raw_gives_the_offset(self, parameter: Parameter) -> None:
        assert parameter.engineering(0) == pytest_approx(parameter.offset)


class TestParseRoundTrip:
    @given(dictionaries())
    @SLOW
    def test_a_dictionary_survives_a_round_trip(self, dictionary: Dictionary) -> None:
        assert round_trip(dictionary).parameters == dictionary.parameters

    @given(dictionaries())
    @SLOW
    def test_a_round_tripped_dictionary_diffs_empty(self, dictionary: Dictionary) -> None:
        assert diff_dictionaries(dictionary, round_trip(dictionary), EXACT).changes == ()


class TestDiffIdentities:
    @given(dictionaries())
    @SLOW
    def test_a_dictionary_against_itself_reports_nothing(self, dictionary: Dictionary) -> None:
        assert diff_dictionaries(dictionary, dictionary, EXACT).changes == ()

    @given(dictionaries(min_size=2, max_size=6))
    @SLOW
    def test_cardinality_is_conserved(self, dictionary: Dictionary) -> None:
        trimmed = Dictionary(version="1.1", parameters=dictionary.parameters[1:])
        result = diff_dictionaries(dictionary, trimmed, EXACT)
        counts = result.class_counts
        predicted = (
            len(dictionary.parameters)
            + counts.get(ChangeClass.ADDED, 0)
            - counts.get(ChangeClass.REMOVED, 0)
        )
        assert predicted == len(trimmed.parameters)

    @given(dictionaries())
    @SLOW
    def test_changes_are_ranked_silent_first(self, dictionary: Dictionary) -> None:
        changed = Dictionary(
            version="1.1",
            parameters=tuple(
                parameter.with_fields(unit=parameter.unit + "x")
                for parameter in dictionary.parameters
            ),
        )
        result = diff_dictionaries(dictionary, changed, EXACT)
        ranks = [change.severity.rank for change in result.changes]
        assert ranks == sorted(ranks)

    @given(dictionaries())
    @SLOW
    def test_every_change_is_filed_under_an_item_of_one_of_the_two_versions(
        self, dictionary: Dictionary
    ) -> None:
        trimmed = Dictionary(version="1.1", parameters=dictionary.parameters[:-1])
        result = diff_dictionaries(dictionary, trimmed, EXACT)
        known = set(dictionary.item_names) | set(trimmed.item_names)
        assert all(change.name in known for change in result.changes)


class TestReorderingControl:
    @given(st.integers(min_value=4, max_value=40), st.integers(min_value=0, max_value=10_000))
    @SLOW
    def test_a_reordered_document_has_an_empty_semantic_diff(
        self, size: int, seed: int
    ) -> None:
        dictionary = baseline_dictionary(size, seed)
        _, reordered = reordering_control(dictionary, seed + 1)
        assert diff_dictionaries(dictionary, loads(reordered), EXACT).changes == ()


class TestRanking:
    @given(
        st.lists(
            st.tuples(st.sampled_from(list(ChangeClass)), names),
            min_size=0,
            max_size=20,
        )
    )
    def test_ranking_sorts_by_tier(self, entries: list[tuple[ChangeClass, str]]) -> None:
        changes = tuple(
            Change(change_class=change_class, kind="parameter", name=name)
            for change_class, name in entries
        )
        ranks = [change.severity.rank for change in rank_changes(changes)]
        assert ranks == sorted(ranks)

    @given(
        st.lists(
            st.tuples(st.sampled_from(list(ChangeClass)), names),
            min_size=0,
            max_size=20,
            unique=True,
        )
    )
    def test_ranking_is_independent_of_input_order(
        self, entries: list[tuple[ChangeClass, str]]
    ) -> None:
        changes = [
            Change(change_class=change_class, kind="parameter", name=name)
            for change_class, name in entries
        ]
        assert rank_changes(tuple(changes)) == rank_changes(tuple(reversed(changes)))

    @given(st.sampled_from(list(ChangeClass)))
    def test_a_class_is_silent_exactly_when_its_tier_is_silent(
        self, change_class: ChangeClass
    ) -> None:
        assert change_class.is_silent == (change_class.severity is Severity.SILENT)


class TestRenameMatching:
    @given(st.lists(parameters(), min_size=0, max_size=5, unique_by=lambda p: p.name),
           st.lists(parameters(), min_size=0, max_size=5, unique_by=lambda p: p.name),
           st.floats(min_value=0.0, max_value=1.0))
    @SLOW
    def test_matching_is_one_to_one(
        self, removed: list[Parameter], added: list[Parameter], threshold: float
    ) -> None:
        candidates = match_renames(removed, added, threshold=threshold)
        assert len({c.old_name for c in candidates}) == len(candidates)
        assert len({c.new_name for c in candidates}) == len(candidates)

    @given(st.lists(parameters(), min_size=1, max_size=5, unique_by=lambda p: p.name),
           st.lists(parameters(), min_size=1, max_size=5, unique_by=lambda p: p.name),
           st.floats(min_value=0.0, max_value=1.0))
    @SLOW
    def test_acceptance_is_exactly_the_threshold_rule(
        self, removed: list[Parameter], added: list[Parameter], threshold: float
    ) -> None:
        for candidate in match_renames(removed, added, threshold=threshold):
            assert candidate.accepted == (candidate.score >= threshold)

    @given(st.lists(parameters(), min_size=0, max_size=5, unique_by=lambda p: p.name),
           st.lists(parameters(), min_size=0, max_size=5, unique_by=lambda p: p.name))
    @SLOW
    def test_the_candidate_count_cannot_exceed_the_smaller_pool(
        self, removed: list[Parameter], added: list[Parameter]
    ) -> None:
        candidates = match_renames(removed, added, threshold=0.0)
        assert len(candidates) <= min(len(removed), len(added))


class TestMetricIdentities:
    @given(
        st.lists(st.tuples(st.sampled_from(list(ChangeClass)), names), max_size=20),
        st.lists(st.tuples(st.sampled_from(list(ChangeClass)), names), max_size=20),
    )
    def test_counts_reconcile_with_the_inputs(
        self,
        truth: list[tuple[ChangeClass, str]],
        detected: list[tuple[ChangeClass, str]],
    ) -> None:
        metrics = per_class_metrics(truth, detected)
        assert sum(m.injected for m in metrics.values()) == len(truth)
        assert sum(m.detections for m in metrics.values()) == len(detected)

    @given(st.integers(min_value=0, max_value=200), st.integers(min_value=0, max_value=200))
    @SLOW
    def test_the_binomial_interval_brackets_the_point_estimate(
        self, successes: int, extra: int
    ) -> None:
        trials = successes + extra
        low, high = clopper_pearson(successes, trials)
        if trials == 0:
            assert (low, high) == (0.0, 0.0)
        else:
            assert low - 1e-12 <= successes / trials <= high + 1e-12

    @given(st.integers(min_value=0, max_value=50), st.integers(min_value=1, max_value=50))
    @SLOW
    def test_the_binomial_interval_is_ordered_and_inside_the_unit_interval(
        self, successes: int, extra: int
    ) -> None:
        low, high = clopper_pearson(min(successes, extra), extra)
        assert 0.0 <= low <= high <= 1.0
