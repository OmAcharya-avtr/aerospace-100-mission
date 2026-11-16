"""Extracting numeric claims from committed text, and aligning two versions of it.

A committed number in a validation output file is a reproducibility claim. To
decide whether that claim survives a seed change, two texts have to be compared
in a way that distinguishes three outcomes, which a plain text diff collapses
into one:

- **a number moved** -- the surrounding words are identical and a numeric token
  in that position changed;
- **the structure moved** -- the surrounding words themselves changed, so there
  is no pair of numbers to compare and the claim cannot be evaluated;
- **nothing moved**.

A plain `diff` reports the first two identically, and reports the first as a
whole changed line rather than as a changed quantity. A numeric-only comparison
has the opposite defect: it is blind to a change that is entirely in the words,
such as a set-iteration order printed as a key name.

So both are compared. Lines are reduced to a **skeleton** -- the line with every
numeric token replaced by a placeholder -- and the two skeleton sequences are
aligned with :class:`difflib.SequenceMatcher`. Inside an aligned pair the
numeric tokens are compared pairwise and positionally. Lines that do not align
are counted as structural change and are never silently dropped.

Tolerance
---------
Comparison is relative-or-absolute with **both bounds explicit**, because
``pytest.approx``'s hidden ``abs=1e-12`` default silently replaced every
relative tolerance tighter than 2e-8 for quantities of order 1e-5 in an earlier
product of this mission. The default here is ``rel_tol = 0.0`` and
``abs_tol = 0.0``: a committed number that changes in its last printed digit
HAS moved, and softening that by default would be the product tuning away its
own findings. A caller that wants slack states it.

Units: this module is unit-agnostic. It compares the decimal representations
that a script chose to print, and reports the token text alongside every value
so that a reader can see the precision the claim was made at.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

#: One numeric token: optional sign, digits with optional fractional part or a
#: leading point, and an optional exponent. Ordered so the longest match wins.
#: A token is required not to be preceded or followed by a word character, so
#: that identifiers such as ``w64_d050`` and ``P100`` do not decompose into
#: numbers; a run of digits inside a name is not a reported quantity.
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"[+-]?"
    r"(?:\d+\.\d*|\.\d+|\d+)"
    r"(?:[eE][+-]?\d+)?"
    r"(?![\w.])"
)

#: What a numeric token is replaced by when a line is reduced to its skeleton.
#:
#: Printable ASCII on purpose. The first version used ``"\x00N\x00"``, which is
#: unambiguous but put NUL bytes into every committed validation output that
#: quoted a skeleton, making the evidence a binary file. Committed evidence has
#: to be readable text, so the placeholder is printable.
#:
#: The cost is a collision: a report line that literally contains ``<#>`` is
#: treated as though it held a number at that position. That is a declared
#: limitation rather than a hidden one, and its only effect is on line
#: alignment, never on whether a numeric token is read as a number.
PLACEHOLDER = "<#>"


@dataclass(frozen=True)
class Token:
    """One numeric token found in a line of text.

    Attributes:
        line: zero-based line index within the text.
        index: zero-based position of this token among the tokens of its line.
        text: the token exactly as printed, so the claimed precision is visible.
        value: the token parsed as a float.
        start: zero-based character offset of the token within its line.
    """

    line: int
    index: int
    text: str
    value: float
    start: int


@dataclass(frozen=True)
class MovedNumber:
    """A numeric claim that changed between two texts.

    Attributes:
        line: zero-based line index in the reference text.
        index: position of the token among the tokens of its line.
        skeleton: the line with its numbers replaced, identical in both texts,
            which is what makes the two tokens comparable at all.
        before: the token text as committed.
        after: the token text as re-measured.
        before_value: the committed value as a float.
        after_value: the re-measured value as a float.
        absolute_change: ``abs(after_value - before_value)``.
        relative_change: ``absolute_change / abs(before_value)``, or ``inf``
            when the committed value is 0 and the new one is not.
    """

    line: int
    index: int
    skeleton: str
    before: str
    after: str
    before_value: float
    after_value: float
    absolute_change: float
    relative_change: float


@dataclass
class Comparison:
    """The result of comparing a committed text against a re-measured one.

    Attributes:
        moved: every numeric claim that changed, in reading order.
        reference_tokens: how many numeric tokens the committed text held.
        compared_tokens: how many of those were aligned to a counterpart and
            therefore actually evaluated. The difference from
            ``reference_tokens`` is the part of the claim the comparison could
            not reach, and it is reported rather than implied.
        structural_lines_reference: committed lines with no aligned counterpart.
        structural_lines_candidate: re-measured lines with no counterpart.
        identical: whether the two texts are byte-identical.
    """

    moved: list[MovedNumber] = field(default_factory=list)
    reference_tokens: int = 0
    compared_tokens: int = 0
    structural_lines_reference: int = 0
    structural_lines_candidate: int = 0
    identical: bool = False

    @property
    def moved_count(self) -> int:
        """Number of numeric claims that moved, dimensionless."""
        return len(self.moved)

    @property
    def unreachable_tokens(self) -> int:
        """Committed tokens the alignment could not pair, dimensionless."""
        return self.reference_tokens - self.compared_tokens

    @property
    def structural_change(self) -> bool:
        """Whether any line failed to align, in either direction."""
        return bool(self.structural_lines_reference or self.structural_lines_candidate)


def numeric_tokens(line: str, line_number: int = 0) -> list[Token]:
    """Return every numeric token in one line, in reading order.

    Args:
        line: the text of one line, without its newline.
        line_number: the zero-based index to record on each token.

    Returns:
        The tokens, left to right.
    """
    if not isinstance(line, str):
        raise TypeError(f"line must be a str, got {type(line).__name__}")
    out: list[Token] = []
    for position, match in enumerate(_NUMBER.finditer(line)):
        text = match.group(0)
        try:
            value = float(text)
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
        out.append(
            Token(line=line_number, index=position, text=text, value=value, start=match.start())
        )
    return out


def skeleton(line: str) -> str:
    """Return ``line`` with every numeric token replaced by :data:`PLACEHOLDER`.

    Two lines with the same skeleton are reporting the same quantities with
    possibly different values, which is exactly the pair worth comparing.
    """
    if not isinstance(line, str):
        raise TypeError(f"line must be a str, got {type(line).__name__}")
    return _NUMBER.sub(PLACEHOLDER, line)


def changed(
    before: float,
    after: float,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> bool:
    """Whether ``after`` differs from ``before`` beyond the stated tolerances.

    Both tolerances are explicit and both default to 0. A value is unchanged
    when ``abs(after - before) <= max(abs_tol, rel_tol * abs(before))``.

    ``nan`` is handled explicitly: ``nan`` against ``nan`` is unchanged, and
    ``nan`` against anything else is changed. Without that, every comparison
    involving a ``nan`` would silently report "unchanged", because every
    comparison with ``nan`` is false.

    Args:
        before: the committed value.
        after: the re-measured value.
        rel_tol: relative tolerance, dimensionless, non-negative.
        abs_tol: absolute tolerance, in the units of the values, non-negative.

    Returns:
        True when the value moved.
    """
    rel = float(rel_tol)
    absolute = float(abs_tol)
    if rel < 0.0 or absolute < 0.0:
        raise ValueError(f"tolerances must be non-negative, got rel={rel!r}, abs={absolute!r}")
    first, second = float(before), float(after)
    if math.isnan(first) or math.isnan(second):
        return not (math.isnan(first) and math.isnan(second))
    if math.isinf(first) or math.isinf(second):
        return first != second
    delta = abs(second - first)
    return delta > max(absolute, rel * abs(first))


def compare(
    reference: str,
    candidate: str,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> Comparison:
    """Compare a committed text against a re-measured one.

    Args:
        reference: the committed text, treated as the claim.
        candidate: the re-measured text.
        rel_tol: relative tolerance for a numeric token, dimensionless.
        abs_tol: absolute tolerance for a numeric token.

    Returns:
        A :class:`Comparison`. A token pair is evaluated only inside a pair of
        lines whose skeletons are equal; everything else is counted as
        structural change, so the comparison never reports agreement it did not
        establish.
    """
    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise TypeError("reference and candidate must both be str")
    result = Comparison(identical=reference == candidate)
    ref_lines = reference.splitlines()
    cand_lines = candidate.splitlines()
    ref_tokens = [numeric_tokens(line, i) for i, line in enumerate(ref_lines)]
    cand_tokens = [numeric_tokens(line, i) for i, line in enumerate(cand_lines)]
    result.reference_tokens = sum(len(t) for t in ref_tokens)

    ref_skeleton = [skeleton(line) for line in ref_lines]
    cand_skeleton = [skeleton(line) for line in cand_lines]
    matcher = difflib.SequenceMatcher(a=ref_skeleton, b=cand_skeleton, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                i, j = i1 + offset, j1 + offset
                left, right = ref_tokens[i], cand_tokens[j]
                # Equal skeletons imply equal token counts: the placeholder
                # count is part of the skeleton. Asserted rather than assumed,
                # because a silent mismatch here would drop claims.
                if len(left) != len(right):  # pragma: no cover - unreachable
                    raise AssertionError("equal skeletons with unequal token counts")
                for token_a, token_b in zip(left, right, strict=True):
                    result.compared_tokens += 1
                    if changed(token_a.value, token_b.value, rel_tol, abs_tol):
                        delta = abs(token_b.value - token_a.value)
                        if token_a.value == 0.0:
                            relative = 0.0 if delta == 0.0 else math.inf
                        else:
                            relative = delta / abs(token_a.value)
                        result.moved.append(
                            MovedNumber(
                                line=i,
                                index=token_a.index,
                                skeleton=ref_skeleton[i],
                                before=token_a.text,
                                after=token_b.text,
                                before_value=token_a.value,
                                after_value=token_b.value,
                                absolute_change=delta,
                                relative_change=relative,
                            )
                        )
        else:
            result.structural_lines_reference += i2 - i1
            result.structural_lines_candidate += j2 - j1
    return result
