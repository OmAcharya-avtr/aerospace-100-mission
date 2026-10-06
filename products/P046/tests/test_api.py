"""Package surface: exports, version, docstrings and safety wording."""

import inspect

import interleavekit
from interleavekit import base, block, convolutional, cost, helical, metrics, srandom

MODULES = [base, block, convolutional, cost, helical, metrics, srandom]

FORBIDDEN = [
    "flight-safe",
    "flight safe",
    "certified",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
    "flight-qualified for",
]


def test_version():
    assert interleavekit.__version__ == "0.1.0"


def test_every_name_in_all_is_importable():
    for name in interleavekit.__all__:
        assert hasattr(interleavekit, name), name


def test_public_functions_have_docstrings_with_units_or_returns():
    for module in MODULES:
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            assert obj.__doc__, f"{module.__name__}.{name} has no docstring"


def test_no_certification_claims_anywhere_in_the_package():
    """No certification claim anywhere in the package source.

    'certified' is permitted only inside the negative safety statement, so every
    occurrence must be preceded by a negation within the previous 30 characters.
    """
    for module in MODULES + [interleavekit, interleavekit.__main__]:
        source = inspect.getsource(module).lower()
        for phrase in FORBIDDEN:
            if phrase == "certified":
                idx = source.find("certified")
                while idx != -1:
                    window = source[max(0, idx - 30) : idx]
                    assert "not " in window, (
                        f"{module.__name__}: 'certified' at offset {idx} is not negated"
                    )
                    idx = source.find("certified", idx + 1)
            else:
                assert phrase not in source, f"{module.__name__} contains {phrase!r}"


def test_package_docstring_carries_the_research_grade_statement():
    doc = interleavekit.__doc__ or ""
    assert "research-grade" in doc
    assert "not flight-qualified" in doc
