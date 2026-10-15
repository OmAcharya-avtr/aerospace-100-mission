"""Exception types raised by hazardtrace.

Two classes only. ``AnalysisFormatError`` means the declared analysis could not
be read as an STPA artifact at all, and it always names the offending YAML key
path so the author can go straight to the line. Everything that *can* be read
but is incomplete is a finding, not an exception, because incompleteness is the
product's output rather than an error.
"""

from __future__ import annotations


class HazardTraceError(Exception):
    """Base class for every error raised by this package."""


class AnalysisFormatError(HazardTraceError):
    """The YAML document is not a well-formed STPA analysis.

    Parameters
    ----------
    message : str
        What is wrong, in the imperative where a fix is possible.
    key_path : str
        Dotted path of the offending key, for example
        ``ucas[2].uca_class``. Empty only for a whole-document failure.
    """

    def __init__(self, message: str, key_path: str = "") -> None:
        self.message = message
        self.key_path = key_path
        if key_path:
            super().__init__(f"{key_path}: {message}")
        else:
            super().__init__(message)
