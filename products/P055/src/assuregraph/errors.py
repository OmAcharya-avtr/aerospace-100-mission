"""Exception hierarchy for assuregraph.

Every exception carries an actionable message: what was wrong, where it was
found (node id or YAML key path), and what the caller should do about it.
Nothing in this module makes a safety claim; these are format errors.
"""

from __future__ import annotations


class AssureGraphError(Exception):
    """Base class for every error raised by assuregraph."""


class CaseFormatError(AssureGraphError):
    """The case document is not a well-formed assurance case.

    Raised for YAML that parses but does not match the schema in
    :mod:`assuregraph.parse`: a missing required key, an unknown node type, a
    duplicate node id, a reference to a node id that does not exist, an edge
    whose relationship is not permitted between those two GSN node types, or a
    value of the wrong Python type.
    """

    def __init__(self, message: str, *, location: str | None = None) -> None:
        self.location = location
        if location is not None:
            message = f"{location}: {message}"
        super().__init__(message)


class CaseIOError(AssureGraphError):
    """The case document could not be read, or is not parsable as YAML."""
