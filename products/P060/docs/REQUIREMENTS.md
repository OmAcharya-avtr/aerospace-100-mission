# traceaudit — numbered requirements

These are the requirements of `traceaudit` itself, in the format `traceaudit`
parses by default. They exist so that the tool can be run against its own
repository, which is the worked example in README.md and the check in
`validation/validate_self_trace.py`.

**What this document is not.** It is not a specification agreed with anybody,
it carries no verification method, no review state, no baseline and no
approval. It is a list of statements about this package that each have at
least one test, and the audit checks exactly that and nothing more.

Declaration format: an id at the start of a line, behind a markdown heading
marker. A reference to an id inside a sentence is not a declaration, which is
why the prose below says things like "the counterpart of the previous one"
rather than repeating the id.

Each requirement is claimed by at least one test through
`@pytest.mark.verifies("REQ-NNN")`, recorded into junit XML by
`tests/conftest.py` using `traceaudit.markers.record_claims`.

## Parsing requirements from markdown

### REQ-001 Declaration forms

The requirement parser shall recognise a declaration written as a markdown
heading, as a list bullet, or as bold text at the start of a line, and shall
record the id, the trailing title, the source label and the one-based line
number.

### REQ-002 References are not declarations

The requirement parser shall not create a declaration from an id that appears
anywhere other than the start of a line.

### REQ-003 Fenced code blocks

The requirement parser shall ignore lines inside fenced code blocks when
configured to do so, and shall include them when not.

### REQ-004 Duplicate declarations retained

The requirement parser shall retain every declaration of a repeated id and
shall expose them grouped by id.

## Parsing test reports

### REQ-005 Outcome classification from junit XML

The junit parser shall classify each test case as passed, failed, errored,
skipped or xfailed, distinguishing a skip from an expected failure by the
`type` attribute of the `skipped` element.

### REQ-006 Claims from junit properties

The junit parser shall read claimed requirement ids from the configured
`property` names on a test case, splitting several ids in one value on the
configured separators.

### REQ-007 Claims from the node id

The report parsers shall optionally read claimed requirement ids out of the
test node id using a configurable pattern and template.

### REQ-008 Collection reports carry no outcomes

The collection-report parser shall accept the output of pytest's quiet
collection mode, shall assign every case an unknown outcome, and the audit
shall state that the outcome-dependent checks were not computed.

### REQ-009 Header totals cross-checked

The audit shall compare the totals declared in the `testsuite` element against
the outcomes counted from the test cases and shall report a note when they
disagree.

## The mapping and its coverage arithmetic

### REQ-010 Bidirectional mapping

The audit shall compute both directions of the mapping: declared id to
claiming test node ids, and test node id to claimed ids, including claims
whose id is not declared.

### REQ-011 Coverage with denominators

The audit shall report nominal, executed and passing coverage, each as an
explicit numerator over an explicit denominator, where the denominator is the
number of unique declared ids, and shall report the executed and passing
figures as undefined when the report carries no outcomes.

### REQ-012 Coverage ordering

Nominal coverage shall never be less than executed coverage, and executed
coverage shall never be less than passing coverage, for any input.

## Findings

### REQ-013 Untraced requirement

The audit shall report finding TA001 for every declared id that no test
claims.

### REQ-014 Undeclared claim

The audit shall report finding TA002 for every claim whose id has no
declaration, naming the claiming test.

### REQ-015 Duplicate id

The audit shall report finding TA003 for every id declared more than once,
naming every declaration location.

### REQ-016 Illusory coverage

The audit shall report finding TA004 for every requirement whose every
claiming test was skipped or xfailed.

### REQ-017 Failing-only trace

The audit shall report finding TA005 for every requirement whose every
claiming test failed or errored.

### REQ-018 Assertion heuristic

The audit shall report finding TA006 for every claiming test in which no
assertion is detected, shall mark the finding as heuristic, and shall not
report it when no test source root is supplied.

### REQ-019 Ignorable codes

The audit shall accept a set of codes that are reported but excluded from the
exit-status calculation, and shall mark them as ignored.

## Interfaces

### REQ-020 Exit statuses

The command line interface shall exit 0 when no blocking finding was reported,
1 when at least one was, and 2 when the audit could not run.

### REQ-021 Configuration

The configuration shall be loadable from a JSON file, shall reject unknown
keys by name, and shall reject an invalid regular expression or an
unformattable template with an actionable message.

### REQ-022 Input validation

Every public entry point shall raise a specific exception with an actionable
message for a missing file, an unparseable document and an unparseable report.

### REQ-023 Machine-readable output

The audit shall emit a JSON document containing the declarations, both
directions of the mapping, every coverage figure with its numerator and
denominator, and every finding.

### REQ-024 Markdown traceability table

The audit shall emit a markdown table of requirement, title, claiming tests
and outcomes.

### REQ-025 Marker recording helper

The package shall provide a helper that copies requirement ids from a pytest
marker onto a test item's user properties, accepting one id, several ids, or a
sequence of ids, and recording each id once.

### REQ-026 Bookkeeping notice

Every rendered report shall carry the statement that traceability is a
necessary bookkeeping condition and not evidence of adequacy.
