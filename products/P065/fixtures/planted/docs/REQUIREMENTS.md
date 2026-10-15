# Requirements fixture for the planted-gaps case

Hand-written. Three declarations, in the format `hazardtrace` parses by
default. `examples/cases/planted_gaps.yaml` cites `REQ-201` (declared here) and
`REQ-299` (deliberately not declared), so HT012 fires exactly once.

A mention of REQ-299 in the middle of a sentence, as in this one, is not a
declaration, which is why the line above does not accidentally declare it.

### REQ-201 Planted requirement, declared

The planted system shall do the thing the planted mitigation claims it does.

### REQ-202 Planted requirement, declared and uncited

Declared here and cited by nothing, so that the requirement index is not a
one-element set.

### REQ-203 Planted requirement, declared and uncited

As above.
