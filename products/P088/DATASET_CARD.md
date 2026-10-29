# Dataset card — the seeded synthetic dictionary generator

This product has no AI component and ships no data file. It does have a
dataset, in the sense that every number in `README.md`, `CHANGELOG.md` and
`validation/` is measured against one: the generator in
`src/tmtcdiff/synth.py`. This card describes it, because a measurement whose
data nobody can characterise is not a measurement.

## What it generates

- **Dictionaries.** `baseline_dictionary(size, seed)` produces `size`
  telemetry parameters and `max(3, size // 4)` commands. Parameters are drawn
  from 10 quantity templates (temperature, voltage, current, pressure, angular
  rate, quaternion component, counter, elapsed time, a float temperature, and
  a one-bit flag), 6 subsystem prefixes, 16 location tokens and 5 status
  enumerations. One parameter in six carries an enumeration. Mnemonics are
  built as `<subsystem>_<token>_<location>_<index>` and are unique through the
  trailing index.
- **Changed copies.** `inject_changes(old, classes, seed)` applies one change
  of each named class to an item no other injection touches, and records the
  exact `(class, item)` ground truth. All 21 change classes are injectable.
- **Corpora.** `synthetic_corpus(dictionary, seed, names=...)` produces
  procedure files, analysis scripts and configuration files that reference
  named items by name, by path, by enumeration label and by command argument,
  plus a configurable number of references assembled by run-time expressions.
- **The negative control.** `reordering_control(dictionary, seed)`
  re-serialises one dictionary with its entries, keys, enumerations and
  command-argument declarations shuffled and most key names respelled to
  accepted aliases, producing a document that differs on essentially every
  line while carrying identical information.

## Provenance

Entirely synthetic, generated at run time from `random.Random(seed)`. **No
flight database is used, read, imported or approximated anywhere in this
repository.** No mnemonic, bit layout, calibration constant, unit, enumeration
or limit here is taken from any mission, any published database, or any
vendor's format. The quantity templates are plausible-looking round numbers
chosen by hand.

## Determinism and reproducibility

Every generator function takes a `seed` and uses `random.Random` alone. No
global random state is touched and no numpy generator is used, so a seed
produces the same output on any machine and across numpy versions. Seeds are
written in the script that uses them and are printed in every validation
output. Nothing is cached and no generated artifact is committed; regenerating
is the only way the data exists.

## Size and compute

A 400-parameter dictionary with 100 commands is the largest generated
anywhere in this repository. The whole validation suite generates a few
thousand dictionaries and completes in under a minute on two cores, apart
from `validation/probe_working_directories.py`, which runs the suite five
times over.

## Known biases and limitations of this dataset

These are the reasons a detection rate measured here does not transfer
unchanged to a real database.

- **Ten templates is not a real dictionary.** A real mission database has
  hundreds of distinct field layouts, inconsistent naming conventions
  accumulated over years, and parameters whose descriptions are blank or
  wrong. The generator's descriptions are always present and always
  well-formed, which inflates the description term of the rename score.
- **One injected change per item.** Injections never collide on the same
  item, so the measured per-class figures say nothing about an item that was
  rescaled *and* bit-shifted *and* renamed at once.
- **Five rename transforms.** Real renames are more varied, and the five here
  were chosen to span a range of textual similarity rather than sampled from
  anything. Rename recall is a property of that choice as much as of the
  matcher.
- **Byte-aligned, wrapping bit offsets.** This is deliberate and it is the
  single most consequential property of the generator. The first version gave
  every parameter a globally unique bit offset; every structural signature was
  then unique, every rename was trivially separable, and the measured
  false-match rate was exactly zero at every threshold from 0.40 to 0.95 —
  a broken experiment rather than a result. Offsets now wrap inside each
  packet and 24 of 400 parameters share an exact signature at the largest
  swept size.
- **No malformed input.** Every generated document parses. The parser's error
  paths are exercised by hand-written tests in `tests/test_parse.py`, not by
  this generator.
- **No real corpus.** The synthetic corpora use a small, regular set of
  reference syntaxes. Real procedures and analysis scripts reach a dictionary
  in ways this package's extractor does not recognise, which is why the
  fraction of dictionary items a corpus never mentions is a returned field of
  the coverage verdict rather than an assumption.

## Licence and use

The generator is part of this package and is covered by its Apache-2.0
licence. It produces research-grade synthetic data for measuring this
package's own behaviour. It is not a mission database, it is not a substitute
for one, and nothing generated by it is flight-qualified, certified, or
approved for operational aerospace use.
