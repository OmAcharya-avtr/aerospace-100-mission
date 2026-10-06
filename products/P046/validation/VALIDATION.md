# Validation evidence — interleavekit 0.1.0

Validation level 2 (research grade). Every number below was produced by running the
script named beside it in this directory, on Python 3.13.16, on two cores shared
with four concurrent builds, on 2026-10-06. The raw stdout of each run is committed
next to the script as `*_output.txt`. Nothing here is quoted from a reference; the
references are used only for the bibliographic details of the constructions and for
one documented cross-check of a delay constant.

Reproduce everything with the commands in the README section "Reproducing every
number".

## What is actually being checked

This library computes exact, deterministic combinatorics on index maps. There is no
physical measurement to agree with and no tolerance to set. So the checks are of
three kinds:

1. **A closed form against its definition.** The block interleaver's minimum-spread
   formula and the burst-dispersion fast path are both derived expressions that
   replace an enumeration. Each is checked against the enumeration it replaces, on
   every case in a stated range, and the only acceptable result is zero mismatches.
2. **An identity that must hold exactly.** De-interleaving must invert interleaving
   with no tolerance, on every construction, including at depth 1, span 1, step 0,
   slope 0 and length 1.
3. **A measurement reported as it came out.** The cost trade-off, the achievable
   S-random spread and the relationship between minimum spread and burst dispersion
   are measurements. Two of them contradict a figure in common circulation, and
   both are reported as measured, with the reason.

## Summary table

| # | Check | Reference | Result | Tolerance | Script |
|---|---|---|---|---|---|
| 1 | Block minimum-spread closed form | brute-force enumeration of all index pairs | **0 mismatches in 2468 (depth, span) pairs** with `depth*span <= 400` | exact | `validate_block_spread.py` |
| 2 | `deinterleave(interleave(x)) == x` and bijection | the definition of a de-interleaver | **0 failures in 1253 parameter sets** across all four constructions | exact, elementwise | `validate_roundtrip.py` |
| 3 | Burst-dispersion fast path | `burst_dispersion_by_window_scan`, the definitional window scan | **0 mismatches in 5304 (position map, burst length) comparisons** | exact | `validate_burst_metric_equivalence.py` |
| 4 | "Block depth = dispersed burst length" | measurement with `max_burst_fully_dispersed` | **false at span 1 and span 2; true from span 3 upward.** At span 2 the measured burst is exactly `depth - 1` for every depth in {4, 8, 16, 32, 64, 127} | exact | `validate_burst_dispersion.py` |
| 5 | `max_burst_fully_dispersed` equals `min|pos[j+1]-pos[j]|` | direct computation of the minimum adjacent transmitted gap | **0 mismatches in 24 block cases** | exact | `validate_burst_dispersion.py` |
| 6 | Convolutional pair latency constant | total delay documented for the MathWorks Communications Toolbox Convolutional Interleaver, `N x slope x (N-1)` (read 2026-10-06) | agrees; derived independently in `src/interleavekit/convolutional.py` and confirmed by running the pair, whose end-to-end delay is constant for every symbol | exact | `validate_roundtrip.py`, `tests/test_convolutional.py` |
| 7 | Cost to disperse a target burst, block vs shift-register bank | measurement; each candidate's burst dispersion measured, not derived from depth | bank needs **3.333x to 3.939x less pair memory** over targets 4 to 64 symbols, and the same ratio in latency. **Larger than the factor of 2 usually quoted** — see "Where this contradicts folklore" | exact under the two stated cost models | `validate_cost_tradeoff.py` |
| 8 | Minimum spread as a proxy for burst dispersion | measurement on a fixed 16x16 array at constant memory | **Spearman rho = 0.0343** over 17 helical steps; **rho = -0.4857** across six constructions at equal 512-symbol pair memory. Choosing on minimum spread costs **8.47x less fully dispersed burst at identical memory** | exact | `validate_spread_vs_burst.py` |
| 9 | S-random achieves the spread it is asked for | `s_parameter`, which re-derives the condition from the permutation | achieved S equals the request at lengths 64, 128, 256, 512, 1024 | exact | `validate_srandom.py` |
| 10 | `minimum_spread >= s_parameter + 1` | proved in `src/interleavekit/metrics.py` | holds in every case measured; also property-tested over random permutations | exact | `validate_srandom.py`, `tests/test_properties.py` |
| 11 | Largest S-random spread this search reaches | the `sqrt(length/2)` rule of thumb, as a yardstick only | **ratio falls from 1.06 at length 64 to 0.53 at length 1024.** This implementation does **not** attain the rule of thumb at long lengths — see "What failed or fell short" | reported, not a pass/fail | `validate_srandom.py` |
| 12 | S-random determinism | re-running the constructor | 6 of 6 seeds reproduced bit-identically; seeds 0..7 gave 8 distinct permutations | exact | `validate_srandom.py` |

## Where this contradicts folklore

**The block-interleaver depth rule (check 4).** The rule a designer is most likely to
apply — "set the depth to the burst length you must survive" — is wrong by one symbol
at span 2 and wrong by `depth - 1` symbols at span 1. The cause is specific: at span
2 the source index in the last column of a row is adjacent to the first column of the
next row, and those two symbols sit only `depth - 1` transmitted positions apart. A
designer who sizes a span-2 interleaver on the rule has a one-symbol error in exactly
the quantity the interleaver exists to provide. `cheapest_for_burst` finds
`depth = target + 1` at span 2 for this reason, which is visible in every row of
`cost_tradeoff_output.txt`.

**The factor of two (check 7).** The commonly quoted figure is that a convolutional
interleaver needs half the memory of a block interleaver for the same job. The
measured ratio here is 3.3 to 3.9. The arithmetic behind that is in
`cost_tradeoff_output.txt` and is not subtle: the cheapest block candidate for a
target burst `L` is depth `L+1` at span 2, so `N = 2L + 2` and full-block buffering
at both ends costs `4L + 4` symbols, while the cheapest bank is two registers of
slope about `L/2`, costing about `L + 2`. **The ratio depends on accepting full-block
buffering as the block cost model.** An implementation that overlaps read-out with
write-in pays less than `2N`, and against such an implementation the ratio would be
smaller. The number is reported under a stated model, not asserted as universal, and
`src/interleavekit/base.py` states the model where the figures are produced.

## What failed or fell short

**The S-random search does not reach the `sqrt(length/2)` rule of thumb at long
lengths (check 11).** Measured ratios: 1.00 at length 32, 1.06 at 64, 0.88 at 128,
0.97 at 256, 0.62 at 512, 0.53 at 1024. The search is a bounded randomised
backjumping search with a node budget of `20 * length` per attempt; it is not an
optimal search and does not claim to be. Raising `max_attempts` buys a little more
spread for a lot more time. A reader who needs a spread near the rule of thumb at
length 1024 will not get it from this implementation inside this compute budget, and
the README says so under Limitations.

**The search is also not monotone in the requested spread.** At length 256 it reaches
spread 11 but fails at spread 10 with the same seed and budget (visible in the "gaps
below largest" column of `srandom_output.txt`). That is the expected behaviour of a
bounded randomised search, not a defect in the condition being enforced — every
permutation it does return is verified by `s_parameter` to satisfy the request.
It does mean that a failure at one spread says nothing about feasibility at the next.

**Forward-only greedy S-random placement failed outright.** The construction as
usually described places source indices left to right with random redraws. Measured
during this build: at length 1024 and spread 16 it reached position 1023 of 1024 and
gave up; at length 512 and spread 12 it failed as well. Single-step backtracking also
failed inside the node budget at length 256 and spread 10. Only backjumping by three
positions worked, and that is what ships. The failures are recorded in
`src/interleavekit/srandom.py`.

**The helical step pattern does not extend.** On a 16-row array the largest fully
dispersed burst is `rows * step - 1` for steps 1 through 8, then breaks: step 9 gives
113, not 143 (`burst_dispersion_output.txt`, section 3). No formula is claimed for
the helical construction; the metric is there to be measured, and this is an example
of why.

## Not validated

* **Nothing here is compared against a hardware interleaver or a link measurement.**
  Every number is combinatorial. The constructions are permutations and delay lines;
  whether a given FPGA or ASIC implementation matches the cost model in
  `src/interleavekit/base.py` is a question about that implementation.
* **No bit-error-rate or frame-error-rate result.** This library does not model a
  channel, a code or a decoder. Whether a surviving run of 1 is correctable depends
  entirely on the code wrapped around it, and that code is not here.
* **The helical construction is not bit-compatible with MATLAB `helintrlv`**, which
  is a streaming block with initial-condition fill. The relationship is stated in
  `src/interleavekit/helical.py` and no equivalence was tested.
* **`commpy` and `komm` were read, not run.** Neither installs in this build
  environment (`komm`'s build backend is unavailable and `commpy` is not installed).
  Their source was downloaded and read to write the alternatives table; no
  side-by-side execution was performed and none is claimed. What was read is stated
  in the README.

## Raw output files

| File | Script | Exit | Wall time |
|---|---|---|---|
| `block_spread_output.txt` | `validate_block_spread.py` | 0 | 2.8 s |
| `roundtrip_output.txt` | `validate_roundtrip.py` | 0 | 0.8 s |
| `burst_metric_equivalence_output.txt` | `validate_burst_metric_equivalence.py` | 0 | 2.2 s |
| `burst_dispersion_output.txt` | `validate_burst_dispersion.py` | 0 | 0.2 s |
| `cost_tradeoff_output.txt` | `validate_cost_tradeoff.py` | 0 | 5.1 s |
| `spread_vs_burst_output.txt` | `validate_spread_vs_burst.py` | 0 | 0.5 s |
| `srandom_output.txt` | `validate_srandom.py` | 0 | 45.7 s |
| `worked_example_output.txt` | `worked_example.py` | 0 | 0.2 s |
| `example_burst_dispersion_curves_output.txt` | `examples/burst_dispersion_curves.py` | 0 | 1.2 s |
| `example_latency_memory_tradeoff_output.txt` | `examples/latency_memory_tradeoff.py` | 0 | 9.4 s |
| `example_spread_vs_burst_output.txt` | `examples/spread_vs_burst.py` | 0 | 1.1 s |
| `example_permutation_structure_output.txt` | `examples/permutation_structure.py` | 0 | 1.3 s |

Longest single run: 45.7 s, inside the 3-minute budget stated in the README.

## Test suite

`python -m pytest tests/ -q`, counted from the junit XML rather than from pytest's
stdout line: **191 tests, 0 failures, 0 errors, 0 skipped**, 5.3 s wall time. The suite
includes hand-computed permutations written out in the test comments for the 4x4 and
2x3 block interleavers, the 3x3 and 2x4 helical interleavers, and the first nine
transmitted positions of a 3-register convolutional interleaver; Hypothesis property
tests for the round-trip identity, bijectivity, the spread bound and burst-metric
monotonicity; and the doctests in every public module.

This software is research-grade. It is not flight-qualified, not certified, and not
approved for operational aerospace use.
