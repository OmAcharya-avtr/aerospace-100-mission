# Recovery — boardprobe

Five procedures, for the five ways a campaign on an NVIDIA Jetson Orin Nano
goes wrong. Each has symptoms, an immediate action, a diagnosis sequence and a
decision about the data already collected.

**NONE OF THESE HAS BEEN EXECUTED ON A JETSON ORIN NANO.** No board was
reachable from the environment this package was built in. The commands that
touch only this package were executed on an x86_64 Linux host; the ones that
touch the board are written from the documentation cited in
`src/boardprobe/orin_nano.py` and are **UNVERIFIED**.

The rule that governs all five: **a run whose provenance cannot be established
aborts. It is never downgraded silently and never completed with a weaker
label than the data deserves.** Procedure 5 is that rule as a procedure.

---

## 1. The board becomes unreachable mid-run

**Symptoms.** A `DeviceUnreachableError` naming a locator, partway through a
campaign that was working. Or `HarnessError: the environment backend ... could
serve none of the N requested channels`. Or the shell itself hangs on a `cat`
of a sysfs node.

**Immediate action.** Do not retry in a loop. A sysfs read that has started
failing will keep failing, and a retry loop turns one error into a log full of
them.

```bash
# 1. is the board alive at all?
uptime
dmesg | tail -40

# 2. is it a thermal event? (procedure 2 if so)
for zone in /sys/devices/virtual/thermal/thermal_zone*/; do
  printf '%s %s %s\n' "$(cat "$zone/type")" "$(cat "$zone/temp")" "$zone"
done

# 3. did the driver unbind?
ls /sys/bus/i2c/drivers/ina3221/
dmesg | grep -iE 'ina3221|thermal|i2c'

# 4. did the kernel drop the PMU?
cat /proc/sys/kernel/perf_event_paranoid
ls /sys/bus/event_source/devices/
```

**Diagnosis.**

| what you see | what it is | what to do |
| --- | --- | --- |
| `dmesg` shows an i2c timeout or NAK | the power monitor stopped answering on the bus | power-cycle the board; the rail channels are lost for this run |
| the `hwmon<N>` index has changed | the driver re-probed and renumbered | nothing to fix in the declaration — the locator is a glob for this reason; re-run |
| a `thermal_zone` directory has vanished | the zone was removed, which on L4T usually follows a driver reload | re-run step 4 of `docs/DEPLOYMENT.md`; the zone index may have moved |
| `perf_event_paranoid` has risen | something reset the sysctl, typically a reboot | re-apply step 6 of `docs/DEPLOYMENT.md` |
| the board does not respond at all | power, thermal shutdown, or a kernel panic | serial console; this is outside this package |

**The data already collected.** Readings taken before the failure are still
valid readings: each carries its own provenance and timestamp and the seal is
intact. What is NOT valid is an aggregate over a window the failure fell
inside, because the sample is truncated at a non-random point.

```bash
# Keep the readings, discard the aggregate. Re-aggregate over a window that
# ENDS before the first refusal, and say in the report that you did.
```

Do not fill the gap from the simulated backend to "complete" the run. If you
do, every figure from that run becomes `SYNTHETIC` — which is this package
working correctly, and is also a wasted campaign.

---

## 2. A thermal throttle during a benchmark

**Symptoms.** Timing figures that drift upward through a run; a die
temperature approaching a trip point; `cpu0_frequency` falling during a
benchmark that is not changing its workload.

**This is not an error and nothing will raise.** A throttled board is working
as designed, and a benchmark that silently includes a throttle is a wrong
number with a plausible magnitude. Detection is the operator's job and this
package gives you the channels to do it with.

**Immediate action.** Stop the run. A partially throttled benchmark cannot be
repaired by averaging.

```bash
# 1. read the trip points for every zone
for zone in /sys/devices/virtual/thermal/thermal_zone*/; do
  printf '%s type=%s temp=%s\n' "$zone" "$(cat "$zone/type")" "$(cat "$zone/temp")"
  for trip in "$zone"trip_point_*_temp; do
    [ -e "$trip" ] && printf '    %s %s\n' "$trip" "$(cat "$trip")"
  done
done

# 2. read the current frequency and the policy's maximum
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq

# 3. check for a throttle count, where the kernel exposes one
dmesg | grep -iE 'throttl|thermal'
```

All values are in millidegrees Celsius and kilohertz respectively.

**Diagnosis and action.**

| what you see | what to do |
| --- | --- |
| die temperature within 5 °C of the lowest trip point | let the board cool; add fan or heatsink before repeating |
| `scaling_cur_freq` below `scaling_max_freq` while the workload is constant | the governor or the thermal framework is limiting the clock; the run is invalid |
| temperature rising monotonically across the whole run | the run is longer than the board's thermal time constant; shorten it or add a cooldown between repeats |

**Re-run so a throttle is visible rather than inferred.** Sample the die
temperature as a resource channel and inspect its own statistics:

```bash
python -m boardprobe benchmark --repeats 200 --environment live
```

The resource table reports the mean of each sampled channel; a run whose
`die_temp_*` mean is near a trip point should be discarded. For a stricter
test, record the per-repeat temperature yourself, fit a slope, and discard any
run whose die temperature rose by more than a declared budget across it — a
few degrees is a reasonable budget and this package does not pick one for you,
because the right budget depends on the trip margin you are prepared to accept.

**Do not** raise the trip points, disable the thermal zones or pin the governor
to performance to make a benchmark reproducible. A number measured with the
board's thermal protection defeated is not a number about that board.

---

## 3. A partially written trace

**Symptoms.**

```
TraceError: the trace 'orin_nano_recorded_trace.json' is not valid JSON at
line 2041 column 18: Expecting ',' delimiter. A trace truncated by an
interrupted run is the expected cause; see docs/RECOVERY.md, 'a partially
written trace'.
```

Or `TraceError: trace document is missing the required key 'samples'`, or
`sample 1043 of the trace is malformed`.

**Why it happens.** `Trace.to_json` returns a string and the caller writes it;
a write interrupted by a signal, a full filesystem or a power loss leaves a
prefix of a valid document on disk. There is no partial-write protection in
this package, deliberately: a half-written trace that LOOKS valid is worse than
one that fails to parse.

**Immediate action. Do not edit the file to make it parse.** A trace closed by
hand has an unknown number of samples on an unknown set of channels, and any
aggregate over it is over a sample whose size you no longer know.

```bash
# 1. how much of it is there?
wc -c fixtures/orin_nano_recorded_trace.json
python -c "
import json, sys
text = open(sys.argv[1]).read()
try:
    json.loads(text)
    print('parses cleanly')
except json.JSONDecodeError as error:
    print(f'fails at line {error.lineno} column {error.colno}: {error.msg}')
    print(f'bytes present: {len(text)}')
" fixtures/orin_nano_recorded_trace.json

# 2. is the metadata there at all? The schema, origin and content_provenance
#    are written FIRST by to_json, so a truncated file usually still has them.
head -c 600 fixtures/orin_nano_recorded_trace.json
```

**Recovery, in order of preference.**

1. **Re-record.** If the board is still available, step 8 of
   `docs/DEPLOYMENT.md` takes a couple of minutes. This is almost always the
   right answer.
2. **Salvage, with the loss declared.** If the board is gone and the samples
   are irreplaceable, truncate the sample array at the last COMPLETE sample
   object, close the JSON, and amend `origin` to say in the file that the
   document was salvaged from a truncated write, how many samples were
   recovered and how many were intended. The `origin` field is printed in
   every report built from the trace, which is the whole reason to put the
   admission there rather than in a commit message.
3. **Discard.** A trace whose `origin` you cannot honestly complete is not
   evidence.

**Prevention.** Write to a temporary path in the same directory and rename:

```bash
python - <<'PY'
import os
from pathlib import Path
target = Path("fixtures/orin_nano_recorded_trace.json")
scratch = target.with_suffix(".json.partial")
scratch.write_text(trace.to_json())      # trace from step 8 of DEPLOYMENT.md
os.replace(scratch, target)              # atomic within one filesystem
PY
```

`os.replace` is atomic within a filesystem, so a reader sees either the old
file or the whole new one.

---

## 4. A stale lock

**Symptoms.** A campaign will not start, or two campaigns are interleaving
their readings; or, in a git working copy, `git commit` fails with
`Unable to create '.git/index.lock': File exists`.

**This package takes no locks of its own, and that is a design decision with a
consequence.** Nothing stops two processes reading the same sysfs node at the
same time — the kernel serialises the read and both get an answer. What two
concurrent campaigns DO collide over is the output: two processes writing the
same trace file produce one interleaved, unparseable document, which arrives as
procedure 3.

**If you have added a lock around a campaign** — a sensible thing to do for a
long one — recover it like this:

Put the lock beside the data it guards, not in the system temporary
directory: a lock in the repository is removed by the same `git clean` that
removes everything else from a failed run, and it is visible to whoever next
opens the directory. `.boardprobe-campaign.lock` is in `.gitignore`.

```bash
# 1. is the owner alive? A lock file should carry the pid that took it.
cat .boardprobe-campaign.lock 2>/dev/null
ps -p "$(cat .boardprobe-campaign.lock 2>/dev/null)" 2>/dev/null || \
  echo 'no such process: the lock is stale'

# 2. what is holding the trace file open?
command -v lsof >/dev/null && lsof fixtures/orin_nano_recorded_trace.json

# 3. only once both say nothing, remove it
rm -f .boardprobe-campaign.lock
```

Never remove a lock whose owner is still running. Check with `ps` first; the
reason a lock exists is that the second process would corrupt the first's
output, and that corruption arrives as procedure 3 with no indication of its
cause.

**For the git lock**, which is the case most likely to bite during a campaign:

```bash
ps aux | grep -c '[g]it '          # 0 means no git process is running
rm -f .git/index.lock              # only then
```

A stale `.git/index.lock` is also the signature of a sandbox where deletion is
not permitted: a `git` operation that cannot remove its own lock file leaves it
behind and every later `git` command fails with `File exists`. If `rm` itself
reports `Operation not permitted`, the environment is refusing deletions and
the fix is at that level, not in the repository.

**Recommended lock, if you want one.** `fcntl.flock` on a file descriptor held
open for the campaign's lifetime, not a lock file whose existence is the lock.
A flock is released by the kernel when the process dies, however it dies, so it
cannot go stale.

---

## 5. A run whose provenance cannot be established

**This is the one that matters, and the rule is: abort.**

**Symptoms.**

```
ProvenanceError: provenance cannot be established from an empty collection of
readings: the meet of nothing is the lattice identity (live), which would let
a derivation with no inputs claim to be a measurement.
```

```
TamperError: reading on channel 'die_temp_cpu' from source 'live:...' fails
its seal: at least one field has been modified since the backend produced it.
```

```
RuntimeError: SimulatedSource.backend_provenance has been rebound from
'simulated' to 'live'.
```

**What each one means.**

| error | what happened | what it would have meant if it had not raised |
| --- | --- | --- |
| `ProvenanceError` on an empty collection | every reading in a window was refused, so the aggregate has no inputs | a figure with no data behind it, labelled `MEASURED`, because the meet of nothing is the lattice's top element |
| `TamperError` on a seal | a reading's fields changed after the backend produced it | a reading whose provenance is whatever the last writer chose |
| `TamperError` on a type | something that is not a `Reading` reached a derivation | a provenance supplied by a subclass rather than by a backend |
| `RuntimeError` on a rebound backend provenance | a backend class's declared provenance was reassigned at run time | a simulated backend stamping `LIVE` |

**The procedure.**

1. **Stop the run.** Do not catch any of these and continue. They are raised
   precisely at the point where continuing would produce a figure whose
   provenance is unknown, and an unknown provenance in this package's
   type system is indistinguishable from a false one.

2. **Do not relabel.** The temptation when a campaign is nearly finished is to
   call the result `SYNTHETIC` and keep it. Resist it for a specific reason:
   `SYNTHETIC` is a positive claim that a generator produced the numbers, and
   a run whose provenance cannot be established does not support that claim
   either. There is no label for "unknown" in this package, deliberately, and
   there is no plan to add one.

3. **Find the cause.**

```bash
# What did the run actually manage to read?
python -m boardprobe sample --backend live

# Does the package still agree with itself?
python -m pytest tests/ -q
python -m boardprobe dry-run

# Is the installed package the one you think it is?
python -c "import boardprobe; print(boardprobe.__version__, boardprobe.VALIDATION_LEVEL)"
python -m boardprobe provenance
```

If `python -m pytest` fails, the installation is wrong or a dependency moved;
the dependency ranges in `pyproject.toml` are bounded on both sides for this
reason. If the suite passes and the campaign still fails, the cause is in the
campaign's own code — most often a reading list that is empty because every
channel was refused, which is procedure 1 wearing a different error message.

4. **Re-run from the beginning, or report the campaign as failed.** There is no
   third option. A campaign that produced some readings and then lost the
   ability to establish provenance has produced evidence for the window before
   the loss and nothing after it; aggregate over that window, say in the report
   where it ends and why, and run again.

5. **If the cause was this package**, the route is a defect and belongs in
   `validation/validate_adversarial_label.py` as a new numbered route, with the
   design changed rather than the check. Two routes were found that way during
   the build and both are recorded in that script's output with what the
   successful attack was.
