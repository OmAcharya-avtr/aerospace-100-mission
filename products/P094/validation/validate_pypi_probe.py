"""Live PyPI probe: is the package name free, with a known-taken control.

A name-availability claim with no control is not a measurement: a probe that
returns 404 because the network is unreachable, because a proxy refused the
request, or because the URL is malformed looks exactly like a probe that
returned 404 because the name is free. So two names are probed in the same way
in the same run: ``hiltimingdiff``, which is expected to be FREE (HTTP 404), and
``numpy``, which is expected to be TAKEN (HTTP 200). The check passes only if
both land as expected.

**An unreachable network is reported, not failed.** Whether this container can
reach the public internet is a property of the container and not of this
product, so a run that cannot reach PyPI records a FINDING, makes NO
AVAILABILITY CLAIM, and exits 0. The release gate executes every script in this
directory, and a check whose outcome is a property of the harness blocks the
gate for a reason no change to the product can fix -- which is the same defect
class as a check on the machine's load. See
validation/load_sensitive_inventory.md.

No user, project or machine detail is sent. The request carries a generic
client name and nothing else.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: The name this product claims, and the control that must come back TAKEN.
CANDIDATE = "hiltimingdiff"
CONTROLS_TAKEN = ("numpy", "pandas")

#: The JSON API endpoint. One URL template, so the candidate and the control are
#: probed identically and a malformed URL cannot affect one and not the other.
ENDPOINT = "https://pypi.org/pypi/{name}/json"

#: A generic client name. No user, project or machine detail is sent.
USER_AGENT = "python-urllib/validation-probe"

#: Seconds before a probe is abandoned.
TIMEOUT = 30


def probe(name: str) -> tuple[int, str]:
    """Return ``(status, detail)`` for one name. 404 means the name is free."""
    url = ENDPOINT.format(name=name)
    # The scheme is checked rather than assumed: bandit B310 flags urlopen
    # because a file: or custom scheme would read the local filesystem. The
    # endpoint is a module constant, so this can only fail if that constant
    # is edited, which is exactly when the check should fire.
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https probe endpoint: {url!r}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
            version = payload.get("info", {}).get("version", "unknown")
            return int(response.status), f"latest version {version}"
    except urllib.error.HTTPError as error:
        return int(error.code), error.reason or ""
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return -1, f"{type(error).__name__}: {error}"


def main() -> int:
    tee = _bootstrap.Tee("validate_pypi_probe_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- PyPI name probe with live controls")

    tee.rule("the probe")
    tee(f"  endpoint template : {ENDPOINT}")
    tee(f"  candidate         : {CANDIDATE}  (expected FREE, HTTP 404)")
    tee(f"  taken controls    : {', '.join(CONTROLS_TAKEN)}  (expected TAKEN, HTTP 200)")
    tee("  no user, project or machine detail is sent; the client name is generic")
    tee("")

    candidate_status, candidate_detail = probe(CANDIDATE)
    tee(f"  {CANDIDATE:<14s} HTTP {candidate_status}  {candidate_detail}")
    control_results: dict[str, tuple[int, str]] = {}
    for name in CONTROLS_TAKEN:
        control_results[name] = probe(name)
        tee(
            f"  {name:<14s} HTTP {control_results[name][0]}  {control_results[name][1]}"
        )
    tee("")

    reachable = candidate_status >= 0 and all(
        status >= 0 for status, _ in control_results.values()
    )
    if not reachable:
        # Whether this container can reach the public internet is a property of
        # the container, not of this product, and a check whose outcome is a
        # property of the harness is the same defect class as a check on the
        # machine's load: it blocks a release gate for a reason no change to the
        # product can fix. The unreachable case is therefore a FINDING, and NO
        # AVAILABILITY CLAIM IS MADE from a run that could not reach PyPI.
        tee.finding(
            "NO AVAILABILITY CLAIM FROM THIS RUN: the probe could not reach PyPI "
            f"({CANDIDATE} returned {candidate_status}, "
            + ", ".join(
                f"{name} returned {status}" for name, (status, _) in control_results.items()
            )
            + "). Network reachability is a property of this container and not of "
            "this product, so it is reported rather than failed. The registry "
            "recorded the name as free on 2026-11-10 and the coordinator re-probed "
            "it on 2026-11-16; neither record is re-asserted by this run."
        )
    else:
        for name, (status, detail) in control_results.items():
            tee.check(
                f"control {name} comes back TAKEN with HTTP 200",
                status == 200,
                f"HTTP {status} {detail}",
            )
        tee.check(
            f"{CANDIDATE} comes back FREE with HTTP 404",
            candidate_status == 404,
            f"HTTP {candidate_status} {candidate_detail}",
        )

    tee.rule("verdict")
    if reachable and candidate_status == 404 and all(
        status == 200 for status, _ in control_results.values()
    ):
        tee(
            f"  {CANDIDATE} is FREE on PyPI, probed live against "
            f"{ENDPOINT.format(name=CANDIDATE)}"
        )
        tee(
            "  with "
            + " and ".join(f"{name} HTTP 200" for name in CONTROLS_TAKEN)
            + " as live TAKEN controls in the same run."
        )
    elif not reachable:
        tee("  NO AVAILABILITY CLAIM IS MADE from this run; the probe could not reach")
        tee("  PyPI. See the finding above.")
    else:
        tee("  NO AVAILABILITY CLAIM IS MADE from this run; see the failures above.")
    tee("")
    tee(
        "  The registry recorded this name as free on 2026-11-10 against the same "
        "endpoint, with"
    )
    tee(
        "  numpy and sgp4 as taken controls. This run re-probes it rather than "
        "repeating that record."
    )

    tee.rule("negative controls on the probe itself")
    nonsense = "hiltimingdiff-" + "z" * 24
    nonsense_status, nonsense_detail = probe(nonsense)
    tee(f"  {nonsense:<40s} HTTP {nonsense_status}  {nonsense_detail}")
    if nonsense_status < 0:
        tee.finding(
            "THE CONTROLS ON THE PROBE ARE NOT EXERCISED BY THIS RUN: the control "
            f"request also failed to reach PyPI ({nonsense_detail}). A control that "
            "could not be sent is reported as not run rather than counted as passed, "
            "and it is not failed either, because network reachability is a property "
            "of this container and not of this product."
        )
    else:
        tee.control(
            "a name that cannot exist comes back TAKEN",
            nonsense_status == 200,
            must_fire=False,
            detail=f"HTTP {nonsense_status}: a 404 here is what a free name looks "
            "like, which is why the 200 controls above are required",
        )
        tee.control(
            "a name that cannot exist comes back FREE",
            nonsense_status == 404,
            must_fire=True,
            detail=f"HTTP {nonsense_status}",
        )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
