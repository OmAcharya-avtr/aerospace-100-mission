# XML risk probe, executed 2026-10-18. Produces the B314 accepted-risk tables in
# products/P064/README.md and products/P065/README.md. Two scripts concatenated;
# run with: python3 -I XML_RISK_PROBE_2026-10-18.py

import sys, resource, xml.etree.ElementTree as ET
sys.path.insert(0, "/home/claude/aerospace-100-mission/products/P065/src")
from hazardtrace.crosslinks import parse_junit

BILLION = """<?xml version="1.0"?>
<!DOCTYPE t [
 <!ENTITY a "aaaaaaaaaa">
 <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
 <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
 <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">
 <!ENTITY e "&d;&d;&d;&d;&d;&d;&d;&d;&d;&d;">
 <!ENTITY f "&e;&e;&e;&e;&e;&e;&e;&e;&e;&e;">
]>
<testsuite name="x" tests="1"><testcase name="&f;"/></testsuite>"""

XXE = """<?xml version="1.0"?>
<!DOCTYPE t [ <!ENTITY x SYSTEM "file:///etc/passwd"> ]>
<testsuite name="x" tests="1"><testcase name="&x;"/></testsuite>"""

for label, payload in (("billion-laughs", BILLION), ("external-entity(file:///etc/passwd)", XXE)):
    for name, fn in (("P065 parse_junit", parse_junit), ("stdlib ET.fromstring", ET.fromstring)):
        try:
            r = fn(payload)
            s = repr(r)[:120]
            leak = "root:" in s or "/bin/" in s
            print(f"{label:38s} {name:22s} -> RETURNED (len {len(s)}) leak={leak}")
        except Exception as e:
            print(f"{label:38s} {name:22s} -> REFUSED {type(e).__name__}: {str(e)[:70]}")
print("peak RSS MiB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1))
import sys, resource, time, xml.etree.ElementTree as ET
def payload(levels):
    ents = ['<!ENTITY e0 "aaaaaaaaaa">']
    for i in range(1, levels+1):
        ents.append(f'<!ENTITY e{i} "' + f'&e{i-1};'*10 + '">')
    return ('<?xml version="1.0"?>\n<!DOCTYPE t [\n' + "\n".join(ents) +
            f']>\n<testsuite name="x" tests="1"><testcase name="&e{levels};"/></testsuite>')
for lv in (3, 5, 7):
    t0=time.time()
    try:
        root = ET.fromstring(payload(lv))
        n = len(root.find("testcase").get("name"))
        print(f"levels={lv} theoretical={10**(lv+1)} expanded_name_chars={n} secs={time.time()-t0:.2f}")
    except Exception as e:
        print(f"levels={lv} -> REFUSED {type(e).__name__}: {str(e)[:80]}")
print("peak RSS MiB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1))
