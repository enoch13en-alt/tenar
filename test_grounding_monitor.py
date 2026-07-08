#!/usr/bin/env python
"""Regression guard for the grounding monitor's matcher (app.grounding_audit).

WHY THIS EXISTS
---------------
The two-axis precision architecture (PRECISION_DISCIPLINE + ARGUMENTATIVE_COMMITMENT)
is only as trustworthy as the monitor that measures it. That monitor classifies every
pinpoint a real answer emits as grounded / flagged / ungrounded by checking it against
the retrieved corpus text. The classifier has TWO ways to lie, and a careless edit can
reintroduce either:

  * OVER-STRICT (phantom leaks): literal substring matching once false-flagged
    "ss.72" / "sections 72" against a corpus that said "Section 72-75" — a 26% fake
    ungrounded rate on the first live run. Fixed by matching (provision-type,
    top-level number) with range/plural/abbreviation tolerance.

  * OVER-LOOSE (false negatives): the fix above, if pushed too far, could wave through
    a genuinely ungrounded cite — e.g. accept "Section 80" merely because the corpus
    mentioned the nearby range "Section 72-75". That would blind the monitor to the
    exact failure it was built to catch.

This test pins BOTH directions. If it goes red, the monitor can no longer be trusted to
measure the architecture, so treat a failure here as blocking. Run:

    venv/bin/python test_grounding_monitor.py      # exit 0 = green, 1 = regression

SCOPE — what this does and does NOT cover
-----------------------------------------
This guards the matcher LOGIC against code edits: the CORPUS constant below is a fixed
synthetic string, so a passing run proves the classification logic is sound for the
citation forms these cases anticipate. It CANNOT catch corpus drift — a real course
that phrases a citation in a form none of these ten cases model could still produce a
false positive/negative this test never sees. That surface is covered by the OTHER
half of the system: the live grounding monitor. If real traffic throws a citation form
the matcher mishandles, it surfaces in /api/admin/grounding (by_course split + leak
log), not here. Test guards the logic; the monitoring cycle guards against unpredicted
corpus forms. Keep both running — they cover different failure surfaces.
"""
import os, sys, json, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
# load .env if present so `import app` works in the normal app env (the monitor itself
# needs no API key — grounding_audit is pure — but import-time config may look).
_envp = os.path.join(HERE, ".env")
if os.path.exists(_envp):
    for ln in open(_envp):
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.strip().split("=", 1)
            os.environ.setdefault(k, v)

import app

# redirect the monitor's log to a throwaway file and exercise the REAL shipped function
_tmp = tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False)
_tmp.close()
app.GROUNDING_LOG = _tmp.name

CORPUS = ("Compensation is payable under Section 72-75 of Act 703 and L.I. 2175. "
          "The Company shall pay ground rent under Section 23 of Act 703.")
CHUNKS = [{"text": CORPUS}]


def classes_for(answer):
    """Run the real monitor on one synthetic answer; return the set of pinpoint
    classes it logged (grounded/flagged/ungrounded/weak)."""
    open(app.GROUNDING_LOG, "w").close()
    app.grounding_audit("q", "TEST", answer, CHUNKS)
    line = open(app.GROUNDING_LOG).read().strip()
    summ = (json.loads(line).get("summary") or {}) if line else {}
    return {k for k, v in summ.items() if v}


# (answer, class the pinpoint-under-test must carry, class it must NOT carry)
CASES = [
    ("The duty arises under ss.72 of Act 703.",        "grounded",   "ungrounded"),  # abbrev, in range
    ("See sections 72 of Act 703.",                    "grounded",   "ungrounded"),  # plural
    ("Compensation under Section 74 applies.",         "grounded",   "ungrounded"),  # inside range
    ("This is governed by Section 23.",                "grounded",   "ungrounded"),  # exact
    ("The rule is in Section 80 of Act 703.",          "ungrounded", None),          # OUTSIDE range -> must flag
    ("It falls under Section 999.",                    "ungrounded", None),          # absent
    ("Governed by L.I. 2175.",                         "grounded",   "ungrounded"),  # distinctive num present
    ("Governed by L.I. 9999.",                         "ungrounded", None),          # distinctive num absent
    ("See Article 5 of the Convention.",               "ungrounded", None),          # no articles in corpus
    ("The exact provision (Section 999) is not in the retrieved material; verify against the Act.",
                                                       "flagged",    "ungrounded"),  # hedged -> not a leak
]

fails = 0
print(f"{'must-have':11} {'must-not':11} {'got':28} {'PASS?':5}  answer")
print("-" * 104)
for answer, must, mustnot in CASES:
    got = classes_for(answer)
    ok = (must in got) and (mustnot is None or mustnot not in got)
    fails += 0 if ok else 1
    print(f"{must:11} {str(mustnot):11} {str(sorted(got)):28} "
          f"{'PASS' if ok else 'FAIL':5}  {answer[:48]}")

os.unlink(_tmp.name)
print("\n" + ("ALL PASS — grounding matcher sound in both directions"
              if not fails else f"*** {fails} FAILED — monitor cannot be trusted ***"))
sys.exit(1 if fails else 0)
