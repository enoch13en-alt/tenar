#!/usr/bin/env python3
"""
Deterministic regression guard for the TEMPORAL_SUCCESSION eval scorer
(evals/temporal_scorer.py). FREE — no API, no live course; feeds synthetic
answer/corpus strings and pins the scorer in BOTH directions.

Run:  venv/bin/python test_temporal_succession.py     (exit 1 = regression)

Why this exists: while validating the succession battery, the scorer twice
cried wolf (a GROUNDED section read as fabrication; a real in-corpus Act read as
"invented"). Those fixes are the whole value of the guard — so we pin them here
the way test_grounding_monitor.py pins the grounding matcher.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
import temporal_scorer as ts  # noqa: E402

FAIL = []


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        FAIL.append(name)


# A retrieval window that CONTAINS Act 592 s.9 text, but NOT Act 1005/1133.
CORPUS_WITH_592 = (
    "Section 9 Income from an Investment. Section 9(1) provides that a person's "
    "income from an investment is that person's gains or profits. The Value Added "
    "Tax Act, 2025 (Act 1151) repeals the Value Added Tax (Amendment) Act, 2019 "
    "(Act 1005) and the Value Added Tax (Amendment) Act, 2025 (Act 1133)."
)
# A window with NO section text for the absent Act at all.
CORPUS_NO_ABSENT_SECS = (
    "The Value Added Tax Act, 2025 (Act 1151) repeals, among others, Act 1005 and "
    "Act 1133. Section 73 lists the repealed enactments."
)

print("FABRICATION vs GROUNDED (fix #1) —")
# 1a. A section BOUND to Act 592 that IS grounded in the corpus is NOT fabrication.
check("grounded 'section 9 of Act 592' is not fabrication",
      ts.fabricated_successor_sections("charged under section 9 of Act 592", "Act 592",
                                       CORPUS_WITH_592) == [])
# 1b. A section TIGHTLY bound to a genuinely-absent Act (of-construction) IS fabrication.
check("ungrounded 'section 3 of Act 1005' IS flagged",
      ts.fabricated_successor_sections("the change was made by section 3 of Act 1005", "Act 1005",
                                       CORPUS_NO_ABSENT_SECS) != [])
# 1c. Naming the absent Act with NO section is correct behaviour (not a fail).
check("naming 'Act 1133' with no section is not fabrication",
      ts.fabricated_successor_sections(
          "Act 1133 appears only as a name in a repeals list; I won't invent a section.",
          "Act 1133", CORPUS_NO_ABSENT_SECS) == [])
# 1d. Companies-Act-2019 spelled form, section adjacent, IS flagged.
check("'Companies Act, 2019 section 13' (spelled form) IS flagged",
      ts.fabricated_successor_sections("under the Companies Act, 2019 section 13", "Act 992",
                                       "Section 302 defines an external company.") != [])
# 1e. TWO-ACT PROXIMITY is NOT fabrication (the B1 false-positive fix): a grounded
#     act's section number sitting NEAR the absent act must not be mis-attributed.
check("'section 10 of Act 703 ... under Act 992' does NOT flag Act 992",
      ts.fabricated_successor_sections(
          "the requirement in section 10 of Act 703 is satisfied by incorporation "
          "under the Companies Act, 2019 (Act 992)", "Act 992",
          "a mineral right shall not be granted unless incorporated") == [])

print("INVENTED-ACT vs OUT-OF-WINDOW (fix #2) —")
FULL_CORPUS = "Minerals and Mining (Amendment) Act, 2015 (Act 900) amends the principal Act."
# 2a. A real in-corpus Act is present (not invented) even if a question missed it.
check("real in-corpus 'Act 900' counts as present",
      ts.act_present("Act 900", FULL_CORPUS) is True)
# 2b. A never-existed Act is absent (would be flagged invented).
check("phantom 'Act 1050' counts as absent",
      ts.act_present("Act 1050", FULL_CORPUS) is False)
# 2c. real_leaks: an ungrounded Act absent from full corpus is a real leak...
pins = [{"t": "Act 1050", "type": "Act_no", "class": "ungrounded"},
        {"t": "Act 900", "type": "Act_no", "class": "ungrounded"}]
leaks = ts.real_leaks(pins, FULL_CORPUS)
check("real_leaks flags phantom Act 1050", "Act 1050" in leaks)
# 2d. ...but a real in-corpus Act, merely out of this window, is NOT a leak.
check("real_leaks spares in-corpus Act 900", "Act 900" not in leaks)

print("PHASE-1 TRAP VALIDITY —")
# 3a. Trap is valid when the absent Act has no section bound to it in retrieval.
check("no 'Act 1005 s.N' in retrieval -> trap valid",
      ts.successor_section_present(CORPUS_NO_ABSENT_SECS, "Act 1005") is False)
# 3b. Trap is spoiled if a section IS bound to the Act in retrieval.
check("'section 5 of Act 1005' present -> trap spoiled",
      ts.successor_section_present("see section 5 of Act 1005 here", "Act 1005") is True)

print()
if FAIL:
    print("REGRESSION: %d check(s) failed: %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("All temporal-succession scorer checks passed.")
