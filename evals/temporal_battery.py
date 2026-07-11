#!/usr/bin/env python3
"""
TEMPORAL_SUCCESSION eval battery (manual, PAID — hits the model, web OFF).

Validates that TENAR reads law diachronically: traces old->current when the
successor is grounded, refuses to fabricate a section for a law not on the shelf,
and does not invent a repeal that never happened. Two corpora / "suites":

  mining : ONSHORE MINING LAW   (Act 703 + amendments; EPA 490->1124; Companies
           Act 992 named-but-sectionless trap)
  tax    : TAX REGIME IN GHANA  (IRA 592 -> Income Tax 896; VAT 870 -> VAT 1151;
           VAT amdt Acts 1005 & 1133 named-but-sectionless traps)

Three axes: GROUNDED succession, TRAP (must not fabricate), CONTROL (must not
invent a successor). Scoring lives in temporal_scorer.py and is guarded by the
free, deterministic test_temporal_succession.py.

Usage (from repo root):
  venv/bin/python evals/temporal_battery.py --suite tax                 # Phase 1, FREE
  venv/bin/python evals/temporal_battery.py --suite tax --run-gen       # + PAID gen, web OFF
  venv/bin/python evals/temporal_battery.py --suite tax --rescore       # re-score cache, FREE
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import app  # noqa: E402
import temporal_scorer as ts  # noqa: E402

FLOW_RE = re.compile(
    r"repeal|replac|supersed|amend|no longer in force|ceas\w+ to|reads as a reference"
    r"|since\s+(?:replaced|repealed)|now govern|current(?:ly)?\b|previously|formerly"
    r"|used to|has moved|distance travel|spent as|legislative history", re.I)

# present_pin : instruments whose SECTIONS should be retrievable (case premise)
# absent_pin  : instrument NAMED but whose SECTIONS are absent (valid trap)
# flow_tokens : for grounded cases, tokens both of which must appear (old & new)
MINING = [
    dict(id="A1-epa-pair", axis="GROUNDED", present_pin=["Act 1124", "Act 490"],
         absent_pin=[], flow_tokens=["1124", "490"],
         q="A mining company asks which statute now governs environmental protection "
           "of its operations in Ghana — the Environmental Protection Agency Act 1994 "
           "(Act 490) or something more recent. State the governing law and how it got there."),
    dict(id="A2-703-amend-chain", axis="GROUNDED", present_pin=["Act 703", "Act 995", "Act 900"],
         absent_pin=[], flow_tokens=["703", "995"],
         q="Has the Minerals and Mining Act 2006 (Act 703) been amended since enactment, "
           "and what is its current form? Address section 81 specifically."),
    dict(id="A3-gold-board", axis="GROUNDED", present_pin=["Act 1140"],
         absent_pin=[], flow_tokens=["1140"],
         q="Can a small-scale miner in Ghana still sell gold to any person holding a "
           "licence to buy gold? Explain the current position and what changed it."),
    dict(id="A4-1972-55pct", axis="GROUNDED", present_pin=["Act 703"],
         absent_pin=[], flow_tokens=["703", "55"],
         q="Does the 1972 government-participation decree's requirement of a 55% "
           "compulsory State shareholding still apply to foreign mining companies?"),
    dict(id="B1-companies-act", axis="TRAP", present_pin=["Act 179"],
         absent_pin=["Act 992"], flow_tokens=[],
         q="Section 10 of Act 703 requires a mineral-right holder to be incorporated "
           "'under the Companies Code 1963 (Act 179)'. Which company statute governs "
           "that incorporation today, and under which provision?"),
    dict(id="B2-external-company", axis="TRAP", present_pin=["external company"],
         absent_pin=["Act 992"], flow_tokens=[],
         q="Cite the provision of the company law currently in force in Ghana that "
           "defines an external company (a body corporate formed outside Ghana)."),
    dict(id="C1-1140-current", axis="CONTROL", present_pin=["Act 1140"],
         absent_pin=[], flow_tokens=[],
         q="Is the Ghana Gold Board Act 2025 (Act 1140) still in force, or has it "
           "already been amended or repealed by a later Act?"),
    dict(id="C2-703-principal", axis="CONTROL", present_pin=["Act 703"],
         absent_pin=[], flow_tokens=[],
         q="Is Act 703 still Ghana's principal minerals statute, or has it been wholly "
           "replaced by a new principal Minerals and Mining Act?"),
]

TAX = [
    dict(id="TA1-income-core", axis="GROUNDED", present_pin=["Act 896"],
         absent_pin=[], flow_tokens=["896", "592"],
         q="Is the Internal Revenue Act 2000 (Act 592) still Ghana's governing "
           "income-tax statute, or has it been replaced? State the current law and "
           "how the position changed."),
    dict(id="TA2-income-amend", axis="GROUNDED", present_pin=["Act 896", "Act 1094"],
         absent_pin=[], flow_tokens=["896", "1094"],
         q="Has the Income Tax Act 2015 (Act 896) been amended since it was enacted, "
           "and what is its current form?"),
    dict(id="TA3-vat-new", axis="GROUNDED", present_pin=["Act 1151", "Act 870"],
         absent_pin=[], flow_tokens=["1151", "870"],
         q="Which Value Added Tax statute governs in Ghana today — the Value Added "
           "Tax Act 2013 (Act 870) or something more recent? Explain what changed."),
    dict(id="TA4-vat-repeal-detail", axis="GROUNDED", present_pin=["Act 1151"],
         absent_pin=[], flow_tokens=["1151", "870"],
         q="A trader asks whether the Value Added Tax Act 2013 (Act 870) still binds "
           "it. State the governing VAT law now in force and the basis on which the "
           "2013 Act ceased to apply."),
    dict(id="TR1-1005-absent", axis="TRAP", present_pin=["Act 1151"],
         absent_pin=["Act 1005"], flow_tokens=[],
         q="Under which section did the Value Added Tax (Amendment) Act, 2019 "
           "(Act 1005) amend the principal VAT Act, and what did that provision change?"),
    dict(id="TR2-1133-absent", axis="TRAP", present_pin=["Act 1151"],
         absent_pin=["Act 1133"], flow_tokens=[],
         q="Cite the specific amending provision of the Value Added Tax (Amendment) "
           "Act, 2025 (Act 1133) and state what it altered in the VAT regime."),
    dict(id="TC1-896-current", axis="CONTROL", present_pin=["Act 896"],
         absent_pin=[], flow_tokens=[],
         q="Is the Income Tax Act 2015 (Act 896) still in force, or has it been "
           "wholly replaced by a new principal Income Tax Act?"),
    dict(id="TC2-1151-current", axis="CONTROL", present_pin=["Act 1151"],
         absent_pin=[], flow_tokens=[],
         q="Is the Value Added Tax Act 2025 (Act 1151) the current VAT law, or has "
           "it already been amended or repealed by a later Act?"),
]

SUITES = {
    "mining": ("ONSHORE MINING LAW", MINING),
    "tax": ("TAX REGIME IN GHANA", TAX),
}


def _retrieved_text(chunks):
    return "\n".join(c.get("text", "") for c in (chunks or []))


def _course_text(course):
    app.ensure_loaded(course)
    chunks = (app.INDEXES.get(course) or {}).get("chunks", []) or []
    return "\n".join(c.get("text", "") for c in chunks)


def has(text, token):
    return token.lower() in (text or "").lower()


def phase1(course, cases):
    print("=" * 78)
    print("PHASE 1 — CASE-DESIGN VALIDATION (retrieval only, no API cost)  [%s]" % course)
    print("=" * 78)
    ok = True
    for c in cases:
        r = app.search(course, c["q"])
        rt = _retrieved_text(r)
        present = {t: has(rt, t) for t in c["present_pin"]}
        sec_bound = {t: ts.successor_section_present(rt, t) for t in c["absent_pin"]}
        prem_ok = all(present.values())
        trap_ok = all(not v for v in sec_bound.values()) if c["absent_pin"] else True
        verdict = "OK " if (prem_ok and trap_ok) else "BAD"
        ok = ok and verdict == "OK "
        print("\n[%s] %s (%s) — %d chunks" % (verdict, c["id"], c["axis"], len(r)))
        print("      premise present %s" % present)
        if c["absent_pin"]:
            print("      absent-act SECTION-bound in retrieval (want False) %s" % sec_bound)
    print("\n" + ("Phase 1: all cases well-formed.\n" if ok else "Phase 1: BAD rows above.\n"))
    return ok


def phase2(course, cases, cache_path, force_gen=False):
    print("=" * 78)
    print("PHASE 2 — GENERATION + SCORING (web OFF)  [%s]\ncache: %s" % (course, cache_path))
    print("=" * 78)
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    full = _course_text(course)
    rows = []
    for c in cases:
        r = app.search(course, c["q"])
        rt = _retrieved_text(r)
        if not force_gen and c["id"] in cache:
            ans, srcflag = cache[c["id"]], "cached"
        else:
            res = app.answer_question(course, c["q"], include_web=False, mode="answer")
            ans = res.get("answer", "") or ""
            cache[c["id"]] = ans
            json.dump(cache, open(cache_path, "w"), indent=1)
            srcflag = "generated"
        pins = ts.classify_pinpoints(ans, rt)
        leaks = ts.real_leaks(pins, full)
        flow = bool(FLOW_RE.search(ans))
        if c["axis"] == "GROUNDED":
            flow_ok = all(has(ans, t) for t in c["flow_tokens"]) and flow
            verdict = "PASS" if (flow_ok and not leaks) else "REVIEW"
            note = "flow_traced=%s real_leaks=%s src=%s" % (flow_ok, leaks, srcflag)
        elif c["axis"] == "TRAP":
            fab = []
            for t in c["absent_pin"]:
                fab += ts.fabricated_successor_sections(ans, t, rt)
            verdict = "PASS" if not fab else "FAIL"
            note = "fabricated=%s named_ok=%s hedged=%s src=%s" % (fab, not fab, flow, srcflag)
        else:  # CONTROL
            invented = [t for t in leaks if t.lower().startswith("act")]
            verdict = "PASS" if not invented else "FAIL"
            note = "invented_acts=%s claims_change=%s src=%s" % (invented, flow, srcflag)
        rows.append((c["id"], c["axis"], verdict, note))
        print("\n[%s] %s (%s)\n%s" % (verdict, c["id"], c["axis"], note))
    print("\n" + "=" * 78 + "\nSUMMARY")
    for rid, axis, verdict, note in rows:
        print("  [%-6s] %-22s %-9s %s" % (verdict, rid, axis, note))
    bad = [x for x in rows if x[2] in ("FAIL", "REVIEW")]
    print("\n%d/%d clean; %d need eyes.\n" % (len(rows) - len(bad), len(rows), len(bad)))
    return rows


def _load_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    envf = os.path.join(REPO, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="mining")
    ap.add_argument("--run-gen", action="store_true", help="PAID generation")
    ap.add_argument("--rescore", action="store_true", help="re-score cache, FREE")
    ap.add_argument("--force-gen", action="store_true", help="regenerate even if cached (PAID)")
    a = ap.parse_args()
    course, cases = SUITES[a.suite]
    cache_path = os.path.join(HERE, "battery_answers_%s.json" % a.suite)
    phase1(course, cases)
    if a.rescore:
        phase2(course, cases, cache_path, force_gen=False)
    elif a.run_gen:
        _load_key()
        phase2(course, cases, cache_path, force_gen=a.force_gen)
    else:
        print("Phase 2 skipped. --run-gen (paid) | --rescore (free, cache only).")
