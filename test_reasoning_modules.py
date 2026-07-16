#!/usr/bin/env python3
"""FREE deterministic guard for the reasoning modules (Safeguard 2).

Sibling of test_grounding_monitor.py / test_temporal_succession.py: it makes NO API
call. It pins the structural invariants of PROPOSITION_VALIDATION + STATUTORY_INTERPRETATION
so a later edit can't silently (a) unwire a module, (b) revert the I7/I8 over-correction fix
(the ambiguity-vs-silence distinction that keeps the canons from tipping into literalism),
or (c) drop the refined modal rule. Exit 1 = regression.

Run: python3 test_reasoning_modules.py
"""
import ast, re, sys

SRC = open("app.py").read()
TREE = ast.parse(SRC)

# Extract the base literal of each named string constant (value BEFORE any later
# `X = X + ...` append statement runs — that is exactly what ast reads at the assignment).
def const(name):
    for node in TREE.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            try:
                return ast.literal_eval(node.value)
            except Exception:
                return None
    return None

def append_present(target, module):
    # e.g. CITATION_INTEGRITY = CITATION_INTEGRITY + "\n\n" + PROPOSITION_VALIDATION
    pat = re.compile(rf"{target}\s*=\s*{target}\s*\+.*\+\s*{module}\b")
    return bool(pat.search(SRC))

fails = []
def check(cond, msg):
    if not cond:
        fails.append(msg)

PV = const("PROPOSITION_VALIDATION") or ""
SI = const("STATUTORY_INTERPRETATION") or ""
AC = const("ALTERNATIVE_CONSTRUCTION") or ""

# 1) Both modules exist and are substantial (not blanked out).
check(len(PV) > 800, "PROPOSITION_VALIDATION missing or too short")
check(len(SI) > 1200, "STATUTORY_INTERPRETATION missing or too short")
check(len(AC) > 600, "ALTERNATIVE_CONSTRUCTION missing or too short")

# 2) Wired into the paths that already carry the grounded-only / doctrinal disciplines.
check(append_present("CITATION_INTEGRITY", "PROPOSITION_VALIDATION"),
      "PROPOSITION_VALIDATION not appended to CITATION_INTEGRITY")
check(append_present("DOCTRINAL_PRECISION", "STATUTORY_INTERPRETATION"),
      "STATUTORY_INTERPRETATION not appended to DOCTRINAL_PRECISION")
check(append_present("DOCTRINAL_PRECISION", "ALTERNATIVE_CONSTRUCTION"),
      "ALTERNATIVE_CONSTRUCTION not appended to DOCTRINAL_PRECISION")

# 2b) ALTERNATIVE_CONSTRUCTION must keep BOTH rails: the arguable-only GATE (anti-padding) and
#     steel-man-then-COMMIT (anti-open-ended). Losing either is the module's known failure mode.
check("GENUINELY ARGUABLE" in AC and ("PADDING" in AC or "not for every provision" in AC),
      "ALTERNATIVE_CONSTRUCTION lost its arguable-only gate (padding risk)")
check("COMMIT" in AC and ("straw man" in AC or "steel-man" in AC),
      "ALTERNATIVE_CONSTRUCTION lost steel-man-then-commit rail")
# 3) And explicitly present on the calibrate path (which does NOT carry CITATION_INTEGRITY).
check(re.search(r"CALIBRATION\s*\+\s*\"\\n\\n\"\s*\+\s*PROPOSITION_VALIDATION", SRC),
      "PROPOSITION_VALIDATION not wired into the calibrate prompt")

# 4) PROPOSITION_VALIDATION: 5-way classification + key checklist rails.
for token in ["EXPRESSLY STATED", "REASONABLE INTERPRETATION", "INFERENCE", "SECONDARY",
              "UNRESOLVED", "POLICY IS NOT BINDING LAW", "ONE INSTANCE IS NOT A PRACTICE",
              "EXISTENCE OF AN INSTRUMENT"]:
    check(token in PV, f"PROPOSITION_VALIDATION missing rail: {token!r}")

# 5) STATUTORY_INTERPRETATION: the canon set.
for canon in ["TEXT FIRST", "EJUSDEM GENERIS", "NOSCITUR A SOCIIS", "EXPRESSIO UNIUS",
              "SPECIFIC OVER GENERAL", "MANDATORY vs DIRECTORY", "PROVISOS",
              "CANON CONFLICT & RESTRAINT"]:
    check(canon in SI, f"STATUTORY_INTERPRETATION missing canon: {canon!r}")

# 6) THE I7/I8 SENTINEL — the over-correction fix must stay in, in BOTH directions:
#    I7 (don't tip into literalism: use purpose to resolve genuine ambiguity) and
#    I8 (don't legislate: purpose can't override clear text / read words in).
check("AMBIGUITY IS NOT SILENCE" in SI,
      "SENTINEL LOST: 'AMBIGUITY IS NOT SILENCE' — the I7 literalism guard was reverted")
check("PURPOSE RESOLVES AMBIGUITY" in SI,
      "SENTINEL LOST: affirmative 'PURPOSE RESOLVES AMBIGUITY' duty was reverted")
check("DO NOT LEGISLATE FROM THE BENCH" in SI,
      "SENTINEL LOST: 'DO NOT LEGISLATE FROM THE BENCH' (I8 read-in guard) was reverted")
check(re.search(r"DUTY to RESOLVE", SI) is not None,
      "SENTINEL WEAKENED: the duty to RESOLVE ambiguity (not merely note it) was removed")

# 7) Refined modal rule present (shall=duty but not auto-invalid; may=discretion).
check("MANDATORY vs DIRECTORY" in SI and re.search(r"'shall'.*duty", SI, re.S | re.I),
      "refined modal rule (shall=duty / may=discretion) weakened")

# 7b) Secondary-source attribution rule in PRIMARY_FIRST (the literature-engagement fix):
#     scholars' analysis must be named, and must NOT over-attribute reproduced primary text.
PF = const("PRIMARY_FIRST") or ""
check("ATTRIBUTE THE SCHOLAR'S OWN ANALYSIS" in PF,
      "PRIMARY_FIRST lost the secondary-source attribution rule (literature engagement)")
check("OVER-ATTRIBUTE PRIMARY TEXT" in PF,
      "PRIMARY_FIRST lost the over-attribution guard (reproduced primary text stays primary)")
# 7c) Context-salience: secondary sources (article/book) get an inline author cue so the model
#     can attribute in prose (the author otherwise sits only in citation metadata).
check('SECONDARY source' in SRC and 'display_type(ch["doc"]) in ("article", "book")' in SRC,
      "secondary-source salience cue missing from the retrieval context builder")

# 8) Versioned-together marker (Safeguard 1) + monitor plumbing (Safeguard 3).
check(const("REASONING_MODULES_VERSION") is not None,
      "REASONING_MODULES_VERSION marker missing (Safeguard 1)")
check("def reasoning_delta_log" in SRC, "reasoning_delta_log monitor missing (Safeguard 3)")
check("/api/admin/reasoning" in SRC, "/api/admin/reasoning readout missing (Safeguard 3)")

# 9) Course-agnostic legal-reasoning charter, appended to LEGAL_METHOD.
CHARTER = const("LEGAL_REASONING_CHARTER") or ""
check(len(CHARTER) > 1000, "LEGAL_REASONING_CHARTER missing or too short")
check(append_present("LEGAL_METHOD", "LEGAL_REASONING_CHARTER"),
      "LEGAL_REASONING_CHARTER not appended to LEGAL_METHOD")
check("COURSE-AGNOSTIC" in CHARTER and "SELF-AUDIT" in CHARTER,
      "charter lost its course-agnostic backbone / self-audit rail")

# 9b) Source-status + thresholds methodology (the authority-hierarchy / gateway fix) — appended
#     to LEGAL_METHOD so it rides gather, essay and compile.
LAM = const("LEGAL_AUTHORITY_METHOD") or ""
check(len(LAM) > 1200, "LEGAL_AUTHORITY_METHOD missing or too short")
check(append_present("LEGAL_METHOD", "LEGAL_AUTHORITY_METHOD"),
      "LEGAL_AUTHORITY_METHOD not appended to LEGAL_METHOD")
check("AUTHORITY LADDER" in LAM and "SATISFY THE THRESHOLDS" in LAM
      and "OUGHT" in LAM and "OVEREXTEND" in LAM,
      "LEGAL_AUTHORITY_METHOD lost a core rail (ladder / thresholds / ought-vs-is / overextension)")
check("WHO OWES IT" in LAM and "NECESSARY vs SUFFICIENT" in LAM and "COMPETENCE vs EXERCISE" in LAM
      and "INTERPRET vs SUPPLEMENT" in LAM,
      "LEGAL_AUTHORITY_METHOD lost a rail-E distinction (obligor / necessary-vs-sufficient / competence / interpret-vs-supplement)")
check("SELF-AUDIT" in LAM and "BINDING, PERSUASIVE or merely EVIDENTIAL" in LAM,
      "LEGAL_AUTHORITY_METHOD lost the six-question self-audit")
check("EXPRESS THE REGISTER IN THE PROSE" in LAM and "visible tags" in LAM,
      "LEGAL_AUTHORITY_METHOD lost the proved/inferred/conditional/recommended prose-register rule")
check("CUSTOM IS PROVISION-BY-PROVISION" in LAM,
      "LEGAL_AUTHORITY_METHOD lost the custom-is-provision-by-provision nuance")

# 10) Exam-firmness (anti-hedging) — wired into FACT_DISCIPLINE and the calibrator.
EF = const("EXAM_FIRMNESS") or ""
check(len(EF) > 800, "EXAM_FIRMNESS missing or too short")
check(append_present("FACT_DISCIPLINE", "EXAM_FIRMNESS"),
      "EXAM_FIRMNESS not appended to FACT_DISCIPLINE")
check('PROPOSITION_VALIDATION + "\\n\\n" + EXAM_FIRMNESS' in SRC,
      "EXAM_FIRMNESS not wired into the calibrator (which inserts the hedges)")
check("ADVISER UNDER REAL-WORLD" in EF and "FINDING-TOOLS ARE NOT AUTHORITY" in EF,
      "EXAM_FIRMNESS lost its core rails (exam-vs-adviser / finding-tools-not-authority)")

# 11) Honest research-guide ethos (default posture) — appended to CITATION_INTEGRITY.
RGE = const("RESEARCH_GUIDE_ETHOS") or ""
check(len(RGE) > 800, "RESEARCH_GUIDE_ETHOS missing or too short")
check(append_present("CITATION_INTEGRITY", "RESEARCH_GUIDE_ETHOS"),
      "RESEARCH_GUIDE_ETHOS not appended to CITATION_INTEGRITY")
check("WHERE TO LOOK" in RGE and "NEVER invent" in RGE,
      "research-guide ethos lost its where-to-look / no-invention rails")

if fails:
    print("❌ REGRESSION — reasoning modules:")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("✅ reasoning modules intact:",
      "PV+SI wired, 5-way classification, canon set, I7/I8 sentinel, modal rule, "
      "version marker + monitor all present.")
