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

# 8) Versioned-together marker (Safeguard 1) + monitor plumbing (Safeguard 3).
check(const("REASONING_MODULES_VERSION") is not None,
      "REASONING_MODULES_VERSION marker missing (Safeguard 1)")
check("def reasoning_delta_log" in SRC, "reasoning_delta_log monitor missing (Safeguard 3)")
check("/api/admin/reasoning" in SRC, "/api/admin/reasoning readout missing (Safeguard 3)")

if fails:
    print("❌ REGRESSION — reasoning modules:")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("✅ reasoning modules intact:",
      "PV+SI wired, 5-way classification, canon set, I7/I8 sentinel, modal rule, "
      "version marker + monitor all present.")
