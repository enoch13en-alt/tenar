#!/usr/bin/env python3
"""On-demand behavioural eval for ALTERNATIVE_CONSTRUCTION (Safeguard 2, paid).

Two arms (reasoning stack with vs without ALTERNATIVE_CONSTRUCTION) on a fact pattern that
has BOTH a genuinely-arguable point (fee as condition of issue vs effect) and clear provisions.
Gate: the module must FIRE on the arguable point (state+reject the competing construction),
stay SILENT on clear text (no padding), still COMMIT, and not inflate length. Needs
ANTHROPIC_API_KEY in ../.env; ~3 Opus calls (~$0.30). Run: python3 evals/alt_construction_eval.py
"""
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import ast, json, urllib.request

src = open(os.path.join(_ROOT, "app.py")).read()
C = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
        n = node.targets[0].id
        if n in ("CALIBRATION","PROPOSITION_VALIDATION","STATUTORY_INTERPRETATION","ALTERNATIVE_CONSTRUCTION"):
            try: C[n] = ast.literal_eval(node.value)
            except Exception: pass

TASK = ("\n\nTASK: Advise the Authority on the problem below. Separate express rules from "
        "interpretive conclusions; apply statutory-interpretation principles where appropriate; "
        "state confidence at the right level; do not infer powers/duties/rights/prohibitions "
        "unsupported by the text; if the Act is silent, identify the silence; distinguish the "
        "existence, the issue, and the legal effect of the permit; explain why each argument "
        "succeeds or fails. Be an appellate-quality opinion, not padded.")
BASE  = C["CALIBRATION"] + "\n\n" + C["PROPOSITION_VALIDATION"] + "\n\n" + C["STATUTORY_INTERPRETATION"]
ARMS = {"A_no_alt": BASE + TASK, "B_with_alt": BASE + "\n\n" + C["ALTERNATIVE_CONSTRUCTION"] + TASK}

PROBLEM = """The Environmental Licensing Act provides:
s.10: "The Authority may issue an environmental permit where it is satisfied that the applicant has complied with the requirements of this Act."
s.12: "An environmental permit shall not take effect until the prescribed environmental fee has been paid."
s.15: "Nothing in this Act requires the consent of adjoining landowners before the Authority issues a permit."
A company satisfies every statutory requirement except that it has not yet paid the prescribed fee. Adjoining landowners argue: (1) the Authority cannot issue without their consent; (2) because the Authority "may" issue, it has unfettered discretion to refuse every application even where all requirements are met; (3) once the Authority signs the permit the company may commence operations without paying the fee; (4) because landowners are mentioned elsewhere, Parliament intended them a veto. Advise the Authority."""

APIKEY = next(l.split("=",1)[1].strip() for l in open(os.path.join(_ROOT, ".env")) if l.startswith("ANTHROPIC_API_KEY="))
def call(system, user, mt=4000):
    body = json.dumps({"model":"claude-opus-4-8","max_tokens":mt,"system":system,
        "messages":[{"role":"user","content":user}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key":APIKEY,"anthropic-version":"2023-06-01","content-type":"application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    return "".join(b.get("text","") for b in r["content"] if b.get("type")=="text").strip()

out = {n: call(s, PROBLEM) for n, s in ARMS.items()}
words = {n: len(o.split()) for n, o in out.items()}

JUDGE = ("You are a strict appellate judge scoring two legal opinions on the SAME problem. For EACH "
 "opinion score four booleans. (1) arguable_handled: on the GENUINELY ARGUABLE point — whether the "
 "unpaid fee is a condition of ISSUE (s.10) or only of EFFECT (s.12) — does the opinion explicitly "
 "STATE the strongest COMPETING construction (that payment is an implied precondition to issue) AND "
 "give a reasoned basis for preferring/rejecting it? (2) clear_not_padded: does it AVOID manufacturing "
 "a competing construction / false 'on the other hand' for the CLEAR provisions (s.15 no-consent; s.12 "
 "no-effect-until-fee)? true = no padding on clear text. (3) committed: does it reach FIRM conclusions "
 "rather than leaving issues open? (4) appellate_quality: reads like a court weighing then deciding, "
 "not a checklist. Judge substance only. STRICT JSON, no fences: "
 '{"A_no_alt":{"arguable_handled":true,"clear_not_padded":true,"committed":true,"appellate_quality":true,"note":"<=20w"},'
 '"B_with_alt":{...}}')
juser = PROBLEM + "\n\n=== OPINION A ===\n" + out["A_no_alt"] + "\n\n=== OPINION B ===\n" + out["B_with_alt"]
j = json.loads((lambda s: s[s.find("{"):s.rfind("}")+1])(call(JUDGE, juser, mt=1500)))

print("="*72,"\nALTERNATIVE CONSTRUCTION — gate\n"+"="*72)
print(f"{'arm':12}{'arguable':>10}{'clear_ok':>10}{'committed':>11}{'appellate':>11}{'words':>8}   note")
for n in ARMS:
    a=j[n]; print(f"{n:12}{str(a['arguable_handled']):>10}{str(a['clear_not_padded']):>10}{str(a['committed']):>11}{str(a['appellate_quality']):>11}{words[n]:>8}   {a.get('note','')}")

b=j["B_with_alt"]; a=j["A_no_alt"]
fires   = b["arguable_handled"]
no_pad  = b["clear_not_padded"]
commits = b["committed"]
no_bloat= words["B_with_alt"] <= 1.35*words["A_no_alt"]
improves= b["arguable_handled"] and not a["arguable_handled"]  # nice-to-have: fires where base didn't
print("\nGATE:")
print(f"  fires on the arguable point (states+rejects competing construction): {fires}")
print(f"  stays silent on clear text (no padding):                            {no_pad}")
print(f"  still commits to firm conclusions:                                  {commits}")
print(f"  no length blow-up (<=1.35x base): {no_bloat}  (A={words['A_no_alt']}w, B={words['B_with_alt']}w)")
print(f"  (bonus) fires where base did NOT: {improves}")
print(f"\n  >>> DEPLOY ALTERNATIVE_CONSTRUCTION: {fires and no_pad and commits and no_bloat}")
