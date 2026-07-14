#!/usr/bin/env python3
"""On-demand behavioural eval for the reasoning modules (Safeguard 2, paid).

The discriminating 8-item suite that gates PROPOSITION_VALIDATION + STATUTORY_INTERPRETATION.
Item I7 (purpose must RESOLVE a genuine ambiguity) and I8 (purpose must NOT override clear
text / read words in) are the permanent SENTINEL pair: together they catch over-correction in
either direction (literalism vs legislating). Runs 3 arms (baseline / +PV / +PV+SI) through an
independent judge. Needs ANTHROPIC_API_KEY in ../.env; ~4 Opus calls (~$0.40).

Run: python3 evals/reasoning_modules_eval.py
"""
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import ast, json, urllib.request

src = open(os.path.join(_ROOT, "app.py")).read()
C = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
        n = node.targets[0].id
        if n in ("CALIBRATION","PROPOSITION_VALIDATION","STATUTORY_INTERPRETATION"):
            try: C[n] = ast.literal_eval(node.value)
            except Exception: pass

OUTFMT = ("\n\nOUTPUT FORMAT — return the corrected advice first, then '===CHANGES===', then one-line "
          "bullets, each naming what you changed and the legal reason. No fences.")
BASE = C["CALIBRATION"]
ARMS = {
 "1_baseline":   BASE + OUTFMT,
 "2_prop":       BASE + "\n\n" + C["PROPOSITION_VALIDATION"] + OUTFMT,
 "3_prop_canon": BASE + "\n\n" + C["PROPOSITION_VALIDATION"] + "\n\n" + C["STATUTORY_INTERPRETATION"] + OUTFMT,
}

STATUTE = """RETRIEVED PROVISIONS — Mineral Development (hypothetical) Act.
Long title: "An Act to secure the productive development of mineral resources and to protect the environment, and to afford aggrieved persons a right of review."
s.3: "infrastructure" means roads, bridges, railways and other works.
s.5: A holder shall maintain records of production, sales and disposal.
s.7: Except as provided in section 22, no person shall export unprocessed ore.
s.8: A holder may, with the consent of the Minister, divert a watercourse.
s.11: An appeal under this section may be brought only on the grounds of (a) an error of law, (b) procedural unfairness, or (c) bad faith.
s.14: The Commission shall publish the register within 14 days of each grant.
s.16: The Tribunal may, on the application of an aggrieved holder, review a refusal of consent.
s.18: A person aggrieved by a decision may apply for review within 30 days of the decision.
s.22: The Minister may by permit authorise the export of unprocessed ore for metallurgical testing."""

DRAFT = """DRAFT ADVICE TO REVIEW:
(I1) Section 14 says the Commission "shall" publish the register within 14 days, so a grant not published within 14 days is invalid and the lease fails.
(I2) Section 16 says the Tribunal "may" review, so review is purely discretionary and the Tribunal may refuse to entertain any application at all.
(I3) A fibre-optic telecommunications cable is plainly "other works" under s.3 and therefore infrastructure; the point is clear and admits no doubt.
(I4) The duty in s.5 to record "disposal" means the holder must log all environmental waste-water discharges, and failure to do so breaches s.5.
(I5) Section 7 imposes an absolute ban on exporting unprocessed ore, with no exceptions.
(I6) Although s.11 lists error of law, procedural unfairness and bad faith, a court could also entertain an appeal on the further ground of general unreasonableness.
(I7) Under s.18 the 30-day period runs from the date the decision was made, even where the aggrieved person was never notified of it.
(I8) Because the long title stresses protecting the environment, s.8 must be read as requiring the holder to obtain an environmental permit before any diversion, in addition to the Minister's consent.
(VA) Under s.8, a holder needs the Minister's consent before diverting a watercourse."""

KEY = """SCORING KEY — for each item, the CORRECT reading/behaviour:
I1 (directory 'shall'): s.14 publication is a procedural/directory timing duty; breach does NOT invalidate the grant or the lease. 'shall' imposes a duty but the Act attaches no nullity. correct=reaches non-invalidity; calibrated=does not overstate.
I2 ('may' made obligatory by context): s.16 'may' is coupled with an aggrieved holder's right and a review function; on the better view the Tribunal cannot simply refuse to entertain a proper application ('may' can be obligatory in exercise). correct=rejects 'purely discretionary/refuse anyone'; calibrated=notes the obligatory-exercise line rather than flipping to a flat duty.
I3 (two-canon collision — MUST state ambiguity): ejusdem generis limits 'other works' to the genus of roads/bridges/railways (physical transport/civil works); a fibre cable is genuinely arguable either way. correct=does NOT accept 'clear/no doubt'; calibrated=STATES the competing readings/ambiguity, applies no canon conclusively.
I4 (noscitur a sociis): 'disposal' among 'production, sales' means commercial disposal of the mineral product, NOT environmental waste-water discharge. correct=narrows via neighbours; calibrated=not overstated.
I5 (cross-referenced proviso): s.7 opens 'Except as provided in section 22'; s.22 permits export for metallurgical testing — so NOT absolute. correct=applies the s.22 exception; calibrated=fine.
I6 (expressio unius GENUINELY applies): s.11 confines appeal 'only on the grounds of' the three listed; the closed list excludes general unreasonableness. correct=applies expressio unius to EXCLUDE the extra ground; calibrated=recognises this list is closed (contrast an 'includes' list). [Discriminator: must NOT over-generalise 'expressio unius never applies to lists'.]
I7 (purpose SHOULD resolve — reverse guard): s.18 'the decision' is genuinely ambiguous (made vs communicated); the Act's stated object of affording aggrieved persons review favours running time from COMMUNICATION; the literal 'date made' reading barring an unnotified person defeats the object. correct=USES purposive interpretation to prefer communication; calibrated=acknowledges ambiguity, does not stay woodenly literal. [Discriminator: module must not become mechanically anti-purpose.]
I8 (do NOT read in): s.8 is clear; the long title cannot add an environmental-permit precondition; reading one in legislates from the bench. correct=refuses to read in the permit; calibrated=purpose does not override clear text.
VA (VALID — leave substantially unchanged): correct statement of s.8; must NOT be rewritten on the merits."""

APIKEY = next(l.split("=",1)[1].strip() for l in open(os.path.join(_ROOT, ".env")) if l.startswith("ANTHROPIC_API_KEY="))
def call(system, user, mt=4000):
    body = json.dumps({"model":"claude-opus-4-8","max_tokens":mt,"system":system,
        "messages":[{"role":"user","content":user}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key":APIKEY,"anthropic-version":"2023-06-01","content-type":"application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    return "".join(b.get("text","") for b in r["content"] if b.get("type")=="text").strip()

outputs = {}
for name, sysp in ARMS.items():
    outputs[name] = call(sysp, STATUTE + "\n\n" + DRAFT)
    ch = outputs[name].split("===CHANGES===",1)
    print("="*74, f"\nARM {name} — CHANGES:\n" + "="*74)
    print((ch[1].strip() if len(ch)>1 else "(none)")[:2800])

JUDGE_SYS = ("You are a strict statutory-interpretation examiner. You are given the statute, a flawed draft, "
 "a KEY with the correct reading for 8 items (I1-I8) and one already-correct item (VA). For EACH arm and "
 "EACH item score two booleans: correct (did it reach the reading/behaviour the key requires?) and "
 "calibrated (did it show the right confidence — e.g. STATE the ambiguity for I3/I7, apply the closed-list "
 "canon for I6, USE purpose for I7, refuse to read-in for I8 — without overcorrecting?). Also: did the arm "
 "wrongly rewrite VA on the merits (VA_overrewritten)? Did the arm show canon_shopping (invoking a canon "
 "conclusively where a competing canon applies, or naming canons as filler)? Judge substance only, strictly. "
 "STRICT JSON only, no fences: "
 '{"arms":{"<arm>":{"correct":{"I1":true,...,"I8":true},"calibrated":{"I1":true,...,"I8":true},'
 '"VA_overrewritten":false,"canon_shopping":false,"note":"<=25 words"}}}')
judge_user = STATUTE + "\n\n" + DRAFT + "\n\n" + KEY + "\n\n===ARM OUTPUTS===\n" + "\n\n".join(f"### ARM {n}\n{outputs[n]}" for n in ARMS)
jraw = call(JUDGE_SYS, judge_user, mt=3000)
j = json.loads(jraw[jraw.find("{"):jraw.rfind("}")+1])

I = [f"I{i}" for i in range(1,9)]
print("\n"+"="*74+"\nSCORES (subtle/discriminating set)\n"+"="*74)
print(f"{'arm':14}{'correct/8':>11}{'calibrated/8':>14}{'total/16':>10}{'VA_kept':>9}{'canon_shop':>12}   note")
tot = {}
for n in ARMS:
    a=j["arms"][n]
    cor=sum(1 for k in I if a["correct"].get(k)); cal=sum(1 for k in I if a["calibrated"].get(k))
    tot[n]=cor+cal
    print(f"{n:14}{cor:>11}{cal:>14}{cor+cal:>10}{str(not a.get('VA_overrewritten')):>9}{str(not a.get('canon_shopping')):>12}   {a.get('note','')}")

pc=j["arms"]["3_prop_canon"]; p=j["arms"]["2_prop"]; b=j["arms"]["1_baseline"]
def ok(a,k): return a["correct"].get(k) and a["calibrated"].get(k)
sig = ok(pc,"I3") and pc["correct"].get("I6") and ok(pc,"I7") and pc["correct"].get("I8")
improved = tot["3_prop_canon"] > tot["1_baseline"] and tot["3_prop_canon"] >= tot["2_prop"]
clean = (not pc.get("VA_overrewritten")) and (not pc.get("canon_shopping"))
print("\nTHRESHOLD:")
print(f"  material improvement over baseline:   {improved}  (arm3={tot['3_prop_canon']}/16, arm2={tot['2_prop']}/16, baseline={tot['1_baseline']}/16)")
print(f"  signature canon behaviours in arm3:   {sig}  (I3 states ambiguity, I6 expressio applies, I7 uses purpose, I8 refuses read-in)")
print(f"  no over-correction / no canon-shop:   {clean}")
print(f"\n  >>> DEPLOY BOTH: {improved and sig and clean}")
