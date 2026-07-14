#!/usr/bin/env python3
"""Real-answer evaluation set (roadmap #4) — the messy, multi-source production test.

Curated synthetic tests saturate against a frontier model; recurring errors surface only
on real, incomplete, multi-source legal problems. This harness runs a fixed set of realistic
questions through the LIVE deployed bot (so grounded-only, retrieval and the reasoning modules
all engage against the real corpus), caches every answer for human labelling, and reports the
server-side monitor signals. Re-run after any prompt/corpus change and diff the answers.

Each question targets a KNOWN failure mode (the 'stresses' tag) so a human labeller knows what
to look for. Labelling itself is human: a law expert marks each answer pass/fail per stress —
that labelled set becomes the regression baseline the counters alone cannot provide.

Credentials come from the environment (never committed):
  TENAR_URL   (default https://tenar.onrender.com)
  TENAR_EMAIL
  TENAR_PW
Run:  TENAR_EMAIL=... TENAR_PW=... python3 evals/real_answer_eval.py [--limit N]
Cost: ~$0.45 per question against the live bot. Answers cache to real_answer_answers.json.
"""
import os, sys, json, time
import urllib.request
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_answer_answers.json")

URL = os.environ.get("TENAR_URL", "https://tenar.onrender.com").rstrip("/")
EMAIL = os.environ.get("TENAR_EMAIL", "")
PW = os.environ.get("TENAR_PW", "")

# Realistic, multi-source problems. 'stresses' = the failure modes each is built to expose.
QUESTIONS = [
    {"course": "ONSHORE MINING LAW", "stresses": ["multi-source", "silence-flagging", "remedy/effect"],
     "q": "A mining company's lease area overlaps a forest reserve and a community's farmland with growing crops and an occupied dwelling. Advise on (a) whether it needs any consent or approval beyond the lease to operate within the forest reserve, and (b) the community's compensation and any resettlement rights. Where the Act is silent on a point, identify the silence rather than filling it."},
    {"course": "ONSHORE MINING LAW", "stresses": ["foreign-ownership", "temporal-succession", "regime-conflation"],
     "q": "A foreign-incorporated company wishes to hold a mining lease and repatriate its profits. Advise on the ownership, incorporation and fiscal conditions under the Act, whether any State interest arises, and whether a repealed enactment still governs any of this."},
    {"course": "OIL AND GAS LAW", "stresses": ["regulatory", "silence-flagging", "may/shall"],
     "q": "A contractor under a petroleum agreement proposes to flare associated gas during early production. Advise on the legal constraints and any approvals required, and whether the regulator has a discretion or a duty in respect of any approval."},
    {"course": "DOWNSTREAMOILANDGAS", "stresses": ["existence/effect", "licensing", "may/shall"],
     "q": "A company imports refined petroleum products and sells them to retail stations. Advise on which licences it needs, whether it may lawfully operate on a licence that is pending renewal, and the legal effect of operating without the required licence."},
    {"course": "TAX REGIME IN GHANA", "stresses": ["temporal-succession", "grounding", "numbers-discipline"],
     "q": "Advise a mining company on how its income is taxed, how capital allowances and any ring-fencing apply, and whether any recent amendment has changed the position. Cite the governing provisions and flag anything not in the materials."},
    {"course": "COMPANY FOREIGNINVESTMENTLAW", "stresses": ["regime-conflation", "existence/effect", "grounding"],
     "q": "A foreign investor sets up a Ghanaian subsidiary to provide services. Advise on the minimum capital and registration conditions, and explain how the foreign-investment requirements interact with (and are distinct from) any sector-specific licensing. Keep the regimes distinct."},
    {"course": "REGULATIONOFINLANDGROUNDCOASTALWTERANDSEABEDRESOURCES", "stresses": ["authorisation", "conditions", "silence-flagging"],
     "q": "A company wishes to abstract water from a river to support its mining operations. Advise on what authorisation is required, who grants it, what conditions may attach, and whether the authorising body has a discretion to refuse."},
    {"course": "REGULATION OF TREE AND FOREST RESERVES", "stresses": ["permissions", "who-grants", "silence-flagging"],
     "q": "A developer wishes to fell trees within a forest reserve to build an access road. Advise on what permissions are required, who may grant them, and whether felling may begin before any required permission is obtained."},
]


def _req(path, data=None, cookie=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(URL + path, data=body, method="POST" if data is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if cookie:
        req.add_header("Cookie", cookie)
    r = urllib.request.urlopen(req, timeout=320)
    setc = r.headers.get("Set-Cookie")
    return json.load(r), (setc.split(";")[0] if setc else None)


def main():
    if not EMAIL or not PW:
        sys.exit("Set TENAR_EMAIL and TENAR_PW in the environment.")
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    d, cookie = _req("/api/login", {"email": EMAIL, "password": PW})
    if not d.get("ok"):
        sys.exit("Login failed: " + json.dumps(d))
    print(f"logged in as {d.get('email')} @ {URL}")

    results = []
    qs = QUESTIONS[:limit] if limit else QUESTIONS
    for i, item in enumerate(qs, 1):
        t0 = time.time()
        try:
            ans, _ = _req("/api/ask", {"course": item["course"], "web": False,
                                       "format": "essay", "question": item["q"]}, cookie)
            a = ans.get("answer", "")
            rec = {"course": item["course"], "stresses": item["stresses"], "q": item["q"],
                   "answer": a, "words": len(a.split()),
                   "sources": len(ans.get("sources", [])),
                   "cost_usd": (ans.get("cost") or {}).get("this_query_usd"),
                   "label": None, "notes": ""}   # <- human fills label per stress + notes
        except Exception as e:
            rec = {"course": item["course"], "stresses": item["stresses"], "q": item["q"],
                   "answer": "", "error": str(e)[:200], "label": None, "notes": ""}
        results.append(rec)
        print(f"[{i}/{len(qs)}] {item['course']:<48} {rec.get('words','ERR'):>5}w  "
              f"${rec.get('cost_usd','?')}  {time.time()-t0:.0f}s  stresses={','.join(item['stresses'])}")

    json.dump({"generated_stresses": sorted({s for q in QUESTIONS for s in q['stresses']}),
               "answers": results}, open(CACHE, "w"), indent=1)
    ok = [r for r in results if r.get("answer")]
    tot = sum(r.get("cost_usd") or 0 for r in ok)
    print(f"\nsaved {len(results)} answers to {CACHE}  (ok={len(ok)}, spend=${tot:.2f})")
    print("NEXT: a human labels each answer per its 'stress' tags (pass/fail + notes) — that "
          "labelled set is the regression baseline. Read monitor signals at /api/admin/reasoning.")


if __name__ == "__main__":
    main()
