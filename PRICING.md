# Pricing — TENAR

**Philosophy:** entry tiers priced at the **2.5× cost floor** (undercut NotebookLM
where buyers are price-sensitive); flagship tiers **value-priced** (the done-for-you
sourcing, Polish, verified case-finder and submission-grade PDF justify it). Every
tier is loss-proof — worst-case fully-utilised cost stays well under price.

Costs assume **Opus 4.8 + prompt caching** (≈25–35% off vs uncached, biggest saving
on the many small calls: gathers, questions, polish).

## Per-operation cost (internal, Opus 4.8, cached)

| Operation | Cost |
|---|---|
| Exam gather / question / Polish | ~$0.08–0.12 |
| Weekly Update — outline parse / week summary | ~$0.10–0.20 |
| Comparative answer / Case-finder (web) | ~$0.45 |
| Exam compile (Opus) | ~$0.90 |
| **Deepen** (examined-argument pass, thinking-on) | ~$0.50–2.00 |
| Exam compile (**Fable 5**) | ~$3.50 |

**Deepen** is the priciest per-call driver (extended thinking), so it has its own
capped meter — it never draws from the cheap questions pool.

## Plans

| Plan | **Price** | Founding (−30%) | Questions¹ | Web² | Exam compiles | Deepen³ | Fable | Courses | Export |
|---|---|---|---|---|---|---|---|---|---|
| **Free** | **$0** | — | 10 | 0 | preview | 0 | 0 | 1 | Word only |
| **Single course** | **$40** | ~$28 | 80 | 8 | 2 | 2 | 1 | 1 | PDF + Word |
| **Semester bundle** | **$120** | ~$85 | 250 | 30 | 6 | 5 | 2 | 5 | PDF + Word |
| **Dissertation** | **$249** | ~$180 | 350 | 50 | 10 | 10 | 4 | ∞ | PDF + Word |
| **Full LLM** | **$599** | ~$449 | 700 | 100 | 20 | 20 | 8 | ∞ | PDF + Word |

¹ Questions also power **Polish my draft**, **Weekly Update** summaries, and Exam-Coach gathers.
² Web credits power **Comparative answers** and the **verified case-finder**.
³ **Deepen** = the examined-argument pass (steel-man + deep comparator + resolved
tensions); own cap because it runs extended thinking. Extra as credits.
Done-for-you sourcing is included on **every** tier, Free included.

Full LLM = ~$50/month over a ~12-month degree; Dissertation ~$21/month.

## What's included where

- **Free:** taster — 10 questions, 1 sourced course, Exam Coach preview, **Word
  export only** (submission-grade **PDF is a paid feature**).
- **Paid (all):** full Exam Coach, **Weekly Update** (syllabus → exam-standard
  weekly summaries), Comparative + case-finder (web), Polish my draft, 🔬 **Deepen**
  (examined-argument pass), **cover-page PDF + Word export**, OSCOLA on tap.
- **Deepen** metered per tier (Free 0 · Single 2 · Semester 5 · Dissertation 10 ·
  Full-LLM 20); own cap because it runs extended thinking (~$0.50–2 each).
- **Fable 5 "max quality" compile:** metered per tier (Free 0 · Single 1 ·
  Semester 2 · Dissertation 4 · Full-LLM 8); extra as credits.

## Credit top-ups (all ≥ 2.5× cost)

| Credit | Cost | **Price** | Pack |
|---|---|---|---|
| +10 comparative / case-finder (web) | ~$4.50 | **$12** | — |
| +5 exam compiles | ~$4.50 | **$12** | — |
| +3 Deepen passes | ~$6.00 | **$15** | **6 for $27** |
| Fable 5 compile | ~$3.50 | **$9** | **3 for $24** |

## Practitioner (lawyer) pricing — private matters + Advisory

A separate market on the **same app** (plan unlocks matters + Advisory instead of
courses + Exam Coach). Priced on hours saved, not cost-plus. Advisory drafts are
metered as **`drafts`** (~$1.20 each — ~10× a question) so caps protect margin.

### Subscription (per seat, monthly)

| Plan | **$/mo** | Matters | Advisory drafts | Deepen | Matter Q&A | Web research | Fable |
|---|---|---|---|---|---|---|---|
| **Solo** | **$149** | 5 | 15 | 10 | 200 | 20 | 2 |
| **Practice** | **$299** | 15 | 40 | 25 | 500 | 50 | 5 |
| **Firm** (per seat, min 3) | **$129/seat** | ∞ | 40/seat | 25/seat | 600/seat | 60/seat | 8/seat |

Deepen sharpens an Advisory draft to examined-argument standard — the same
capped meter (own thinking-on cost).

### À la carte

| Product | **Price** | What you get |
|---|---|---|
| **Single Matter** | **$99** | 1 matter, upload documents, **8 drafts** + 150 queries + 15 web research (60 days) |

### Loss-proofing (practitioner)

| Plan | Max cost | Price | Margin |
|---|---|---|---|
| Solo | ~$47 | $149 | ~68% |
| Practice | ~$120 | $299 | ~60% |
| Single Matter | ~$26 | $99 | ~74% |

Cost driver is drafts (~$1.20 Opus / ~$3.50 Fable); every driver capped.

### Selling to a lawyer

One drafted advice note that would take ~3 billable hours pays back a month of
Solo. Per-matter $99 is a client disbursement. Confidentiality (Anthropic ZDR +
DPA, encrypted storage) must be in place before real client files — see
LEGAL_CHECKLIST.md.

## Margins & loss-proofing (students)

Every cost driver is capped by metering — `questions`, `comparative`,
`exam_sessions`, `fable_compiles`, **`deepens`** — so no user can run a loss.
Deepen was the one leak (thinking-on, metered as a cheap question); it now has its
own cap, folded into the worst-case below. Indicative worst-case (fully maxed):

| Plan | Max cost | Price | Margin |
|---|---|---|---|
| Semester | ~$57 | $120 | ~53% |
| Dissertation | ~$92 | $249 | ~63% |
| Full LLM | ~$184 | $599 | ~69% |

Most users never max out, so realised margins run higher.
