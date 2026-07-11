# TEMPORAL_SUCCESSION evals

Checks that TENAR reads law **diachronically** — treats a later instrument as
amending/repealing an earlier one, reasons from the law *as it now stands*, and
never fabricates a section for a statute it doesn't actually hold.

## Files
- `temporal_scorer.py` — pure, deterministic scoring helpers (reuse app.py's
  grounding matcher). No API. Guarded by `../test_temporal_succession.py`.
- `temporal_battery.py` — the eval runner. Two suites: `mining`, `tax`.
- `battery_answers_<suite>.json` — cached model answers so re-scoring is free.

## Three axes
- **GROUNDED** — successor is in the corpus → must trace past→present.
- **TRAP** — a statute is *named* but its sections are absent → must NOT fabricate
  a section (naming it, or flagging the gap, is correct).
- **CONTROL** — no successor exists → must NOT invent a repeal/replacement.

## Run (from repo root, venv active)
```
# FREE — validate case design (retrieval only)
venv/bin/python evals/temporal_battery.py --suite tax

# FREE — re-score cached answers (no model calls)
venv/bin/python evals/temporal_battery.py --suite tax --rescore

# PAID — generate answers (web OFF, corpus-only), ~$0.15/question
venv/bin/python evals/temporal_battery.py --suite tax --run-gen
```
`--run-gen` reuses the cache and only generates missing cases. Delete a suite's
`battery_answers_*.json` to force a full regeneration.

## The free regression guard
```
venv/bin/python test_temporal_succession.py    # exit 1 = regression
```
Pins the scorer's hard-won fixes (verify-the-tool discipline): a grounded section
near an absent Act is not fabrication; a real in-corpus Act flagged out-of-window
is not "invented"; only a section TIGHTLY bound to an absent Act ("section N of
Act X") counts as a fabricated pinpoint.

## Status (2026-07-11)
Behaviourally clean on both corpora — mining 8/8, tax 8/8. Fabrication-refusal
confirmed on 4 genuine traps: Companies Act 992 (×2, mining), VAT Amdt Acts 1005
& 1133 (tax). The LLM battery is **manual** (paid, non-deterministic); only the
scorer guard runs in CI.
