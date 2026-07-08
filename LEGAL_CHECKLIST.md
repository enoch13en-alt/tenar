# Legal & Compliance Checklist — before you charge anyone

Ordered by how badly it can hurt you. Get 1–3 right before a paid launch.

## 1. Copyright of hosted materials — THE big one (you chose model B)
You decided to **source and host everything, including paywalled textbooks and
journal articles**. Hosting + serving those to paying users **without a licence
is infringement**, and publishers enforce it. You cannot ship model B at scale
without one of:
- **Copyright Clearance Center (CCC)** annual/again licence — the realistic
  first stop for a startup. Get a quote for the works you host.
- **Direct publisher licences** (Springer, Elsevier, Wiley, OUP, CUP, Sweet &
  Maxwell/Thomson) — slower, pricier, but definitive.
- **Interim safe mode** until a licence is in place: host only freely-
  distributable sources (legislation, case law, IGO/OECD/IAEA reports, open
  access) and have the **student upload** their own licensed copies of paywalled
  texts. This is legal today and needs no deal.
- Action: **email CCC for a quote this week**; run the interim safe mode until it lands.

## 2. "Not legal advice" + Terms of Service
- State plainly: **study aid for law students; not legal advice; no
  solicitor–client relationship; verify all output.**
- Limit liability; disclaim accuracy; no warranty for exam/coursework outcomes.
- Acceptable use; account terms; suspension rights.

## 3. Data protection & privacy
- **Ghana Data Protection Act, 2012** — register with the Data Protection
  Commission if processing personal data; lawful basis; security; retention.
- **GDPR/UK GDPR** if you take EU/UK students (LLM cohorts often are) — privacy
  notice, lawful basis, data-subject rights, processor terms.
- **Anthropic data handling:** inputs are retained ~30 days then deleted, not
  used for training; sign Anthropic's **commercial terms + DPA**; request
  **zero-data-retention** if you handle anything genuinely confidential. Note
  Claude Fable 5 can't run under ZDR — Opus (what you use) is fine.
- Publish a **privacy policy** covering: what you store (accounts, uploads,
  questions), where (your server + Anthropic), retention, and deletion on
  request.

## 4. Anthropic commercial terms
- Move from personal key to a **commercial account**; accept commercial ToS;
  set up billing + spend alerts; understand rate limits (you hit the web-search
  limit in testing — check the quota for your tier).

## 5. Academic integrity
- Position as **revision/practice/structuring aid**; the student owns, verifies
  and edits output. Universities increasingly police AI-generated submissions —
  make responsible-use explicit in your Terms so you're not seen as a
  cheating service.

## 6. Payments & consumer
- Clear pricing, what a "credit" is, refund policy; VAT/tax as applicable;
  use a reputable processor (Paystack/Flutterwave locally, Stripe international).

## 7. Company basics
- Register the entity; keep student data and payment data on compliant
  infrastructure; basic cyber hygiene (below in DEPLOY.md).

---
**Minimum to launch a pilot legally:** interim safe-sourcing mode (1), a short
"not legal advice" Terms + privacy notice (2–3), and Anthropic commercial terms
(4). The CCC licence (1) is what unlocks *full* model B afterwards.
