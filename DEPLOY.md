# Deploy & Billing — from local prototype to hosted product

## Where it is now
- Single Flask app (`app.py`) with real **accounts/login**, **per-user plan +
  usage metering**, **per-user course access**, and **shared course packs**.
- Data lives in flat files: `users.json`, `config.json`, `sources.json`,
  `doctypes.json`, `meta.json`, `courses/<Course>/`. Fine for a pilot + small
  cohort; move to a database + object storage for real scale (below).

## Hosting (pilot → small cohort)
1. A small Linux cloud VM (2 vCPU / 4 GB is plenty; embeddings run on CPU).
2. Install Python 3, `pip install -r requirements.txt`.
3. Set env: `ANTHROPIC_API_KEY`, and copy `.flask_secret` (or let it generate).
4. Run behind a production server, not the Flask dev server:
   `pip install gunicorn` then
   `gunicorn -w 2 -b 127.0.0.1:5000 app:app`
   — **note:** `gunicorn` doesn't run the `__main__` block, so move the startup
   (secret key, `load_users/sources/doctypes/meta`, owner seed, reindex threads)
   into a function called at import, or use `--preload` + an init hook.
5. Put **nginx** in front with **HTTPS** (Let's Encrypt). Never serve without TLS.
6. Persist the data files + `courses/` on a mounted volume; **back them up**.
7. **Change the owner password** immediately (edit `users.json` → re-hash, or
   add a change-password endpoint — see TODO).

## Security must-dos
- HTTPS everywhere; secure/HttpOnly session cookies.
- Rotate off `owner@local / letmein`.
- Rate-limit `/api/login` and `/api/signup` (brute-force + signup abuse).
- Email verification on signup (stops free-tier farming).
- Spend alerts on the Anthropic account.

## Billing scaffold (Stripe / Paystack / Flutterwave)
The metering already exposes the exact hooks payments need:
- **Plan change** → set `USERS[email]["plan"]` (same as `POST /api/plan`).
- **Credit purchase** → increment `USERS[email]["credits"]` (same as
  `POST /api/credits`).
- **Enrolment** → `POST /api/enroll` (admin) adds a course to a user.

Add a webhook endpoint that the processor calls on successful payment:
```
@app.route("/api/billing/webhook", methods=["POST"])
def billing_webhook():
    verify_signature(request)                 # processor's signing secret
    ev = request.json
    email = ev["customer_email"]
    if ev["type"] == "plan":     USERS[email]["plan"] = ev["plan"]
    if ev["type"] == "credits":  add_credits(email, ev["kind"], ev["qty"])
    save_users(); return "", 200
```
Map your price IDs → plan names (`single`, `semester`, `dissertation`,
`full_llm`) and credit packs → `comparative` / `exam_sessions`.

## Scaling beyond a pilot (when you outgrow flat files)
- Move users/plans/usage to **Postgres**; course packs' chunks+embeddings to a
  proper **vector store** (pgvector / Qdrant / LanceDB) and files to **object
  storage** (S3-compatible).
- Run embeddings/reindex as a background worker, not in the web process.
- Add per-user rate limiting and audit logging.

## TODO endpoints worth adding next
- `POST /api/change-password` (self-serve).
- `POST /api/admin/users` (list/manage users).
- Email verification + password reset.
