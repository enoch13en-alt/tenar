# Hosting TENAR (real, always-on)

TENAR is now deploy-ready: it's a stateless code image plus a **persistent data
volume**. The app reads/writes everything (corpus, indexes, model cache, accounts,
usage) under `TENAR_DATA` — set that to a mounted volume and nothing is lost on
restart or redeploy.

## Non-negotiables for the host

1. **A persistent disk** (your `courses/` is ~1.1 GB and grows per student — it
   CANNOT live on an ephemeral/serverless filesystem). Start with ~5–10 GB.
2. **Always-on**, single instance (`--workers 1` — see below). ~1–2 vCPU, **2–4 GB
   RAM** (embedding + loaded indexes).
3. **HTTPS** + a domain.
4. Env vars for secrets (below).

> Keep **one worker**. The app holds state in memory in one process and the storage
> layer is single-instance-safe (atomic writes). Multiple workers/instances would
> need a database first — you don't need that yet; one good instance serves a lot of
> students.

## Environment variables

| Var | Value |
|---|---|
| `ANTHROPIC_API_KEY` | your Anthropic key |
| `TENAR_DATA` | `/data` (the mounted volume) — the Dockerfile sets this |
| `SIGNUP_CODE` | comma-separated invite codes (gates signup; unset = closed) |
| `PORT` | set by the platform; the image honours it |

The Flask secret and extension token self-generate onto `/data` on first run.

## Option A — Fly.io (Docker + volume, recommended)

```
fly launch --no-deploy            # creates the app; keep the Dockerfile
fly volumes create tenar_data --size 10        # persistent disk
# in fly.toml add:  [mounts] source="tenar_data"  destination="/data"
fly secrets set ANTHROPIC_API_KEY=sk-ant-...  SIGNUP_CODE=inv-a,inv-b
fly deploy
```
Fly gives you HTTPS + a `*.fly.dev` domain automatically.

## Option B — Any VPS (Hetzner/DigitalOcean) with Docker + Caddy

```
docker build -t tenar .
docker volume create tenar_data
docker run -d --name tenar --restart unless-stopped \
  -e ANTHROPIC_API_KEY=sk-ant-... -e SIGNUP_CODE=inv-a,inv-b \
  -v tenar_data:/data -p 127.0.0.1:8080:8080 tenar
```
Put **Caddy** in front for automatic HTTPS (one line: `yourdomain.com { reverse_proxy 127.0.0.1:8080 }`).

## Move your existing corpus onto the volume (one-time)

Your 1.1 GB corpus + its metadata must land on `/data` (else the host starts empty).
Copy `courses/`, `matters/` (if any), and the JSON state files:

```
# from your Mac, into the volume (adjust host/path per platform):
rsync -avz courses sources.json doctypes.json meta.json users.json \
  YOUR_HOST:/path/to/data/
```
On Fly, `fly ssh console` + `fly sftp` (or a one-off machine with the volume
mounted) to place the files under `/data`. After copying, restart the app and hit
**Re-index** per course (or it indexes on first query).

Bring your **owner account** by copying the existing `users.json` (contains
`owner@local`). **Rotate the owner password for production** before going live.

## First-run checklist

- [ ] Volume mounted at `/data`, `TENAR_DATA=/data`
- [ ] `ANTHROPIC_API_KEY` set; `SIGNUP_CODE` set (or signup stays closed)
- [ ] Corpus + `users.json` copied to `/data`; re-indexed
- [ ] HTTPS working; owner password rotated
- [ ] Log in as owner → enrol each student in their sourced courses (`/api/enroll`)

The embedding model (~64 MB) downloads to `/data/.fastembed_cache` on first run —
needs outbound internet once, then it's cached.
