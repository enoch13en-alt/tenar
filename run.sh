#!/bin/bash
# One command to launch the bot. Double-click in Finder or run: ./run.sh
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "First-time setup — creating environment (this takes a few minutes)…"
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip >/dev/null
  ./venv/bin/pip install -r requirements.txt
fi

if [ ! -f .env ]; then
  echo ""
  echo "  ⚠  No .env file found. Copy .env.example to .env and paste your"
  echo "     Anthropic API key into it, then run again."
  echo ""
  cp .env.example .env
  open -e .env 2>/dev/null || true
  exit 1
fi

# Launch under gunicorn (robust under streaming load; the Flask dev server wedges).
# 1 worker + threads matches the app's in-memory state (USERS/INDEXES live in one
# process). --timeout 0 is ESSENTIAL: compiles/advisory drafts stream for minutes
# and must not be killed by gunicorn's default 30s worker timeout. --preload runs
# startup once. Falls back to the dev server if gunicorn isn't installed.
if [ -x ./venv/bin/gunicorn ] && ./venv/bin/python -c "import gevent" 2>/dev/null; then
  echo ""
  echo "  TENAR (gunicorn + gevent) →  http://127.0.0.1:5000"
  echo "  (indexes load on first use; use the Re-index button after adding files)"
  echo ""
  # gevent async worker: handles many concurrent long streams (compiles/advisory)
  # and slow/abandoned clients WITHOUT exhausting a thread pool. CPU-bound
  # embedding is offloaded to a real-thread pool in app.py so it can't freeze the
  # hub. --timeout 0 because streams run for minutes. NO --preload (gevent must
  # monkey-patch before the app imports httpx/ssl).
  exec ./venv/bin/gunicorn app:app \
    --bind 127.0.0.1:5000 \
    --workers 1 --worker-class gevent --worker-connections 1000 \
    --timeout 0 --graceful-timeout 30
else
  ./venv/bin/python app.py
fi
