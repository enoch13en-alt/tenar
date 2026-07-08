#!/bin/bash
# Smart launcher: ensures setup, starts the server if it isn't already
# running, waits for it, then opens the browser. Called by the desktop app.
cd "$(dirname "$0")"

# first-time safety: build the environment if it's missing
if [ ! -d venv ]; then
  /usr/bin/python3 -m venv venv
  ./venv/bin/pip install --upgrade pip >/dev/null 2>&1
  ./venv/bin/pip install -r requirements.txt >/dev/null 2>&1
fi

# start the server only if it isn't already responding
if ! /usr/bin/curl -s http://127.0.0.1:5000/api/docs >/dev/null 2>&1; then
  nohup ./venv/bin/python app.py > /tmp/legalbot.log 2>&1 &
  for i in $(seq 1 30); do
    /usr/bin/curl -s http://127.0.0.1:5000/api/docs >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

/usr/bin/open http://127.0.0.1:5000
