# TENAR production image. CODE only — the corpus, indexes, model cache, accounts and
# usage state live on a mounted PERSISTENT volume at /data (TENAR_DATA). The image is
# stateless, so redeploying it never touches your data.
FROM python:3.11-slim

WORKDIR /app

# deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code (corpus/secrets/state are excluded by .dockerignore — they belong on /data)
COPY . .

ENV TENAR_DATA=/data
ENV PORT=8080
EXPOSE 8080

# gunicorn + gevent, ONE worker — matches the single-process in-memory state and the
# atomic-write storage model. --timeout 0 because compiles/drafts stream for minutes.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 --worker-class gevent --worker-connections 1000 \
    --timeout 0 --graceful-timeout 30
