# Receipt extraction service + eval harness, one image.
#
#   docker compose up                      # serve the API on :8000
#   docker compose run --rm eval           # score the running API -> ./outputs
#
# Nothing heavy runs in here: both models are hosted, so the image is the API, the
# harness, and the curated eval set. No model weights, no GPU.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# No system packages: PDFs render through pypdfium2, a self-contained wheel, so the
# image no longer needs poppler-utils.
WORKDIR /app

# Dependencies before source, so a code change doesn't reinstall everything.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY extractor ./extractor
COPY api ./api
COPY eval ./eval
# The curated eval set ships in the image so `docker compose run --rm eval` works from a
# fresh clone. data/raw is excluded by .dockerignore.
COPY data ./data

# Non-root. APP_UID is overridable so a Linux host can match the owner of the bind-mounted
# ./outputs and ./.cache (Docker Desktop on Windows/macOS does not need this).
ARG APP_UID=10001
RUN useradd --create-home --uid "${APP_UID}" app \
    && mkdir -p /app/.cache/extractions /app/outputs \
    && chown -R app:app /app/.cache /app/outputs
USER app

EXPOSE 8000

# Liveness only: /health answers 200 even when misconfigured, and reports WHY in its body.
# A missing API key should be visible to an operator, not look like a crash loop.
# python -c rather than curl, which the slim image does not include.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
