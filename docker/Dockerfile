# Batch scoring API. Multi-stage so the runtime carries no build tooling, and installed
# from uv.lock with --frozen so the image pins exactly what the lockfile resolved.

# Same base as the runtime, with uv copied in: a virtualenv records the absolute path of
# its interpreter, so building against a different python would produce a venv the
# runtime cannot execute.
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /build
COPY pyproject.toml uv.lock ./
# Only the serving group: the champion is a logistic regression and needs neither the
# boosters, nor MLflow, nor Feast's dask stack to score a request. Built at /app/.venv,
# the path it will run from, so nothing has to be rewritten on copy.
RUN uv sync --frozen --no-dev --no-install-project --only-group serving


FROM python:3.12-slim-bookworm AS runtime

# Non-root from here on; nothing in the image needs to write outside /tmp.
RUN groupadd --system --gid 1001 scoring \
 && useradd --system --uid 1001 --gid scoring --no-create-home scoring

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder --chown=scoring:scoring /app/.venv /app/.venv
COPY --chown=scoring:scoring src/ /app/src/
COPY --chown=scoring:scoring config/ /app/config/
# Built by `make export-champion`, not committed: an artifact with no provenance has no
# business in an image.
COPY --chown=scoring:scoring models/ /app/models/

USER scoring
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
