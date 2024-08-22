FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# hadolint ignore=DL3008
RUN \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends apache2

COPY src /app/src
COPY README.md /app
COPY pyproject.toml /app
WORKDIR /app

# Sync the project into a new environment
RUN uv sync

CMD ["uv", "run", "/app/src/fileloader/main.py", "-i", "/app/data", "-o", "/app/output"]
