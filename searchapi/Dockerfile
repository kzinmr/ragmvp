FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY src /app/src
COPY README.md /app
COPY pyproject.toml /app
WORKDIR /app

# Sync the project into a new environment
RUN uv sync

COPY .env /app/.env

CMD ["uv", "run", "/app/src/searchapi/main.py", "-i", "/app/data"]
