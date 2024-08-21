FROM mcr.microsoft.com/devcontainers/base:debian

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.3.0 /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.3.0 /uvx /bin/uvx

COPY . /opt
WORKDIR /opt

RUN uv sync

ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=True
RUN chown -R vscode $VIRTUAL_ENV && \
    git config --system --add safe.directory /workspaces/ragmvp
