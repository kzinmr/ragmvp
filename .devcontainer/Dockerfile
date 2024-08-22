FROM mcr.microsoft.com/devcontainers/base:debian
ARG USERNAME=vscode

# hadolint ignore=DL3008
RUN \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends git

COPY --from=ghcr.io/astral-sh/uv:0.3.0 /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.3.0 /uvx /bin/uvx
ENV PYTHONUNBUFFERED=True

WORKDIR /workspace
RUN chown -R $USERNAME /workspace && git config --system --add safe.directory /workspace
USER $USERNAME

# Create uv venv (package install & venv creation)
RUN \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=src,target=src \
    uv sync
ENV PATH="/workspace/.venv/bin:$PATH"
