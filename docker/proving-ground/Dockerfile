# HermesKatana Proving Ground worker image.
#
# Public-safe variant: this image does NOT bake auth.json, .env files, corpora,
# sessions, or results. Mount runtime credentials/config explicitly.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    HERMES_HOME=/opt/hermes_data \
    PROVING_GROUND=/workspace \
    PYTHONPATH=/app/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash ca-certificates curl git jq openssh-client procps ripgrep tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY policies /app/policies
RUN pip install -e ".[proving-ground,hf]"

RUN mkdir -p "$HERMES_HOME" "$PROVING_GROUND" && chmod 700 "$HERMES_HOME"
COPY --chmod=755 docker/proving-ground/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chmod=755 docker/proving-ground/healthcheck.sh /usr/local/bin/healthcheck.sh

WORKDIR /workspace
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD /usr/local/bin/healthcheck.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
