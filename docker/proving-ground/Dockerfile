# HermesKatana Proving Ground worker image.
#
# Public-safe variant: this image does NOT bake auth.json, .env files, corpora,
# sessions, or results. Mount runtime credentials/config explicitly.

FROM python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9 AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    HERMES_HOME=/opt/hermes_data \
    PROVING_GROUND=/workspace \
    PYTHONPATH=/app/src

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash ca-certificates curl git jq openssh-client procps ripgrep tini && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade \
        pip==26.1.1 \
        setuptools==82.0.1 \
        wheel==0.47.0

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY policies /app/policies
RUN python -m pip install --no-cache-dir -e ".[proving-ground,hf]"

RUN groupadd --system hermes && \
    useradd --system --gid hermes --home-dir "$HERMES_HOME" --shell /usr/sbin/nologin hermes && \
    mkdir -p "$HERMES_HOME" "$PROVING_GROUND" && \
    chmod 700 "$HERMES_HOME" && \
    chown -R hermes:hermes "$HERMES_HOME" "$PROVING_GROUND" /app
COPY --chmod=755 docker/proving-ground/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chmod=755 docker/proving-ground/healthcheck.sh /usr/local/bin/healthcheck.sh

USER hermes
WORKDIR /workspace
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD /usr/local/bin/healthcheck.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
