FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/KookiesNKareem/cuxray"
LABEL org.opencontainers.image.description="GPU-free static analysis for CUDA kernel binaries"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cuxray
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

ENV CUXRAY_CACHE=/cuxray-cache
WORKDIR /work
ENTRYPOINT ["cuxray"]
