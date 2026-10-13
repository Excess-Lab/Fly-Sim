FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md README.ru.md LICENSE NOTICE.md ./
COPY flysim ./flysim
RUN pip install --no-cache-dir -e .
COPY scripts ./scripts
COPY docker ./docker
COPY ASSUMPTIONS.md KNOWN_ISSUES.md PLAN.md ./
COPY data/raw/odor ./data/raw/odor

ENV NUMBA_CACHE_DIR=/app/data/.numba-cache \
    PYTHONUNBUFFERED=1
EXPOSE 8800
# scripts may arrive with Windows line endings (git autocrlf): normalise them
RUN sed -i 's/\r$//' scripts/*.sh docker/*.sh && chmod +x scripts/*.sh docker/*.sh
# the server starts immediately; first start downloads ~1.2 GB of public connectome data into
# the mounted data volume in the background (see data/download.log)
CMD ["sh", "docker/entrypoint.sh"]
