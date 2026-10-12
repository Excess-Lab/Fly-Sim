FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md README.ru.md LICENSE NOTICE.md ./
COPY flysim ./flysim
RUN pip install --no-cache-dir -e .
COPY scripts ./scripts
COPY ASSUMPTIONS.md KNOWN_ISSUES.md PLAN.md ./
COPY data/raw/odor ./data/raw/odor

ENV NUMBA_CACHE_DIR=/app/data/.numba-cache \
    PYTHONUNBUFFERED=1
EXPOSE 8800
# first start downloads ~0.7 GB of public connectome data into the mounted data volume
CMD ["sh", "-c", "scripts/download_data.sh && python -m flysim.live --host 0.0.0.0 --port 8800"]
