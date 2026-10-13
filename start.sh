#!/bin/sh
# Start Fly-Sim in Docker; uses the NVIDIA GPU automatically when one is available.
cd "$(dirname "$0")"
# take the latest version from GitHub first (skipped if git is missing or there are local changes)
if command -v git >/dev/null 2>&1 && [ -d .git ]; then git pull --ff-only || echo "git pull skipped"; fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
   && docker run --rm --gpus all hello-world >/dev/null 2>&1; then
  echo "NVIDIA GPU found: the brain will run on the GPU"
  exec docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build "$@"
else
  echo "No usable NVIDIA GPU: running on the CPU"
  exec docker compose up --build "$@"
fi
