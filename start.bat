@echo off
rem Start Fly-Sim in Docker; uses the NVIDIA GPU automatically when one is available.
cd /d "%~dp0"
rem take the latest version from GitHub first
where git >nul 2>nul && if exist .git (git pull --ff-only || echo git pull skipped)
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    docker run --rm --gpus all hello-world >nul 2>nul
    if not errorlevel 1 (
      echo NVIDIA GPU found: the brain will run on the GPU
      docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
      goto :eof
    )
  )
)
echo No usable NVIDIA GPU: running on the CPU
docker compose up --build
