"""The CUDA backend must reproduce the CPU backend (run in the CUDA simulator if no GPU)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_cuda_matches_cpu():
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM=os.environ.get("NUMBA_ENABLE_CUDASIM", "1"),
               FLYSIM_BACKEND="cpu", NUMBA_NUM_THREADS="2")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_cuda.py")], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
