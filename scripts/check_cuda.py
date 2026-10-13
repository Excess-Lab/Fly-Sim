"""Compare the CUDA backend with the CPU backend on a small random network.

Runs on any machine: without an NVIDIA GPU set NUMBA_ENABLE_CUDASIM=1 (the CUDA simulator
executes the same kernels on the CPU, slowly). With a GPU it also reports the speed-up on a
real connectome:  .venv/bin/python scripts/check_cuda.py [--bench female_banc]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def small_brain(seed=0, n=300, k=30):
    from flysim.brain.lif import Brain, LIFParams
    from flysim.data.flywire import Connectome
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), k)
    post = rng.integers(0, n, n * k)
    syn = rng.integers(1, 6, n * k) * np.where(rng.random(n * k) < 0.25, -1, 1)
    order = np.lexsort((post, pre))
    pre, post, syn = pre[order], post[order], syn[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))]).astype(np.int64)
    con = Connectome(ids=np.arange(n, dtype=np.int64), indptr=indptr, post=post.astype(np.int32),
                     syn=syn.astype(np.int32))
    b = Brain(con, LIFParams(), seed=seed)
    kc, mbon, dan = np.arange(0, 120), np.arange(120, 135), np.arange(135, 150)
    b.enable_plasticity(kc, mbon, dan)
    pairs = [(int(a), int(a) + 1, 0.1, 1.0, 1.0) for a in range(160, 180, 2)] + [(200, 201, 0.05, 20.0, 0.0)]
    b.set_gap_junctions(pairs)
    b.v_rest[220:240] = -44.3          # tonic cells
    b.reset(seed)
    return b


def run(backend, steps=600):
    from flysim.brain import lif
    b = small_brain()
    if backend == "cuda":
        from flysim.brain.lif_cuda import CudaRunner
        b._gpu = CudaRunner(b)
    else:
        b._gpu = None
    rng = np.random.default_rng(5)
    ks = np.sort(rng.integers(0, steps, 1500))
    ki = rng.integers(0, 210, 1500)          # deterministic kicks: KCs, MBONs, DANs, others
    counts, _ = b.step(steps, kicks=(ks, ki))
    return counts, b.kc_mbon_strength(), b


def main():
    c_cpu, m_cpu, _ = run("cpu")
    c_gpu, m_gpu, _ = run("cuda")
    diff = int(np.abs(c_cpu - c_gpu).sum())
    print(f"spikes cpu {c_cpu.sum()} gpu {c_gpu.sum()}  |diff| {diff}")
    print(f"KC->MBON strength max |diff| {np.abs(m_cpu - m_gpu).max():.2e} "
          f"(plasticity exercised: min strength {m_cpu.min():.3f})")
    ok = diff <= 0.002 * c_cpu.sum() and np.abs(m_cpu - m_gpu).max() < 1e-3 and m_cpu.min() < 0.999
    # Poisson drive: different random draws, same statistics
    rates = {}
    for backend in ("cpu", "cuda"):
        b = small_brain()
        if backend == "cuda":
            from flysim.brain.lif_cuda import CudaRunner
            b._gpu = CudaRunner(b)
        idx = np.arange(0, 60)
        b.set_poisson_targets(idx)
        c, _ = b.step(2000, idx, np.full(60, 100.0))
        rates[backend] = c[idx].sum() / (60 * 0.2)
    print(f"Poisson 100 Hz -> cpu {rates['cpu']:.1f} Hz, gpu {rates['cuda']:.1f} Hz")
    ok = ok and abs(rates["cpu"] - rates["cuda"]) < 0.1 * rates["cpu"]
    if "--bench" in sys.argv:
        ds = sys.argv[sys.argv.index("--bench") + 1]
        from flysim.brain.factory import make_brain
        for backend in ("cpu", "auto"):
            os.environ["FLYSIM_BACKEND"] = backend
            b, ann = make_brain(ds)
            idx = np.flatnonzero(ann.type.astype(str).str.match("^ORN_").to_numpy())
            b.set_poisson_targets(idx)
            b.step(1000, idx, np.full(len(idx), 8.0))           # warm-up / JIT
            t = time.time()
            b.step(10000, idx, np.full(len(idx), 8.0))          # 1 s of brain time
            print(f"{ds} {backend}: {time.time() - t:.2f} s per simulated second")
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
