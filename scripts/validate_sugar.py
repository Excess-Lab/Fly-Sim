"""Stage 1 validation: sugar GRN activation -> MN9 (proboscis extension motor neuron).

Reference: Shiu et al. 2024, Nature, Fig. 1-2. Sugar GRNs at 100-200 Hz Poisson drive MN9;
bitter GRNs alone do not.
Usage: python scripts/validate_sugar.py [--brian2] [--trials N]
"""
import argparse
import time

import numpy as np

from flysim.brain.lif import Brain
from flysim.data import flywire

# Shiu et al. 2024 example.ipynb (v630 ids; those absent in v783 are dropped and reported)
SUGAR_GRN = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = [720575940660219265, 720575940618238523]


def run_numba(con, sugar, mn9, rate, trials, t_ms=1000.0):
    brain = Brain(con)
    rates = []
    for k in range(trials):
        brain.reset()
        t0 = time.time()
        counts, _ = brain.run(t_ms, stim={int(i): rate for i in sugar}, seed=k)
        rates.append(counts[mn9] / (t_ms / 1000))
        print(f"  trial {k}: MN9 {rates[-1]} Hz, active neurons {np.count_nonzero(counts)}, "
              f"{time.time() - t0:.1f}s wall")
    return np.array(rates)


def run_brian2(con, sugar, mn9, rate, trials, t_ms=1000.0):
    from brian2 import (Hz, NeuronGroup, PoissonInput, SpikeMonitor, Synapses, mV, ms,
                        Network, prefs, seed as bseed)
    prefs.codegen.target = "numpy"
    p = dict(v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms)
    pre = np.repeat(np.arange(con.n), np.diff(con.indptr))
    rates = []
    for k in range(trials):
        bseed(k)
        neu = NeuronGroup(con.n, """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                                    dg/dt = -g / tau : volt (unless refractory)
                                    rfc : second""",
                          method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                          refractory="rfc", namespace=p)
        neu.v = p["v_0"]
        neu.rfc = 2.2 * ms
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms)
        syn.connect(i=pre, j=con.post)
        syn.w = con.syn * 0.275 * mV
        pois = []
        for i in sugar:
            pois.append(PoissonInput(neu[int(i)], "v", N=1, rate=rate * Hz, weight=0.275 * 250 * mV))
            neu[int(i)].rfc = 0 * ms
        mon = SpikeMonitor(neu)
        net = Network(neu, syn, mon, *pois)
        t0 = time.time()
        net.run(t_ms * ms)
        c = np.bincount(np.asarray(mon.i), minlength=con.n)
        rates.append(c[mn9] / (t_ms / 1000))
        print(f"  brian2 trial {k}: MN9 {rates[-1]} Hz, active {np.count_nonzero(c)}, "
              f"{time.time() - t0:.1f}s wall")
    return np.array(rates)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--brian2", action="store_true")
    a = ap.parse_args()

    con = flywire.build()
    present = con.filter_ids(SUGAR_GRN)
    print(f"neurons {con.n}, edges {len(con.post)}; sugar GRNs present {len(present)}/{len(SUGAR_GRN)}")
    sugar = con.index_of(present)
    mn9 = con.index_of(MN9)

    print(f"[numba] sugar GRNs @ {a.rate} Hz")
    r = run_numba(con, sugar, mn9, a.rate, a.trials)
    print(f"[numba] MN9 mean rate {r.mean(0)} Hz (sd {r.std(0)})")
    print("[numba] control: no stimulus")
    r0 = run_numba(con, [], mn9, a.rate, 1)
    if a.brian2:
        print("[brian2] reference")
        rb = run_brian2(con, sugar, mn9, a.rate, a.trials)
        print(f"[brian2] MN9 mean rate {rb.mean(0)} Hz (sd {rb.std(0)})")
