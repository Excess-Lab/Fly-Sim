"""Compare per-neuron firing rates: flysim Numba simulator vs Brian2 reference, same stimulus."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from validate_sugar import MN9, SUGAR_GRN  # noqa: E402

from flysim.brain.lif import Brain  # noqa: E402
from flysim.data import flywire  # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RATE = 100.0
T = 1000.0


def brian2_counts(con, sugar, seed):
    from brian2 import (Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses, mV, ms,
                        prefs)
    from brian2 import seed as bseed
    prefs.codegen.target = "numpy"
    bseed(seed)
    p = dict(v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms)
    neu = NeuronGroup(con.n, """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                                dg/dt = -g / tau : volt (unless refractory)
                                rfc : second""",
                      method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                      refractory="rfc", namespace=p)
    neu.v = p["v_0"]
    neu.rfc = 2.2 * ms
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms)
    syn.connect(i=np.repeat(np.arange(con.n), np.diff(con.indptr)), j=con.post)
    syn.w = con.syn * 0.275 * mV
    pois = []
    for i in sugar:
        pois.append(PoissonInput(neu[int(i)], "v", N=1, rate=RATE * Hz, weight=0.275 * 250 * mV))
        neu[int(i)].rfc = 0 * ms
    mon = SpikeMonitor(neu)
    Network(neu, syn, mon, *pois).run(T * ms)
    return np.bincount(np.asarray(mon.i), minlength=con.n)


con = flywire.build()
sugar = con.index_of(con.filter_ids(SUGAR_GRN))
mn9 = con.index_of(MN9)
brain = Brain(con)
cn = np.zeros(con.n)
cb = np.zeros(con.n)
for k in range(TRIALS):
    brain.reset()
    cn += brain.run(T, stim={int(i): RATE for i in sugar}, seed=100 + k)[0]
    cb += brian2_counts(con, sugar, 100 + k)
    print(f"trial {k} done", flush=True)
cn /= TRIALS
cb /= TRIALS

# hop distance from sugar GRNs
hop = np.full(con.n, -1)
hop[sugar] = 0
frontier = sugar
for h in range(1, 6):
    nxt = np.unique(np.concatenate([con.post[con.indptr[i]:con.indptr[i + 1]] for i in frontier]))
    nxt = nxt[hop[nxt] < 0]
    hop[nxt] = h
    frontier = nxt

print(f"MN9 numba {cn[mn9]} brian2 {cb[mn9]}")
print(f"sugar GRN mean rate numba {cn[sugar].mean():.1f} brian2 {cb[sugar].mean():.1f}")
act = (cn > 0) | (cb > 0)
print(f"active numba {np.count_nonzero(cn)} brian2 {np.count_nonzero(cb)} union {act.sum()}")
print(f"corr over active neurons {np.corrcoef(cn[act], cb[act])[0, 1]:.4f}")
for h in range(0, 6):
    m = act & (hop == h)
    if m.any():
        print(f"hop {h}: n={m.sum():4d}  numba {cn[m].sum():9.1f}  brian2 {cb[m].sum():9.1f}  "
              f"ratio {cn[m].sum() / max(cb[m].sum(), 1e-9):.3f}")
