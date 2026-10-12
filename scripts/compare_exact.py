"""Deterministic spike-by-spike comparison flysim vs Brian2 (same kick schedule, no RNG)."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from validate_sugar import SUGAR_GRN  # noqa: E402

from flysim.brain.lif import Brain  # noqa: E402
from flysim.data import flywire  # noqa: E402

T_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
DT = 0.1
con = flywire.build()
sugar = con.index_of(con.filter_ids(SUGAR_GRN))
n_steps = int(T_MS / DT)

# 100 Hz kick trains on sugar GRNs, fixed schedule
rng = np.random.default_rng(1)
kick = rng.random((n_steps, len(sugar))) < 100 * DT * 1e-3
ks, kj = np.nonzero(kick)
ki = sugar[kj]

b = Brain(con)
b.set_poisson_targets(sugar)
counts, _ = b.step(n_steps, kicks=(ks, ki), record=np.arange(con.n))
# re-run with full raster is too big (n_steps x 138k); use record of active neurons
active = np.flatnonzero(counts)
b.reset()
b.set_poisson_targets(sugar)
_, rast = b.step(n_steps, kicks=(ks, ki), record=active)
st, sc = np.nonzero(rast)
mine = sorted(zip(st.tolist(), active[sc].tolist()))

from brian2 import (Network, NeuronGroup, SpikeGeneratorGroup, SpikeMonitor, Synapses,  # noqa
                    defaultclock, mV, ms, prefs)
prefs.codegen.target = "numpy"
defaultclock.dt = DT * ms
p = dict(v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms)
neu = NeuronGroup(con.n, """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                            dg/dt = -g / tau : volt (unless refractory)
                            rfc : second""",
                  method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                  refractory="rfc", namespace=p)
neu.v = p["v_0"]
neu.rfc = 2.2 * ms
neu.rfc[sugar] = 0 * ms
syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms)
syn.connect(i=np.repeat(np.arange(con.n), np.diff(con.indptr)), j=con.post)
syn.w = con.syn * 0.275 * mV
gen = SpikeGeneratorGroup(len(sugar), kj, ks * DT * ms)
gs = Synapses(gen, neu, on_pre="v_post += 68.75*mV")
gs.connect(i=np.arange(len(sugar)), j=sugar)
mon = SpikeMonitor(neu)
Network(neu, syn, gen, gs, mon).run(T_MS * ms)
ref = sorted(zip(np.round(np.asarray(mon.t / ms) / DT).astype(int).tolist(),
                 np.asarray(mon.i).tolist()))

print(f"spikes mine {len(mine)} brian2 {len(ref)}")
for a, r in zip(mine, ref):
    if a != r:
        print("first divergence: mine", a, "brian2", r)
        break
else:
    print("identical over common prefix", min(len(mine), len(ref)))
sm, sr = set(mine), set(ref)
print("only mine:", sorted(sm - sr)[:10])
print("only brian2:", sorted(sr - sm)[:10])
