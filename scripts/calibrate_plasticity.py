"""Calibrate KC->MBON depression on Hige et al. 2015 (Neuron): one 1 s odour paired with
4 x PPL1-gamma1pedc activations at 2 Hz (from 0.2 s after odour onset) depresses the
MBON-gamma1pedc>a/b response to that odour by ~90 % (EPSC; spikes -80 %); a different odour
by ~20-27 %.  .venv/bin/python scripts/calibrate_plasticity.py female_banc 2 5 10 20
"""
import sys

import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain

ds = sys.argv[1]
etas = [float(x) for x in sys.argv[2:]] or [2.0]
b, ann = make_brain(ds)
t = pd.Series(ann.type.astype(str).to_numpy())
kc = np.flatnonzero(t.str.match("^KC").to_numpy())
mb = np.flatnonzero(t.str.match("^MBON").to_numpy())
ppl1 = np.flatnonzero(t.str.match("^PPL1").to_numpy())
pam = np.flatnonzero(t.str.match("^PAM").to_numpy())
g1 = np.flatnonzero(t.str.match(r"^PPL101$|^PPL1-?g1pedc").to_numpy())      # PPL1-gamma1pedc
m11 = np.flatnonzero(t.str.match(r"^MBON11$|^MBON-?g1pedc").to_numpy())     # MBON-gamma1pedc>a/b
orn = np.flatnonzero(t.str.match("^ORN_").to_numpy())
A = np.flatnonzero(t.str.match("^ORN_(DM1|DM4|VA2)$").to_numpy())
B = np.flatnonzero(t.str.match("^ORN_(DL5|DM2|VM2)$").to_numpy())
print(ds, "PPL1-g1pedc", len(g1), "MBON-g1pedc", len(m11))
w_naive = b.w.copy()
for eta in etas:
    b.w[:] = w_naive                     # every eta starts from the naive weights
    b._gpu_w_dirty = True
    b.enable_plasticity(kc, mb, np.concatenate([ppl1, pam]), eta_dep=eta)
    idx = np.concatenate([orn, g1]); b.reset(1); b.set_poisson_targets(idx)

    def run(sec, od=None, kicks=()):
        r = np.full(len(idx), 8.0)
        if od is not None:
            r[np.isin(idx, od)] = 70.0
        r[np.isin(idx, g1)] = 0.0
        n = int(round(sec * 1000 / b.p.dt))
        ks = [(int(round(k * 1000 / b.p.dt)), int(i)) for k in kicks for i in g1]
        ks.sort()
        kk = (np.array([k for k, _ in ks], np.int64), np.array([i for _, i in ks], np.int64)) if ks else None
        return b.step(n, idx, r, kicks=kk)[0]

    run(5)
    kcA = kc[run(1, A)[kc] > 0]; kcB = kc[run(1, B)[kc] > 0]
    kcA, kcB = np.setdiff1d(kcA, kcB), np.setdiff1d(kcB, kcA)
    run(5)
    slot = {int(m): k for k, m in enumerate(b.mbon_idx)}

    def w(kcs):
        m = np.isin(b.pe_pre, kcs) & np.isin(b.pe_m, [slot[int(x)] for x in m11 if int(x) in slot])
        return float((b.w[b.pe_edge[m]] / b.pe_w0[m]).mean()) if m.any() else float("nan")
    a0, b0 = w(kcA), w(kcB)
    # Hige protocol: odour A 1 s, PPL1-g1pedc pulses at 0.2, 0.7, 1.2, 1.7 s (2 Hz)
    run(2.0, A, kicks=(0.2, 0.7, 1.2, 1.7))
    run(3)
    print(f"eta_dep {eta}: odour A synapses {a0:.2f} -> {w(kcA):.2f} (real ~0.1), "
          f"odour B {b0:.2f} -> {w(kcB):.2f} (real ~0.75)", flush=True)
