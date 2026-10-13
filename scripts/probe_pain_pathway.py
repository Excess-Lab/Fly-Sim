"""Does noxious input from leg nociceptors reach punishment dopamine neurons (PPL1) through the
connectome? Only possible in full-CNS models (BANC female, Male CNS).

Controls (same rate, same number of stimulated neurons as the nociceptor set):
  baseline        no stimulus (spontaneous PPL1 rate of the model)
  proprioceptors  random leg proprioceptors (mechanosensory_proprioceptive, leg nerves)
  other_leg       random other leg-nerve sensory neurons (not nociceptor class, not proprioceptor)
A nociceptor-specific pathway should drive PPL1 clearly above all three controls.

  NUMBA_NUM_THREADS=2 python scripts/probe_pain_pathway.py [rate_Hz ...]   (default 150)
"""
import sys

import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain

RATES = [float(x) for x in sys.argv[1:]] or [150.0]
SEEDS = (0, 1)
T_MS = 1000.0


def ppl1_rate(b, ids, rate, ppl1):
    stim = {int(i): rate for i in ids}
    c = np.mean([b.run(T_MS, stim=stim, seed=k)[0] for k in SEEDS], 0)
    return c, float(c[ppl1].mean())


for ds in ("female_banc", "male"):
    b, ann = make_brain(ds)
    t = pd.Series(ann.type.astype(str).to_numpy())
    cc = ann.cell_class.astype(str).to_numpy()
    sc = ann.super_class.astype(str).to_numpy()
    leg = np.isin(ann["entry_nerve"].fillna("").to_numpy(), ["ProLN", "MesoLN", "MetaLN"])
    noci_m = ((cc == "multidendritic_neuron") if ds == "female_banc" else (cc == "unknown_sensory")) & leg
    prop_m = (cc == "mechanosensory_proprioceptive") & leg
    other_m = np.char.find(sc.astype(str), "sensory") >= 0
    other_m &= leg & ~noci_m & ~prop_m
    noci = np.flatnonzero(noci_m)
    ppl1 = np.flatnonzero(t.str.match("^PPL1").to_numpy())
    rng = np.random.default_rng(0)
    n = len(noci)
    sets = {"nociceptors": noci,
            "proprioceptors": rng.choice(np.flatnonzero(prop_m), min(n, prop_m.sum()), replace=False),
            "other_leg": rng.choice(np.flatnonzero(other_m), min(n, other_m.sum()), replace=False)}
    _, base = ppl1_rate(b, [], 0.0, ppl1)
    print(f"{ds:12s} PPL1 n={len(ppl1)}; pools: nociceptor class {n}, proprioceptors {prop_m.sum()}, "
          f"other leg sensory {other_m.sum()}", flush=True)
    print(f"{ds:12s} baseline (no stimulus) -> PPL1 mean {base:.1f} Hz", flush=True)
    for r in RATES:
        for name, ids in sets.items():
            c, m = ppl1_rate(b, ids, r, ppl1)
            top = sorted(((round(float(c[i]), 1), ann.type[i]) for i in ppl1), reverse=True)[:3]
            print(f"{ds:12s} {name:15s} {len(ids):4d} @ {r:.0f} Hz -> PPL1 mean {m:.1f} Hz "
                  f"(baseline {base:.1f}), top {top}", flush=True)
    del b
