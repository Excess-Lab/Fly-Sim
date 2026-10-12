"""Does noxious input from leg nociceptors reach punishment dopamine neurons (PPL1) through the
connectome? Only possible in full-CNS models (BANC female, Male CNS)."""
import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain

for ds in ("female_banc", "male"):
    b, ann = make_brain(ds)
    t = pd.Series(ann.type.to_numpy())
    cc = ann.cell_class.to_numpy()
    leg = np.isin(ann["entry_nerve"].fillna("").to_numpy(), ["ProLN", "MesoLN", "MetaLN"])
    noci = np.flatnonzero(((cc == "multidendritic_neuron") if ds == "female_banc" else (cc == "unknown_sensory")) & leg)
    ppl1 = np.flatnonzero(t.str.match("^PPL1").to_numpy())
    for r in (50, 100, 150):
        c = np.mean([b.run(1000.0, stim={int(i): r for i in noci}, seed=k)[0] for k in range(2)], 0)
        top = sorted(((c[i], ann.type[i]) for i in ppl1), reverse=True)[:3]
        print(f"{ds:12s} {len(noci)} leg nociceptors @ {r} Hz -> PPL1 mean {c[ppl1].mean():.1f} Hz, top {top}")
