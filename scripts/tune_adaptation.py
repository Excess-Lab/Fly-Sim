"""Spike-frequency adaptation strength vs. network stability and the published sugar->MN9 result,
with experimentally verified transmitters (BANC) applied to all datasets."""
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from flysim.brain import factory
from flysim.brain.lif import Brain

for ds in sys.argv[1:] or ["female", "female_banc", "male"]:
    con, ann = factory.load(ds)
    t = pd.Series(ann.type.to_numpy())
    sug = np.flatnonzero(t.str.match(r"^LB3[bc]$").to_numpy())
    mn9 = np.flatnonzero(t.str.match(r"^(MN9|CB0701)$").to_numpy())
    trn = np.flatnonzero(t.str.match(r"^TRN_VP2").to_numpy())
    for bw in (0.0, 1.0, 2.0, 4.0):
        b0, _ = factory.make_brain(ds)
        b = Brain(con, replace(b0.p, b_w=bw))
        b.apply_point_neuron_corrections(ann)
        b.reset(0)
        b.set_poisson_targets(trn)
        b.step(1000, trn, 50.0)
        c, _ = b.step(3000)
        run = int(np.count_nonzero(c))
        r = np.mean([b.run(1000.0, stim={int(i): 100.0 for i in sug}, seed=k)[0][mn9].mean() for k in range(2)])
        print(f"{ds:12s} b_w={bw:3.1f} mV  active after offset {run:6d}  sugar->MN9 {r:5.1f} Hz", flush=True)
