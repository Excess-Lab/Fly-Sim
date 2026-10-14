"""Calibrate the proboscis-extension threshold on behaviour, then test hunger as a prediction.

MN9 firing during PER has never been recorded (Shiu et al. 2024), so the threshold cannot come
from physiology. Behaviour: fed flies extend the proboscis on 50 % of trials at ~800 mM
sucrose; after 1 day of starvation ~300 mM, after 2 days ~200 mM (Inagaki et al. 2014).
Method: sugar GRNs fire R(C) = 1 + 60 C / (C + 35 mM) (Gustation); hunger multiplies their output
synapses by 1.0 (fed) .. 1.4 (starved; Inagaki 2012). The threshold = MN9 rate of a FED fly at
800 mM. The starved C50 is then a prediction of the connectome, not a fit.
  .venv/bin/python scripts/calibrate_per.py female_banc
"""
import sys

import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain

ds = sys.argv[1] if len(sys.argv) > 1 else "female_banc"
b, ann = make_brain(ds)
t = pd.Series(ann.type.astype(str).to_numpy())
sug = np.flatnonzero(t.str.match(r"^LB3[bc]$").to_numpy())
mn9 = np.flatnonzero(t.str.match(r"^(MN9|CB0701)$").to_numpy())
con = b.con
edges = np.concatenate([np.arange(con.indptr[i], con.indptr[i + 1]) for i in sug])
w0 = b.w[edges].copy()
conc = [0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
curves = {}
for label, f in (("fed", 1.0), ("starved", 1.4)):
    b.w[edges] = w0 * np.float32(f)
    b._gpu_w_dirty = True
    row = []
    for c in conc:
        r = 1 + 60 * c / (c + 0.035)
        row.append(float(np.mean([b.run(1000.0, stim={int(i): r for i in sug}, seed=k)[0][mn9].mean() for k in range(3)])))
    curves[label] = row
    print(label, " ".join(f"{int(c * 1000)}mM:{v:.1f}" for c, v in zip(conc, row)), flush=True)
thr = float(np.interp(0.8, conc, curves["fed"]))
print(f"threshold (fed, 800 mM) = {thr:.1f} Hz")
st = np.array(curves["starved"])
c50 = next((c for c, v in zip(conc, st) if v >= thr), None)
print(f"starved fly reaches it at ~{int(c50 * 1000) if c50 else '>1000'} mM (real 1 d: ~300, 2 d: ~200 mM)")
