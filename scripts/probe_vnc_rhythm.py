"""Does the connectome VNC (Male CNS) produce a stepping rhythm in leg motor neurons when walking
descending neurons are driven? Decides how deep the brain->VNC->MN->muscle chain can go.

Drive: DNp09 (P9, forward walking; Bidaye et al. 2020) bilaterally, optionally plus other
walking-related DNs. Record leg MN populations per (leg, muscle) at 1 ms; report mean rate,
dominant frequency (stepping is ~5-15 Hz) and flexor/extensor antiphase.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flysim.brain.factory import make_brain
from flysim.data.malecns import ANN

b, ann = make_brain("male")
a = feather.read_table(ANN, columns=["bodyId", "type", "subclass"]).to_pandas() \
    .set_index("bodyId").reindex(b.con.ids)
t = pd.Series(ann.type.to_numpy())
side = ann.side.to_numpy()
mn = ann.super_class.to_numpy() == "vnc_motor"
legs = {"fl": "front", "ml": "mid", "hl": "hind"}
groups = {}
for sub, leg in legs.items():
    for s in ("left", "right"):
        for musc in ("Ti flexor MN", "Ti extensor MN", "Tr flexor MN", "Tr extensor MN",
                     "Sternal anterior rotator MN", "Sternal posterior rotator MN"):
            m = mn & (a.subclass.to_numpy() == sub) & (side == s) & (t == musc).to_numpy()
            if m.any():
                groups[f"{leg}-{s[0].upper()}-{musc.replace(' MN', '')}"] = np.flatnonzero(m)

drives = {"DNp09": r"^DNp09$", "DNp09+DNa02+oDN": r"^(DNp09|DNa02|DNb02|DNg100|DNg97)$"}
which = sys.argv[1] if len(sys.argv) > 1 else "DNp09"
idx = np.flatnonzero(t.str.match(drives[which]).to_numpy())
print(f"drive {which}: {len(idx)} DNs; MN groups {len(groups)}")
rec = np.concatenate(list(groups.values()))
b.reset(0)
b.set_poisson_targets(idx)
_, rast = b.step(20000, idx, 150.0, record=rec)  # 2 s
ms = rast.reshape(2000, 10, -1).sum(1)          # 1 ms bins
off = 0
rows = []
for k, g in groups.items():
    x = ms[:, off:off + len(g)].sum(1).astype(float)
    off += len(g)
    rate = x.sum() / len(g) / 2.0
    f = np.fft.rfftfreq(len(x) - 500, 1e-3)
    p = np.abs(np.fft.rfft(x[500:] - x[500:].mean())) ** 2
    band = (f > 2) & (f < 30)
    peak = f[band][np.argmax(p[band])] if p[band].sum() > 0 else 0.0
    power_frac = p[band].max() / p[(f > 0.5)].sum() if p[(f > 0.5)].sum() > 0 else 0.0
    rows.append((k, round(rate, 1), round(peak, 1), round(power_frac, 3)))
df = pd.DataFrame(rows, columns=["group", "rate_Hz", "peak_freq_Hz", "peak_power_frac"])
print(df.to_string(index=False))
