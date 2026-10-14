"""Does tonic drive of DNg100 (BDN2) evoke a stepping rhythm or graded leg-MN output in the
connectome VNC? (Tuthill/Brunton 2025-26 preprint: a 3-interneuron VNC loop downstream of
DNg100 oscillates in a connectome RNN model.) Here: our LIF model, both datasets.

Drive DNg100 (both sides) with Poisson input at 0, 50, 100, 200 Hz for 2 s each (fresh state);
record leg MNs per (leg, side, muscle type) at 1 ms bins. Report DNg100 rate, mean leg-MN
rate, and for the tibia flexor/extensor groups the dominant 2-30 Hz frequency and its share
of power (stepping is ~5-15 Hz).

Usage: python scripts/probe_dng100_rhythm.py female_banc|male
"""
import sys

import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain

DS = sys.argv[1] if len(sys.argv) > 1 else "female_banc"
b, ann = make_brain(DS)
t = ann.type.astype(str).to_numpy()
side = ann.side.astype(str).to_numpy()
leg = ann["leg"].fillna("").astype(str).to_numpy()
mn = (ann.super_class.astype(str).to_numpy() == "vnc_motor") & (leg != "")
groups = {}
for lg in ("fl", "ml", "hl"):
    for s in ("left", "right"):
        for musc in ("Ti flexor MN", "Ti extensor MN", "Tr flexor MN", "Tr extensor MN"):
            m = mn & (leg == lg) & (side == s) & (t == musc)
            if m.any():
                groups[f"{lg}-{s[0].upper()}-{musc.replace(' MN', '')}"] = np.flatnonzero(m)
legmn = np.flatnonzero(mn)
drv = np.flatnonzero(t == "DNg100")
print(DS, "DNg100:", len(drv), "leg MNs:", len(legmn), "groups:", len(groups))
rec = np.concatenate([drv] + list(groups.values()))
for hz in (0, 50, 100, 200):
    b.reset(0)
    b.set_poisson_targets(drv)
    counts, rast = b.step(20000, drv, float(hz), record=rec)
    ms = rast.reshape(2000, 10, -1).sum(1).astype(float)
    r_drv = ms[:, :len(drv)].sum() / len(drv) / 2.0
    off = len(drv)
    rows = []
    for k, g in groups.items():
        x = ms[:, off:off + len(g)].sum(1)
        off += len(g)
        f = np.fft.rfftfreq(len(x) - 500, 1e-3)
        p = np.abs(np.fft.rfft(x[500:] - x[500:].mean())) ** 2
        band = (f > 2) & (f < 30)
        tot = p[f > 0.5].sum()
        rows.append((k, x.sum() / len(g) / 2.0, f[band][np.argmax(p[band])] if tot > 0 else 0.0,
                     p[band].max() / tot if tot > 0 else 0.0))
    df = pd.DataFrame(rows, columns=["group", "rate", "peak_hz", "peak_frac"])
    print(f"drive {hz:3d} Hz: DNg100 {r_drv:5.1f} Hz | all leg MNs {counts[legmn].mean() / 2:5.2f} Hz | "
          f"groups active(>1Hz) {int((df.rate > 1).sum())}/{len(df)} | max peak_frac {df.peak_frac.max():.3f}")
    if hz == 200:
        print(df.round(2).to_string(index=False))
