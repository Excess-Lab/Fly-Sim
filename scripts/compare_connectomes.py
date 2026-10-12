"""Relate the female full CNS (BANC) to the validated female brain (FlyWire v783).

Same stimuli, same readouts identified by cell type in both datasets:
 1. runaway check (activity after stimulus offset)
 2. sugar GRNs -> MN9 (Shiu et al. 2024), plus bitter suppression
 3. sensory -> reinforcement DANs (PPL1/PAM), ORN -> PN, heat -> TRN targets
Output: results/connectome_comparison.json
"""
import json

import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain
from flysim.sim.experiments import ROOT

TESTS = {
    "sugar->MN9": ([r"^LB3[bc]$"], r"^(MN9|CB0701)$"),
    "sugar+bitter->MN9": ([r"^LB3[bc]$", r"^LB1[a-d]$"], r"^(MN9|CB0701)$"),
    "CO2(ORN_V)->PPL1": ([r"^ORN_V$"], r"^PPL1"),
    "DM1+DM4->DM1/DM4 PNs": ([r"^ORN_(DM1|DM4)$"], r"^(DM1|DM4)_lPN"),
    "heat(TRN_VP2)->DN": ([r"^TRN_VP2"], r"^DN"),
    "DNp09->leg MNs": ([r"^DNp09$"], r"^(Ti flexor MN|Ti extensor MN|Tr flexor MN|Tr extensor MN)$"),
}
out = {}
for ds in ("female", "female_banc", "male"):
    b, ann = make_brain(ds)
    t = pd.Series(ann.type.to_numpy())
    res = {}
    for name, (stims, read) in TESTS.items():
        idx = np.concatenate([np.flatnonzero(t.str.match(p).to_numpy()) for p in stims])
        rd = np.flatnonzero(t.str.match(read).to_numpy())
        if len(idx) == 0 or len(rd) == 0:
            res[name] = {"n_stim": int(len(idx)), "n_read": int(len(rd)), "rate_Hz": None}
            continue
        r = np.mean([b.run(1000.0, stim={int(i): 100.0 for i in idx}, seed=k)[0][rd].mean() for k in range(2)])
        res[name] = {"n_stim": int(len(idx)), "n_read": int(len(rd)), "rate_Hz": round(float(r), 2)}
    # runaway
    idx = np.flatnonzero(t.str.match(r"^TRN_VP2").to_numpy())
    b.reset(0)
    b.set_poisson_targets(idx)
    b.step(1000, idx, 50.0)
    c, _ = b.step(3000)
    res["active_after_offset"] = int(np.count_nonzero(c))
    out[ds] = res
    print(ds, json.dumps(res, ensure_ascii=False), flush=True)
(ROOT / "connectome_comparison.json").write_text(json.dumps(out, indent=1))
