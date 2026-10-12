"""Brain-level tracking measures (no body): sensory integration and persistence of internal states.

1. Multisensory integration: responses of key readouts to A, B and A+B.
   Additivity index = R(A+B) / (R(A) + R(B)); >1 superadditive, <1 subadditive, ~1 linear.
2. Persistence: after a 300 ms stimulus, how long does activity outlast it (half-decay time of
   population activity and of P1/arousal, MBON, DAN groups)?
Results -> results/tracking.json
"""
import json
import sys

import numpy as np
import pandas as pd

from flysim.brain.factory import make_brain
from flysim.sim.experiments import ROOT

sex = sys.argv[1] if len(sys.argv) > 1 else "female"
b, ann = make_brain(sex)
t = pd.Series(ann.type.to_numpy())
sel = lambda pat: np.flatnonzero(t.str.match(pat).to_numpy())  # noqa: E731

stims = {
    "odor_food(DM1+DM4+DM3)": sel(r"^ORN_(DM1|DM4|DM3)$"),
    "taste_sugar": sel(r"^LB3[bc]$"),
    "taste_bitter": sel(r"^LB1[a-d]$"),
    "co2(V)": sel(r"^ORN_V$"),
    "heat(VP2)": sel(r"^TRN_VP2"),
}
readouts = {
    "MN9_feeding": sel(r"^(MN9|CB0701)$"),
    "DNa02_steer": sel(r"^DNa02$"),
    "MBON_all": sel(r"^MBON"),
    "PPL1_punish": sel(r"^PPL1"),
    "PAM_reward": sel(r"^PAM"),
    "LH_output": np.flatnonzero(ann.cell_class.isin(["LHCENT", "LHLN"]).to_numpy()),
    "DN_all": np.flatnonzero(ann.super_class.str.contains("descending").to_numpy()),
}


def run(idx_list, ms=1000.0, seeds=3):
    stim = {}
    for idx in idx_list:
        stim.update({int(i): 80.0 for i in idx})
    acc = np.zeros(len(ann))
    for s in range(seeds):
        c, _ = b.run(ms, stim=stim, seed=s)
        acc += c
    acc /= seeds * ms / 1000
    return {k: float(acc[v].mean()) if len(v) else 0.0 for k, v in readouts.items()} | \
        {"active_neurons": int(np.count_nonzero(acc))}


out = {"sex": sex, "integration": [], "persistence": []}
pairs = [("odor_food(DM1+DM4+DM3)", "taste_sugar"), ("odor_food(DM1+DM4+DM3)", "co2(V)"),
         ("taste_sugar", "taste_bitter"), ("odor_food(DM1+DM4+DM3)", "heat(VP2)")]
single = {k: run([v]) for k, v in stims.items()}
for a, c in pairs:
    both = run([stims[a], stims[c]])
    row = {"A": a, "B": c, "R_A": single[a], "R_B": single[c], "R_AB": both, "additivity": {}}
    for k in readouts:
        den = single[a][k] + single[c][k]
        row["additivity"][k] = round(both[k] / den, 3) if den > 0.5 else None
    out["integration"].append(row)
    print(a, "+", c, {k: v for k, v in row["additivity"].items() if v is not None}, flush=True)

# persistence after a brief stimulus
for name, idx in stims.items():
    b.reset(0)
    b.set_poisson_targets(idx)
    trace = {k: [] for k in readouts}
    pop = []
    for k in range(40):  # 40 x 25 ms = 1 s; stimulus during the first 300 ms
        on = k < 12
        c, _ = b.step(250, idx if on else None, 80.0 if on else None)
        pop.append(int(np.count_nonzero(c)))
        for r, v in readouts.items():
            trace[r].append(float(c[v].mean() / 0.025) if len(v) else 0.0)
    pop = np.array(pop)
    peak = pop[:12].max()
    after = pop[12:]
    half = next((i * 25 for i, x in enumerate(after) if x < peak / 2), None)
    out["persistence"].append({"stimulus": name, "peak_active": int(peak),
                               "active_at_end": int(after[-1]),
                               "half_decay_ms_after_offset": half,
                               "traces_25ms": trace, "population_25ms": pop.tolist()})
    print(name, "peak", peak, "half-decay ms", half, "active at +700ms", after[-1], flush=True)

(ROOT / f"tracking_{sex}.json").write_text(json.dumps(out, indent=1))
