"""Which sensory inputs reach reinforcement dopamine neurons through the connectome alone?

Stimulate a sensory population (Poisson, 1 s) and report PPL1 (punishment) and PAM (reward)
DAN firing. No pathway is hand-wired; this tests whether the wiring supports learning signals.
"""
import sys

import numpy as np

from flysim.brain.lif import Brain
from flysim.data import annotations as A

sex = sys.argv[1] if len(sys.argv) > 1 else "female"
con, ann = A.load(sex)
brain = Brain(con)
print('dropped synapses', brain.apply_point_neuron_corrections(ann))
groups = {
    "heat TRN_VP2": "^TRN_VP2",
    "cold TRN_VP3": "^TRN_VP3",
    "bitter LB1a-d": "^LB1[a-d]$",
    "sugar LB3c": "^LB3c$",
    "ORN_DM1 (food odor)": "^ORN_DM1$",
    "ORN_V (CO2)": "^ORN_V$",
}
ppl1 = A.select(ann, "^PPL1")
pam = A.select(ann, "^PAM")
mbon = A.select(ann, "^MBON")
for name, pat in groups.items():
    idx = A.select(ann, pat)
    if len(idx) == 0:
        print(f"{name}: not found")
        continue
    rates = []
    for k in range(3):
        c, _ = brain.run(1000.0, stim={int(i): 100.0 for i in idx}, seed=k)
        rates.append(c)
    c = np.mean(rates, 0)
    top = sorted(((c[i], ann.type[i]) for i in ppl1), reverse=True)[:4]
    print(f"{name:22s} n={len(idx):3d} | PPL1 mean {c[ppl1].mean():5.1f} Hz top {top} | "
          f"PAM mean {c[pam].mean():5.2f} Hz ({(c[pam] > 5).sum()} >5Hz) | "
          f"MBON active {(c[mbon] > 5).sum()} | total active {np.count_nonzero(c)}")
