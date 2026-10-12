"""Male CNS: do body (VNC) sensory inputs reach punishment (PPL1) / reward (PAM) DANs?"""
import numpy as np
from flysim.brain.lif import Brain
from flysim.data import annotations as A
con, ann = A.load("male")
from flysim.brain.factory import make_brain
b, _ = make_brain("male")

ppl1, pam = A.select(ann, "^PPL1"), A.select(ann, "^PAM")
sc, cc, sub, rec = ann.super_class, ann.cell_class, ann.cell_sub_class, ann.receptor
groups = {
    "heat TRN_VP2": ann.type.str.match("^TRN_VP2"),
    "CO2 ORN_V": ann.type.str.match("^ORN_V$"),
    "leg gustatory (VNC)": (sc == "vnc_sensory") & (cc == "gustatory"),
    "VNC unknown sensory": (sc == "vnc_sensory") & (cc == "unknown_sensory"),
    "leg bristle touch": sub == "leg bristle",
    "abdomen sensory": sub == "abdomen",
    "ppk23 (pheromone)": rec == "putative_ppk23",
}
for name, m in groups.items():
    idx = np.flatnonzero(m.to_numpy())
    if len(idx) > 400:
        idx = np.random.default_rng(0).choice(idx, 400, replace=False)
    c = np.mean([b.run(1000.0, stim={int(i): 50.0 for i in idx}, seed=k)[0] for k in range(2)], 0)
    top = sorted(((round(c[i], 1), ann.type[i]) for i in ppl1), reverse=True)[:3]
    print(f"{name:22s} n={len(idx):4d} | PPL1 {c[ppl1].mean():5.1f} Hz {top} | "
          f"PAM {c[pam].mean():5.2f} Hz ({(c[pam] > 5).sum()} >5Hz) | active {np.count_nonzero(c)}",
          flush=True)
