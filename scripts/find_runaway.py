"""Diagnose self-sustaining activity: brief stimulus, then check what keeps firing."""
import sys
import numpy as np
from flysim.brain.lif import Brain
from flysim.data import annotations as A
con, ann = A.load(sys.argv[1] if len(sys.argv) > 1 else "female")
from flysim.brain.factory import make_brain
b, ann = make_brain(sys.argv[1] if len(sys.argv) > 1 else "female")

idx = A.select(ann, "^TRN_VP2")
b.reset(0); b.set_poisson_targets(idx)
b.step(1000, idx, 50.0)
c, _ = b.step(3000)  # 300 ms after stimulus off
print("active after stim off:", np.count_nonzero(c))
a = ann.assign(r=c / 0.3)
act = a[a.r > 0]
print(act.groupby("cell_class").size().sort_values(ascending=False).head(12))
g = act.groupby(["super_class", "type"]).r.agg(["size", "mean"]); print(g.sort_values("size", ascending=False).head(30)); print(act.groupby("super_class").size()); print(act.groupby("nt").size())
