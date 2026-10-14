"""Time course of the walking DN DNg100 (BDN2) and its SEZ loop under hunger x food odour.

Closed loop, A30 off (spontaneous=False), fly held 15 mm from a 2-day fermenting apple.
Per seed and state (sated 0 h / starved 24 h): 0-5 s no odour, 5-10 s apple odour, 10-15 s no
odour again (does the activity outlast the stimulus?). 100 ms bins.

Usage: python scripts/probe_walk_state.py female_banc|male [seeds]
"""
import json
import sys
from pathlib import Path

import numpy as np

from flysim.agent.fly import Fly, FlyConfig
from flysim.agent.metabolism import Metabolism
from flysim.world.apple import Apple, AppleParams
from flysim.world.world import World

DS = sys.argv[1] if len(sys.argv) > 1 else "female_banc"
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3]
OUT = Path(__file__).resolve().parents[1] / "results" / "walking"
OUT.mkdir(parents=True, exist_ok=True)
TRACK = ["DNg100", "DNg16", "CB0202", "DNge046", "CB0244", "DNp09", "DNg97", "DNa02"]

apple = Apple(AppleParams(radius_mm=35.0), center=(0.0, 0.0))
apple.age(48)
w = World(apple, arena=60.0)
f = Fly(FlyConfig("F", "male" if DS == "male" else "female", starved_h=0, seed=1,
                  spontaneous=False, plasticity=False, connectome=DS), pos=(50.0, 0.0, 0.0), yaw=np.pi)
w.flies.append(f)
w.rebuild_physics()
t = f.ann.type.astype(str).to_numpy()
ids = {k: np.flatnonzero(t == k) for k in TRACK}
out = {}
for seed in SEEDS:
    for hunger in (0, 24):
        f.brain.reset(seed)
        f.motor.__init__(f.ann)
        f.meta = Metabolism.starved(hunger)
        f.body.teleport((50.0, 0.0, 0.0), np.pi)
        trace = {k: [] for k in TRACK}
        acc = np.zeros(f.brain.con.n, np.int64)
        for k in range(1500):
            w.odor_scale = 1.0 if 500 <= k < 1000 else 0.0
            w.step(0.01)
            acc += f.pop_counts()
            if k % 10 == 9:
                for n, ix in ids.items():
                    trace[n].append(float(acc[ix].mean() / 0.1) if len(ix) else 0.0)
                acc[:] = 0
        key = f"s{seed}_h{hunger}"
        out[key] = trace
        x = np.array(trace["DNg100"])
        print(key, "DNg100 mean Hz  pre %.1f  odour %.1f  post %.1f" % (x[:50].mean(), x[50:100].mean(),
                                                                        x[100:].mean()),
              "| DNg16 %.1f %.1f %.1f" % tuple(np.array(trace["DNg16"]).reshape(3, 50).mean(1)), flush=True)
json.dump(out, open(OUT / f"probe_walk_state_{DS}.json", "w"))
