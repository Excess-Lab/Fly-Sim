"""Usage: python scripts/run_experiment.py <name> [key=value ...]
names: place_avoidance, tmaze_choice, food_search_after_aversion, conflict, pair
"""
import json
import sys
import time

from flysim.sim import experiments as E

name = sys.argv[1]
kw = {}
for a in sys.argv[2:]:
    k, v = a.split("=", 1)
    try:
        kw[k] = json.loads(v)
    except json.JSONDecodeError:
        kw[k] = v
if "retention_h" in kw:
    kw["retention_h"] = tuple(kw["retention_h"])
t0 = time.time()
res = getattr(E, name)(**kw)
res["wall_s"] = round(time.time() - t0)
print("RESULT", json.dumps(res, default=float))
with open(E.ROOT / "results.jsonl", "a") as fh:
    fh.write(json.dumps(res, default=float) + "\n")
