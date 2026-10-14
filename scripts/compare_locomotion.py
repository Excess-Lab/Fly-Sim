"""Closed-loop comparison: locomotion="a30" vs "brain" (A101), NeuroMechFly body.

One fly near a 2-day fermenting apple (radius 35 mm), starting 15 mm from its surface at a
seed-drawn position with random heading. Records thorax xy at 50 ms and reports
path length, fraction of time walking (speed > 2 mm/s), mean speed while walking,
number / median duration of walking bouts (>= 0.3 s), latency to reach the apple,
and the forward-drive / DNg100 rate during the run.
Reference (real flies, flat arena): walking bouts of ~1-10 s separated by stops; walking speed
~10-20 mm/s (Katsov & Clandinin 2008; Berman et al. 2014; Mendes et al. 2013).

Usage: python scripts/compare_locomotion.py female_banc a30|brain starved_h seed seconds
"""
import json
import sys
from pathlib import Path

import numpy as np

from flysim.agent.fly import Fly, FlyConfig
from flysim.world.apple import Apple, AppleParams
from flysim.world.world import World

DS, LOCO, H, SEED, T = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
OUT = Path(__file__).resolve().parents[1] / "results" / "walking"
OUT.mkdir(parents=True, exist_ok=True)

g = np.random.default_rng([SEED, 1])
ang, yaw0 = g.uniform(0, 2 * np.pi, 2)
start = (50.0 * np.cos(ang), 50.0 * np.sin(ang), 0.0)
apple = Apple(AppleParams(radius_mm=35.0), center=(0.0, 0.0))
apple.age(48)
w = World(apple, arena=60.0)
f = Fly(FlyConfig("F", "male" if DS == "male" else "female", starved_h=H, seed=SEED,
                  connectome=DS, locomotion=LOCO), pos=start, yaw=float(yaw0))
w.flies.append(f)
w.rebuild_physics()
t = f.ann.type.astype(str).to_numpy()
dng100 = np.flatnonzero(t == "DNg100")
xy, fwd, rate100, surf = [], [], [], []
acc = np.zeros(len(dng100))
n = int(T / 0.01)
for k in range(n):
    w.step(0.01)
    c = f.pop_counts()
    acc += c[dng100]
    if k % 5 == 4:
        xy.append(f.body.pos[:2].copy())
        surf.append(f.body.surface)
        fwd.append(f.cmd["forward_pop"] if LOCO == "brain" else
                   min(1.2, f.cmd["forward"] - f.cmd["backward"] + 0.35))
        rate100.append(float(acc.mean() / 0.05))
        acc[:] = 0
xy = np.array(xy)
# speed from 0.5 s displacement (removes the within-stride sway of the thorax that inflates a
# 50 ms path); walking = speed > 2 mm/s
W = 10
v = np.r_[np.zeros(W), np.linalg.norm(xy[W:] - xy[:-W], axis=1) / (W * 0.05)]
walk = v > 2.0
bouts, cur = [], 0
for wk in walk:
    if wk:
        cur += 1
    elif cur:
        bouts.append(cur * 0.05)
        cur = 0
if cur:
    bouts.append(cur * 0.05)
bouts = [b for b in bouts if b >= 0.3]
on = [i for i, s in enumerate(surf) if s == "apple"]
dist = np.linalg.norm(xy, axis=1) - 35.0
m = {"dataset": DS, "locomotion": LOCO, "starved_h": H, "seed": SEED, "T": T,
     "path_mm": float(np.linalg.norm(np.diff(xy[::W], axis=0), axis=1).sum()), "net_mm": float(np.linalg.norm(xy[-1] - xy[0])),
     "frac_walking": float(walk.mean()),
     "speed_walking_mm_s": float(v[walk].mean()) if walk.any() else 0.0,
     "n_bouts": len(bouts), "median_bout_s": float(np.median(bouts)) if bouts else 0.0,
     "latency_to_apple_s": on[0] * 0.05 if on else None,
     "frac_on_apple": len(on) / len(surf), "dist_start_mm": float(dist[0]), "dist_end_mm": float(dist[-1]),
     "min_dist_mm": float(dist.min()),
     "fwd_drive_mean": float(np.mean(fwd)), "DNg100_mean_hz": float(np.mean(rate100)),
     "DNg100_frac_on": float(np.mean(np.array(rate100) > 2.0)), "events": f.events[:20]}
print(json.dumps(m))
json.dump({"metrics": m, "xy": xy.tolist(), "fwd": fwd, "dng100": rate100},
          open(OUT / f"loco_{DS}_{LOCO}_h{int(H)}_s{SEED}.json", "w"))
