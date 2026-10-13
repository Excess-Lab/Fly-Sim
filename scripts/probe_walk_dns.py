"""Do walking descending neurons fire spontaneously, and do walking-evoking stimuli recruit them?

Closed loop with the full Fly (senses -> connectome -> NeuroMechFly body), but WITHOUT the A30
spontaneous drive (spontaneous=False): the body moves only if the brain's own DN readout says so.
Resting input = what Fly.sense() gives (ORNs ~8 Hz spontaneous, photoreceptors + tonic lamina
cells, leg proprioceptors from the physics, IPCs, thermo- and hygro-neutral air).

Conditions (each from a fresh brain state, 0.5 s settle + T s measure):
  rest_fed        sated (0 h), no odour, static visual panorama (ring of dark posts)
  rest_hungry     24 h starved, no odour
  hungry_food     24 h starved, fermenting apple odour (2-day apple, fly 15 mm from surface)
  fed_food        sated, same odour
  optomotor       sated, panorama rotating around the fly at 90 deg/s (to its left)
  touch_hindleg   sated, bristle mechanosensory neurons of the left hind leg at 100 Hz
  shock_legs      sated, leg nociceptors 150 Hz (floor shock, A24b)

Usage: python scripts/probe_walk_dns.py female_banc|male [T_seconds]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flysim.agent.fly import Fly, FlyConfig
from flysim.agent.metabolism import Metabolism
from flysim.world.apple import Apple, AppleParams
from flysim.world.world import World

DS = sys.argv[1] if len(sys.argv) > 1 else "female_banc"
T = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
OUT = Path(__file__).resolve().parents[1] / "results" / "walking"
OUT.mkdir(parents=True, exist_ok=True)

# literature walking / steering DNs (see RESEARCH_WALKING.md for the mapping)
CANDIDATES = ["DNp09", "DNg100", "DNg97", "DNa02", "DNa01", "DNa03", "DNb05", "DNg13", "DNb02",
              "MDN", "DNp42"]

apple = Apple(AppleParams(radius_mm=35.0), center=(0.0, 0.0))
apple.age(48)
w = World(apple, arena=60.0)
sex = "male" if DS == "male" else "female"
f = Fly(FlyConfig("F", sex, starved_h=0, seed=1, spontaneous=False, plasticity=False,
                  connectome=DS), pos=(50.0, 0.0, 0.0), yaw=np.pi)
w.flies.append(f)
w.rebuild_physics()
ann = f.ann
t = ann.type.astype(str).to_numpy()
side = ann.side.astype(str).to_numpy()
sc = ann.super_class.astype(str).to_numpy()
dn = np.flatnonzero(np.isin(sc, ["descending", "descending_neuron"]))
leg_mn = np.flatnonzero((sc == "vnc_motor") & (ann["leg"].fillna("").astype(str).to_numpy() != ""))
cc = ann.cell_class.astype(str).to_numpy()
en = ann["entry_nerve"].fillna("").astype(str).to_numpy()
bristle = np.flatnonzero(np.isin(cc, ["bristle_neuron", "mechanosensory_tactile"])
                         & (en == "MetaLN") & (side == "left"))
print(f"{DS}: {len(dn)} DNs, {len(leg_mn)} leg MNs, {len(bristle)} left hind-leg bristle neurons")

# static panorama: 16 dark posts on a 90 mm circle around the arena centre (visual texture)
posts = [((90 * np.cos(a), 90 * np.sin(a), 10.0), 9.0) for a in np.linspace(0, 2 * np.pi, 17)[:-1]]
orig_spheres = w.visual_spheres
w.visual_spheres = lambda me: orig_spheres(me) + posts
orig_vis = f.vision.rates
state = {"omega": 0.0, "t": 0.0, "touch": False}


def vis(pos, yaw, spheres, lux=None):
    # rotating the world by +omega*t around the fly == fly's view rotated by -omega*t
    return orig_vis(pos, yaw - state["omega"] * state["t"], spheres, lux)


f.vision.rates = vis
orig_sense = f.sense
base_stim = f.stim_idx.copy()


def sense(world):
    r = orig_sense(world)
    if state["touch"]:
        return np.concatenate([r, np.full(len(bristle), 100.0)])
    return r


f.sense = sense


def run(name, starved_h, odor, omega=0.0, touch=False, shock=0.0):
    f.brain.reset(1)
    f.motor.__init__(ann)
    f.meta = Metabolism.starved(starved_h)
    w.odor_scale = odor
    w.shocks["F"] = shock
    state.update(omega=np.radians(omega), t=0.0, touch=touch)
    f.stim_idx = np.concatenate([base_stim, bristle]) if touch else base_stim
    f.body.teleport((50.0, 0.0, 0.0), np.pi)
    acc = np.zeros(f.brain.con.n, np.int64)
    fwd, turn, pos0 = [], [], None
    t0 = time.time()
    n_settle, n_meas = 50, int(T / 0.01)
    for k in range(n_settle + n_meas):
        w.step(0.01)
        state["t"] += 0.01
        c = f.pop_counts()
        if k >= n_settle:
            if pos0 is None:
                pos0 = f.body.pos.copy()
            acc += c
            fwd.append(f.cmd["forward"] - f.cmd["backward"])
            turn.append(f.cmd["turn"])
    rate = acc / T
    res = {"cond": name, "wall_s": round(time.time() - t0, 1),
           "disp_mm": float(np.linalg.norm(f.body.pos[:2] - pos0[:2])),
           "cmd_forward_mean": float(np.mean(fwd)), "cmd_turn_mean": float(np.mean(turn)),
           "dn_mean_hz": float(rate[dn].mean()), "dn_active_frac": float((rate[dn] > 1).mean()),
           "legmn_mean_hz": float(rate[leg_mn].mean()) if len(leg_mn) else 0.0}
    for ty in CANDIDATES:
        for s in ("left", "right"):
            ids = np.flatnonzero((t == ty) & (side == s))
            res[f"{ty}_{s[0].upper()}"] = round(float(rate[ids].mean()), 2) if len(ids) else None
    return res, rate


conds = [("rest_fed", 0, 0.0), ("rest_hungry", 24, 0.0), ("hungry_food", 24, 1.0),
         ("fed_food", 0, 1.0), ("optomotor", 0, 0.0, 90.0), ("touch_hindleg", 0, 0.0, 0.0, True),
         ("shock_legs", 0, 0.0, 0.0, False, 60.0)]
rows, rates = [], {}
for c in conds:
    r, rate = run(*c)
    rows.append(r)
    rates[c[0]] = rate
    print(json.dumps(r), flush=True)

df = pd.DataFrame(rows).set_index("cond")
pd.set_option("display.width", 250)
print(df.T.to_string())
# which DNs does each stimulus recruit most, relative to rest_fed?
base = rates["rest_fed"]
top = {}
for c in rates:
    if c == "rest_fed":
        continue
    d = rates[c][dn] - base[dn]
    o = np.argsort(-d)[:12]
    top[c] = [(t[dn[i]] or "?", side[dn[i]], round(float(rates[c][dn[i]]), 1), round(float(d[i]), 1))
              for i in o if d[i] > 0.5]
    print(c, "top recruited DNs:", top[c])
# resting DN rate distribution
print("rest_fed DN rates: frac>0.5Hz", float((base[dn] > 0.5).mean()), "median", float(np.median(base[dn])),
      "top:", [(t[dn[i]], side[dn[i]], round(float(base[dn[i]]), 1)) for i in np.argsort(-base[dn])[:15]])
json.dump({"rows": rows, "top": top}, open(OUT / f"probe_walk_dns_{DS}.json", "w"), indent=1)
np.savez_compressed(OUT / f"probe_walk_dns_{DS}_rates.npz", dn=dn, **rates)
