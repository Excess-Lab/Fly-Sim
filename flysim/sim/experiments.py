"""Experiments. Each returns a dict of metrics and saves a recording for the viewer.

Conditions shared by the learning experiments (alternative-explanation controls):
  full            plasticity on, measured US->DAN input (A24/A25)
  no_plasticity   identical, but KC->MBON synapses fixed: only reflexes / innate wiring
  no_us_shortcut  plasticity on, but punishment must reach DANs through the connectome alone
"""
from pathlib import Path

import numpy as np

from flysim.agent.fly import Fly, FlyConfig
from flysim.sim.record import Recorder
from flysim.world.apple import Apple
from flysim.world.world import World, Zone

ROOT = Path(__file__).resolve().parents[2] / "results"

CONDITIONS = {
    "full": dict(plasticity=True, us_punish_shortcut=True),
    "no_plasticity": dict(plasticity=False, us_punish_shortcut=True),
    "no_us_shortcut": dict(plasticity=True, us_punish_shortcut=False),
    "no_reward_shortcut": dict(plasticity=True, us_reward_shortcut=False),
    "reflex_only": dict(plasticity=False, us_punish_shortcut=False, us_reward_shortcut=False),
}


def _rng(seed, tag):
    """Experiment-side random stream (start pose, counterbalancing), independent of the fly's
    own noise streams that also derive from `seed`."""
    return np.random.default_rng([int(seed), tag])


def _pose_on_circle(seed, r, center=(0.0, 0.0)):
    """Start on a circle of radius r around `center` at a random angle, random heading."""
    g = _rng(seed, 1)
    a, yaw = g.uniform(0, 2 * np.pi, 2)
    return (center[0] + r * np.cos(a), center[1] + r * np.sin(a), 0.0), float(yaw)


def _landmarks():
    """Visual landmarks for place learning: three dark spheres of different size outside the
    arena walls (non-odorous because the place protocols set odor_scale = 0). Flies see them
    through Vision (world.visual_spheres); they never reach them."""
    return [_apple(0, center=c, radius=r) for c, r in
            (((85.0, 30.0), 22.0), ((-80.0, -15.0), 12.0), ((10.0, -85.0), 16.0))]


def _apple(age_days, center=(0.0, 0.0), radius=35.0):
    from flysim.world.apple import AppleParams
    a = Apple(AppleParams(radius_mm=radius), center=center)
    a.age(age_days * 24)
    return a


def _frac_in(frames, center, radius, t0=None, t1=None):
    xs = [f for f in frames if (t0 is None or f["t"] >= t0) and (t1 is None or f["t"] <= t1)]
    if not xs:
        return float("nan")
    c = np.asarray(center)
    return float(np.mean([np.linalg.norm(np.asarray(f["pos"][:2]) - c) < radius for f in xs]))


def _path_metrics(frames, t0, t1):
    xs = np.array([f["pos"][:2] for f in frames if t0 <= f["t"] <= t1])
    if len(xs) < 3:
        return {"path_mm": 0.0, "net_mm": 0.0, "tortuosity": float("nan")}
    seg = np.linalg.norm(np.diff(xs, axis=0), axis=1).sum()
    net = np.linalg.norm(xs[-1] - xs[0])
    return {"path_mm": float(seg), "net_mm": float(net), "tortuosity": float(seg / max(net, 1e-3))}


def _first_time(frames, pred, t0=0.0):
    for f in frames:
        if f["t"] >= t0 and pred(f):
            return f["t"] - t0
    return float("nan")


# ------------------------------------------------------------------------------------------
def place_avoidance(cond="full", seed=0, t_base=30.0, t_train=30.0, t_test=30.0, zone_temp=44.0):
    """Heat-zone place learning (cf. Ofstad et al. 2011). Hot floor zone during training;
    avoidance = drop of time in the (now neutral) zone from baseline to test.

    Zone 44 C: leg nociceptors fire only above ~40 C (A80; 2 + 20 Hz/C), and every fly has a VNC
    since 0.4.0, so a 40 C zone delivered no punishment at all. 44 C is noxious for adults
    (hot-plate nociception assays use ~40-46 C, Neely et al. 2010; knockdown above ~39 C body
    temperature, Huey et al. 1992) yet far below the minutes needed for heat death (A94).
    Spatial cues: visual landmarks outside the walls (_landmarks); no odours (odor_scale = 0).
    Baseline and test windows have equal length and both start from the same pose (drawn from
    the seed), so the index is not biased by where the fly happened to be after training."""
    if t_test != t_base:
        raise ValueError("baseline and test windows must be equal (t_base == t_test)")
    marks = _landmarks()
    w = World(marks[0], apples=marks, arena=50.0)
    w.odor_scale = 0.0
    zone = Zone(center=(20.0, 20.0), radius=20.0, temp_c=zone_temp, active=False)
    w.zones.append(zone)
    g = _rng(seed, 2)
    while True:   # start anywhere in the arena at least 5 mm outside the zone
        xy = g.uniform(-42.0, 42.0, 2)
        if np.linalg.norm(xy - np.asarray(zone.center)) > zone.radius + 5.0:
            break
    start, yaw0 = (float(xy[0]), float(xy[1]), 0.0), float(g.uniform(0, 2 * np.pi))
    f = Fly(FlyConfig("F", "female", starved_h=0, seed=seed, **CONDITIONS[cond]), pos=start, yaw=yaw0)
    w.flies.append(f)
    w.rebuild_physics()
    rec = Recorder(w)
    rec.run(t_base, log_every=10)
    zone.active = True
    rec.run(t_train, log_every=10, t0=t_base)
    zone.active = False
    f.body.teleport(start, yaw0)
    rec.run(t_test, log_every=10, t0=t_base + t_train)
    fr = rec.frames["F"]
    m = {
        "experiment": "place_avoidance", "condition": cond, "seed": seed, "zone_temp": zone_temp,
        "start": [round(v, 1) for v in start], "start_yaw": round(yaw0, 3),
        "frac_zone_baseline": _frac_in(fr, zone.center, zone.radius, 0, t_base),
        "frac_zone_training": _frac_in(fr, zone.center, zone.radius, t_base, t_base + t_train),
        "frac_zone_test": _frac_in(fr, zone.center, zone.radius, t_base + t_train, None),
        "mbon_w_min_end": fr[-1]["mbon_w_min"],
    }
    m["avoidance_index"] = m["frac_zone_baseline"] - m["frac_zone_test"]
    m.update(t_base=t_base, t_train=t_train, t_test=t_test)
    rec.save(ROOT / f"place_avoidance_{cond}_s{seed}", meta=m)
    return m


ODOURS = ({"ethyl acetate": 1.0}, {"isoamyl acetate": 1.0})


def tmaze_choice(cond="full", seed=0, retention_h=(0.0,), t_train=30.0, t_test=30.0):
    """Two-odour choice after aversive conditioning (cf. Tully & Quinn 1985).
    Reciprocal design: which odour is punished (CS+) and on which side of the maze CS+ is
    offered at test are counterbalanced by the seed (seed % 2 -> CS+ odour, (seed // 2) % 2 ->
    CS+ side), so 4 consecutive seeds cover every combination. CS+ is paired with shocks, CS-
    is presented without. PI = time near CS- - time near CS+ (fraction of test) for this half;
    protocol.analyze() averages the two reciprocal halves into one N, as Tully & Quinn did.
    Retention: passive forgetting applied between training and each test (A23)."""
    k_odour, k_side = seed % 2, (seed // 2) % 2
    cs_plus, cs_minus = ODOURS[k_odour], ODOURS[1 - k_odour]
    x_plus = -40.0 if k_side == 0 else 40.0
    w = World(_apple(0, center=(-500.0, -500.0), radius=5.0), arena=50.0)
    w.odor_scale = 0.0
    g = _rng(seed, 3)
    start = (0.0, float(g.uniform(-3.0, 3.0)), 0.0)    # choice point, small jitter
    yaw0 = float(g.uniform(0, 2 * np.pi))
    f = Fly(FlyConfig("F", "female", starved_h=0, seed=seed, **CONDITIONS[cond]), pos=start, yaw=yaw0)
    w.flies.append(f)
    w.rebuild_physics()
    rec = Recorder(w)
    t = 0.0
    # Tully & Quinn 1985: CS+ with 12 x 1.25 s shocks over 60 s (one every 5 s), 45 s rest,
    # CS- 60 s without shocks. Durations scaled by t_train / 60.
    scale = t_train / 60.0
    w.extra_odor_sources = [((0, 0, 0), cs_plus, 50.0)]
    for k in range(int(60 * scale / 0.01)):
        w.shocks["F"] = 60.0 if ((k * 0.01) / scale) % 5.0 < 1.25 else 0.0
        w.step(0.01)
        t += 0.01
    w.shocks["F"] = 0.0
    f.pop_counts()  # unrecorded training spikes must not leak into the next frame
    w.extra_odor_sources = []
    rec.run(45 * scale, log_every=0, t0=t)
    t += 45 * scale
    w.extra_odor_sources = [((0, 0, 0), cs_minus, 50.0)]
    rec.run(60 * scale, log_every=0, t0=t)
    t += 60 * scale
    results = {"experiment": "tmaze_choice", "condition": cond, "seed": seed,
               "cs_plus": next(iter(cs_plus)), "cs_plus_side": "left" if x_plus < 0 else "right",
               "mbon_w_min_after_training": float(f.brain.kc_mbon_strength().min()) if f.brain.pl_on else 1.0,
               "tests": []}
    elapsed_h = 0.0
    sgn = np.sign(x_plus)          # +1: CS+ on the right
    for h in retention_h:
        f.pass_time(h - elapsed_h, w)  # retention interval: the fly lives on, memories fade
        elapsed_h = h
        f.body.teleport(start, yaw0)
        w.extra_odor_sources = [((x_plus, 0, 0), cs_plus, 8.0), ((-x_plus, 0, 0), cs_minus, 8.0)]
        t_start = t
        rec.run(t_test, log_every=0, t0=t)
        t += t_test
        fr = [x for x in rec.frames["F"] if x["t"] > t_start]
        xs = sgn * np.array([x["pos"][0] for x in fr])   # >0 = CS+ side
        near_plus = float(np.mean(xs > 10))
        near_minus = float(np.mean(xs < -10))
        lat = _first_time(fr, lambda x: abs(x["pos"][0]) > 10, t0=t_start)
        out_i = np.flatnonzero(np.abs(xs) > 10)
        side = ("CS+" if xs[out_i[0]] > 0 else "CS-") if out_i.size else "none"
        # vacillation: number of crossings of the midline (indecision indicator, see TRACKING.md)
        cross = int(np.sum(np.diff(np.sign(xs[np.abs(xs) > 2])) != 0))
        results["tests"].append({"retention_h": h, "PI": near_minus - near_plus,
                                 "near_CS+": near_plus, "near_CS-": near_minus, "first_choice": side,
                                 "decision_latency_s": lat, "midline_crossings": cross,
                                 "mbon_w_min": float(f.brain.kc_mbon_strength().min()) if f.brain.pl_on else 1.0})
    rec.save(ROOT / f"tmaze_{cond}_s{seed}", meta=results)
    return results


def food_search_after_aversion(cond="full", seed=0, t_naive=40.0, t_train=20.0, t_test=40.0):
    """Hungry fly finds a fermenting apple; then is punished while on it; then search again.
    Measures change of search strategy: latency to reach food, tortuosity, time on apple."""
    w = World(_apple(2), arena=60.0)
    start, yaw0 = _pose_on_circle(seed, 50.0)   # 15 mm from the fruit surface, random side/heading
    f = Fly(FlyConfig("F", "female", starved_h=24, seed=seed, **CONDITIONS[cond]),
            pos=start, yaw=yaw0)
    w.flies.append(f)
    w.rebuild_physics()
    rec = Recorder(w)
    rec.run(t_naive, log_every=10)
    on = lambda x: x["surface"] == "apple"  # noqa: E731
    fr = rec.frames["F"]
    naive = {"latency_to_apple_s": _first_time(fr, on), "frac_on_apple":
             float(np.mean([on(x) for x in fr])), **_path_metrics(fr, 0, t_naive)}
    # training: punishment whenever on the apple
    t = t_naive
    for _ in range(int(t_train / 0.01)):
        w.shocks["F"] = 60.0 if f.body.surface == "apple" else 0.0
        w.step(0.01)
        t += 0.01
    w.shocks["F"] = 0.0
    f.pop_counts()
    f.body.teleport(start, yaw0)
    t0 = t
    rec.run(t_test, log_every=10, t0=t)
    fr2 = [x for x in rec.frames["F"] if x["t"] > t0]
    after = {"latency_to_apple_s": _first_time(fr2, on, t0=t0),
             "frac_on_apple": float(np.mean([on(x) for x in fr2])),
             **_path_metrics(fr2, t0, t0 + t_test)}
    m = {"experiment": "food_search_after_aversion", "condition": cond, "seed": seed,
         "naive": naive, "after_aversion": after, "hunger_end": fr2[-1]["hunger"]}
    rec.save(ROOT / f"food_search_{cond}_s{seed}", meta=m)
    return m


def conflict(starved_h=24, seed=0, t=60.0, ring_temp=42.0):
    """Food vs aversion: fermenting apple surrounded by a hot floor ring. Hungry vs fed.
    Ring 42 C: above the leg-nociceptor threshold (~40 C, A80), i.e. noxious but crossable in
    seconds (heat death needs minutes above 40 C body temperature, A94); hot-plate assays use
    ~40-46 C (Neely et al. 2010). The former 34 C ring was below the nociceptor threshold and
    therefore not aversive for flies with a VNC."""
    w = World(_apple(2, radius=20.0), arena=60.0)
    w.zones.append(Zone(center=(0.0, 0.0), radius=32.0, temp_c=ring_temp))
    start, yaw0 = _pose_on_circle(seed, 50.0)
    f = Fly(FlyConfig("F", "female", starved_h=starved_h, seed=seed), pos=start, yaw=yaw0)
    w.flies.append(f)
    w.rebuild_physics()
    rec = Recorder(w)
    rec.run(t, log_every=15)
    fr = rec.frames["F"]
    m = {"experiment": "conflict", "starved_h": starved_h, "ring_temp": ring_temp, "seed": seed,
         "frac_on_apple": float(np.mean([x["surface"] == "apple" for x in fr])),
         "frac_in_warm_ring": float(np.mean([x["temp"] > w.ambient_c + 5 for x in fr])),
         "latency_to_apple_s": _first_time(fr, lambda x: x["surface"] == "apple"),
         "feeding_frac": float(np.mean([x["proboscis"] for x in fr]))}
    rec.save(ROOT / f"conflict_h{starved_h}_T{int(ring_temp)}_s{seed}", meta=m)
    return m


def pair(mode="courtship", seconds=60.0, apple_days=2, seed=0):
    """Two independent flies at an apple. courtship: male + female; fight: two males."""
    w = World(_apple(apple_days), arena=60.0)
    if mode == "courtship":
        cfgs = [FlyConfig("male", "male", starved_h=12, seed=seed),
                FlyConfig("female", "female", starved_h=12, seed=seed + 1)]
    else:
        cfgs = [FlyConfig("male_A", "male", starved_h=12, seed=seed),
                FlyConfig("male_B", "male", starved_h=12, seed=seed + 1)]
    # small-chamber start (courtship assays use ~10 mm chambers), next to the apple
    starts = [((-48, -4, 0), 0.3), ((-41, 4, 0), np.pi + 0.3)]
    for c, (p, y) in zip(cfgs, starts):
        w.flies.append(Fly(c, pos=p, yaw=y))
    w.rebuild_physics()
    rec = Recorder(w)
    rec.run(seconds, log_every=10)
    m = {"experiment": f"pair_{mode}", "seed": seed, "apple": w.apple.summary(),
         "events": {f.cfg.name: f.events for f in w.flies}}
    for f in w.flies:
        fr = rec.frames[f.cfg.name]
        m[f.cfg.name] = {
            "frac_on_apple": float(np.mean([x["surface"] == "apple" for x in fr])),
            "feeding_frac": float(np.mean([x["proboscis"] for x in fr])),
            "song_frac": float(np.mean([x["wing"] > 0.3 for x in fr])),
            "max_P1_Hz": float(max(x["rates"]["arousal_P1"] for x in fr)),
            "lunges": sum(1 for _, e in f.events if e.startswith("lunge->")),
            "copulation": any("copulation" in e for _, e in f.events),
            "flights": sum(1 for _, e in f.events if "->('fly'" in e),
        }
    d = [np.linalg.norm(np.asarray(a["pos"]) - np.asarray(b["pos"]))
         for a, b in zip(*[rec.frames[f.cfg.name] for f in w.flies])]
    m["mean_distance_mm"] = float(np.mean(d))
    m["frac_within_5mm"] = float(np.mean(np.asarray(d) < 5))
    rec.save(ROOT / f"pair_{mode}_s{seed}", meta=m)
    return m
