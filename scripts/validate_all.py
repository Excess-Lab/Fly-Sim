"""Correctness checks for every fly nervous system (run before each release).

  .venv/bin/python scripts/validate_all.py [female_banc male female] [--long 20]

For each dataset:
 1. coverage   - every sensory group, motor read-out and learning group is non-empty
 2. propagation- each sensory group at 100 Hz for 0.5 s activates more neurons downstream
                 (direct postsynaptic, non-sensory) than at baseline
 3. signs      - transmitters of cells with well-known identity (APL GABA, KC ACh, uPN ACh,
                 MN9 ACh, lLN GABA mostly)
 4. stability  - 'long' seconds with resting sensory input: active neurons per second stay
                 bounded (no runaway, no silence)
 5. determinism- same seed gives identical spikes, different seed differs
Writes results/validation/<dataset>.json and prints a table. Exit code 1 on any failure.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation"


def fly_for(ds):
    from flysim.agent.fly import Fly, FlyConfig
    sex = "male" if ds == "male" else "female"
    return Fly(FlyConfig("V", sex, starved_h=0, seed=1, connectome=ds), pos=(0, 0, 0), yaw=0.0)


def check(ds, long_s):
    t0 = time.time()
    f = fly_for(ds)
    b, ann = f.brain, f.ann
    t = pd.Series(ann.type.astype(str).to_numpy())
    res = {"dataset": ds, "checks": []}

    def rec(name, ok, value, expect):
        res["checks"].append({"check": name, "ok": bool(ok), "value": value, "expect": expect})

    # ---- 1. coverage ----------------------------------------------------------------------
    groups = {"olfaction ORNs": f.olf.idx, "thermo hot": f.thermo.hot, "thermo cold": f.thermo.cold,
              "vision photoreceptors": f.vision.idx, "hearing JO-B": f.hearing.idx,
              "wind JO-C/E": f.windsense.idx, "interoception IPC": f.intero.idx,
              "contact pheromone ppk23": f.ppk23}
    groups.update({f"taste {k}": v for k, v in f.gust.groups.items()})
    has_vnc = len(f.noci_idx) > 0
    if f.chain is not None:
        groups.update({f"nociceptors {leg}": v for leg, v in f.leg_noci.items()})
        groups.update({"proprioceptors": f.prop_idx, "halteres": f.haltere_idx})
    groups.update({f"motor {k}": v for k, v in f.motor.groups.items()})
    # sex-specific cells that must exist in one sex and are absent in the other
    SEX_SPECIFIC = {"male": {"motor receptive", "motor ovi"}, "female": {"motor song", "motor arousal_P1"}}
    sex = "male" if ds == "male" else "female"
    for k, v in groups.items():
        if k in SEX_SPECIFIC[sex]:
            rec(f"coverage: {k} (absent in this sex)", len(v) == 0, int(len(v)), "0 (sex-specific)")
        else:
            rec(f"coverage: {k}", len(v) > 0, int(len(v)), "> 0")
    # both eyes see something
    az = np.arctan2(f.vision.dirs[:, 1], f.vision.dirs[:, 0])
    left = float((az > 0).mean()) if len(az) else 0.0
    rec("coverage: vision both eyes (left share)", 0.05 < left < 0.95, round(left, 2), "0.05..0.95")

    # ---- 2. propagation -------------------------------------------------------------------
    sens = set(np.flatnonzero(ann.super_class.astype(str).str.contains("sensory").to_numpy()))
    base, _ = b.run(500.0, stim={}, seed=3)
    # vision: photoreceptors are histaminergic (inhibitory) - light must SUPPRESS the tonic
    # first-order cells (A54c); insulin cells act hormonally (no synaptic test)
    tv = getattr(b, "tonic_visual", np.zeros(0, int))
    if len(tv):
        dark, _ = b.run(500.0, stim={int(i): 5.0 for i in f.vision.idx}, seed=3)
        lit, _ = b.run(500.0, stim={int(i): 60.0 for i in f.vision.idx}, seed=3)
        d, l_ = dark[tv].mean() * 2, lit[tv].mean() * 2
        rec("propagation: vision (light suppresses first-order cells)", l_ < 0.7 * d,
            f"dark {d:.1f} Hz, bright {l_:.1f} Hz", "bright < 0.7 x dark")
    else:
        rec("propagation: vision tonic first-order cells", False, 0, "> 0")
    prop = {k: v for k, v in groups.items() if not k.startswith("motor") and len(v)
            and k not in ("vision photoreceptors", "interoception IPC")}
    for k, idx in prop.items():
        idx = np.asarray(idx)
        post = np.unique(np.concatenate([b.con.post[b.con.indptr[i]:b.con.indptr[i + 1]] for i in idx]))
        own = set(idx.tolist())
        post = np.array([p for p in post if p not in sens and p not in own])
        if len(post) == 0:
            rec(f"propagation: {k}", False, "no non-sensory targets", "targets > 0")
            continue
        c, _ = b.run(500.0, stim={int(i): 100.0 for i in idx}, seed=3)
        n_on, n_base = int((c[post] > 0).sum()), int((base[post] > 0).sum())
        rec(f"propagation: {k}", n_on > n_base, f"{n_on}/{len(post)} targets (rest {n_base})", "more than at rest")

    # ---- 3. transmitter signs of well-known cells ------------------------------------------
    nt = ann.nt.astype(str).str.lower().to_numpy()
    def share(pat, want):
        m = t.str.match(pat).to_numpy()
        return (float(np.isin(nt[m], want).mean()) if m.any() else None), int(m.sum())
    for pat, want, need, label in ((r"^APL$", ["gaba"], 0.99, "APL is GABAergic"),
                                   (r"^KC", ["acetylcholine", "ach"], 0.9, "Kenyon cells cholinergic"),
                                   (r"^[A-Z][A-Za-z0-9]*_[a-z0-9]*adPN$", ["acetylcholine", "ach"], 0.9,
                                    "adPN projection neurons cholinergic"),
                                   (r"^(MN9|CB0701)$", ["acetylcholine", "ach"], 0.99, "MN9 cholinergic"),
                                   (r"^(R1-6|R1-R6|R7|R8)", ["histamine"], 0.99, "photoreceptors histaminergic"),
                                   (r"^lLN2P_b$", ["gaba"], 0.7, "lLN2P_b GABAergic")):
        s, n = share(pat, want)
        rec(f"sign: {label}", s is not None and s >= need, None if s is None else f"{s:.2f} of {n}", f">= {need}")

    # ---- 4. long stability with resting input ----------------------------------------------
    rest = {}
    rest.update({int(i): 8.0 for i in f.olf.idx})
    rest.update({int(i): 1.0 for i in f.gust.idx})
    rest.update({int(i): 2.0 for i in f.thermo.idx})
    rest.update({int(i): 30.0 for i in f.vision.idx})
    rest.update({int(i): 5.0 for i in f.windsense.idx})
    rest.update({int(i): 20.0 for i in f.prop_idx})
    b.reset(5)
    idx = np.array(list(rest), np.int64)
    rates = np.array([rest[i] for i in idx])
    b.set_poisson_targets(idx)
    act = []
    for s in range(int(long_s)):
        c, _ = b.step(int(1000 / b.p.dt), idx, rates)
        act.append(int((c > 0).sum()))
    frac = np.array(act) / b.con.n
    half = len(act) // 2
    trend = (np.mean(act[half:]) + 1) / (np.mean(act[:half]) + 1) if half else 1.0
    rec("stability: active fraction per s (max)", frac.max() < 0.15, round(float(frac.max()), 4), "< 0.15")
    rec("stability: not silent (min active per s)", min(act) > 100, int(min(act)), "> 100")
    rec("stability: no runaway (2nd half / 1st half)", trend < 1.5, round(float(trend), 2), "< 1.5")
    res["active_per_s"] = act

    # ---- 5. determinism ---------------------------------------------------------------------
    st = {int(i): 100.0 for i in f.gust.groups["sugar"]}
    a1, _ = b.run(300.0, stim=st, seed=11)
    a2, _ = b.run(300.0, stim=st, seed=11)
    a3, _ = b.run(300.0, stim=st, seed=12)
    rec("determinism: same seed identical", np.array_equal(a1, a2), int(np.abs(a1 - a2).sum()), "0 differences")
    rec("determinism: other seed differs", not np.array_equal(a1, a3), int(np.abs(a1 - a3).sum()), "> 0")
    res["seconds"] = round(time.time() - t0)
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    long_s = 20
    if "--long" in sys.argv:
        long_s = int(sys.argv[sys.argv.index("--long") + 1])
        args = [a for a in args if a != str(long_s)]
    dsets = args or ["female_banc", "male"]
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for ds in dsets:
        r = check(ds, long_s)
        (OUT / f"{ds}.json").write_text(json.dumps(r, indent=1, default=str, ensure_ascii=False))
        print(f"\n=== {ds} ({r['seconds']} s)")
        for c in r["checks"]:
            bad += not c["ok"]
            print(f"  {'OK  ' if c['ok'] else 'FAIL'} {c['check']:<48} {c['value']}  (expect {c['expect']})", flush=True)
    print(f"\n{bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
