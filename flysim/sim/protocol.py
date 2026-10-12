"""Experiment protocols, replication runner, statistics and power analysis.

Separation of roles (the organism never knows it is in an experiment):
  Organism    flysim.agent.fly.Fly            - senses, brain, body; only reacts to the world
  Environment flysim.world.*                   - apple, air, zones, odour sources, shocks
  Protocol    PROTOCOLS below                  - hypothesis, groups, measurements, replications
  Runner      run_series()                     - executes replicates (multi-process)
  Analyzer    analyze()                        - means, CIs, effect sizes, power
  Report      scripts/report.py                - human-readable report

Each replicate is reproducible from (protocol, group, seed, model version) -> experiment_id.
"""
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from flysim.sim.experiments import ROOT

PROTOCOLS = {
    "tmaze_choice": {
        "hypothesis": "After pairing odour A with punishment, flies avoid A relative to B; the "
                      "effect needs KC->MBON plasticity and decays over hours.",
        "function": "tmaze_choice",
        "groups": {"experimental": {"cond": "full"}, "no_plasticity": {"cond": "no_plasticity"},
                   "connectome_only": {"cond": "reflex_only"}},
        "metric": ("tests", 0, "PI"),
        "kwargs": {"retention_h": [0.0]},
        "reference": "real flies: PI ~0.8-0.9 immediately (Tully & Quinn 1985)",
    },
    "place_avoidance": {
        "hypothesis": "Flies spend less time in a place previously paired with heat.",
        "function": "place_avoidance",
        "groups": {"experimental": {"cond": "full"}, "no_plasticity": {"cond": "no_plasticity"}},
        "metric": ("avoidance_index",),
        "kwargs": {},
        "reference": "Ofstad et al. 2011: visual place learning in a heat arena",
    },
    "conflict": {
        "hypothesis": "Hungry flies tolerate a warm ring to reach fermenting fruit more than fed flies.",
        "function": "conflict",
        "groups": {"hungry_24h": {"starved_h": 24}, "fed": {"starved_h": 0}},
        "metric": ("frac_on_apple",),
        "kwargs": {},
        "reference": "Sayin et al. 2019; Lewis et al. 2015",
    },
    "food_search": {
        "hypothesis": "Punishment on the fruit delays the return to it (latency increases).",
        "function": "food_search_after_aversion",
        "groups": {"experimental": {"cond": "full"}, "no_plasticity": {"cond": "no_plasticity"}},
        "metric": ("after_aversion", "frac_on_apple"),
        "kwargs": {},
        "reference": "-",
    },
}


def _get(d, path):
    for p in path:
        d = d[p]
    return d


def _run_one(args):
    name, group, seed, gkw, kw = args
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    from flysim.sim import experiments as E
    t0 = time.time()
    fn = getattr(E, PROTOCOLS[name]["function"])
    res = fn(seed=seed, **gkw, **kw)
    res.update(protocol=name, group=group, seed=seed, wall_s=round(time.time() - t0))
    return res


def run_series(name, n_reps=5, processes=1, seed0=100, out=None, **override):
    """Run n_reps independent replicates (different seeds) of every group."""
    p = PROTOCOLS[name]
    kw = {**p["kwargs"], **override}
    if "retention_h" in kw:
        kw["retention_h"] = tuple(kw["retention_h"])
    jobs = [(name, g, seed0 + r, gkw, kw) for g, gkw in p["groups"].items() for r in range(n_reps)]
    out = Path(out or ROOT / f"series_{name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if processes > 1:
        with mp.get_context("spawn").Pool(processes) as pool:
            for res in pool.imap_unordered(_run_one, jobs):
                results.append(res)
                with open(out, "a") as fh:
                    fh.write(json.dumps(res, default=float) + "\n")
    else:
        for j in jobs:
            res = _run_one(j)
            results.append(res)
            with open(out, "a") as fh:
                fh.write(json.dumps(res, default=float) + "\n")
    return results


# ---------------------------------------------------------------------------------------------
def _t_ppf(q, df):
    """Student t quantile (Hill 1970 approximation, adequate for CIs)."""
    from scipy import stats
    return float(stats.t.ppf(q, df))


def analyze(name, path=None) -> dict:
    p = PROTOCOLS[name]
    path = Path(path or ROOT / f"series_{name}.jsonl")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        try:
            v = float(_get(r, p["metric"]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not math.isnan(v):
            by.setdefault(r["group"], []).append(v)
    out = {"protocol": name, "hypothesis": p["hypothesis"], "reference": p["reference"],
           "metric": ".".join(map(str, p["metric"])), "groups": {}}
    for g, xs in by.items():
        x = np.array(xs)
        n = len(x)
        sd = float(x.std(ddof=1)) if n > 1 else float("nan")
        half = _t_ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        out["groups"][g] = {"n": n, "mean": float(x.mean()), "sd": sd,
                            "ci95": [float(x.mean() - half), float(x.mean() + half)]}
    gs = list(by)
    if len(gs) >= 2:
        a, b = np.array(by[gs[0]]), np.array(by[gs[1]])
        if len(a) > 1 and len(b) > 1:
            from scipy import stats
            sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / sp if sp > 0 else float("inf")
            t = stats.ttest_ind(a, b, equal_var=False)
            u = stats.mannwhitneyu(a, b) if len(set(a) | set(b)) > 1 else None
            out["comparison"] = {"groups": gs[:2], "cohens_d": d, "welch_t_p": float(t.pvalue),
                                 "mannwhitney_p": float(u.pvalue) if u else None,
                                 "n_per_group_for_80pct_power": power_n(d)}
    return out


def power_n(d, alpha=0.05, power=0.8) -> int | None:
    """Replicates per group needed to detect effect size d (two-sided two-sample t-test)."""
    if d is None or not math.isfinite(d) or d == 0:
        return None
    from scipy import stats
    for n in range(2, 10000):
        df = 2 * n - 2
        nc = abs(d) * math.sqrt(n / 2)
        tc = stats.t.ppf(1 - alpha / 2, df)
        pw = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
        if pw >= power:
            return n
    return None


def compute_budget(name, n_reps, seconds_per_sim_s=10.0) -> dict:
    """Rough wall-clock estimate for a series on this machine class (M2: ~10 s per sim-s/fly)."""
    sim_s = {"tmaze_choice": 30 + 45 + 30 + 30, "place_avoidance": 80, "conflict": 60,
             "food_search": 100}.get(name, 60)
    groups = len(PROTOCOLS[name]["groups"])
    total = sim_s * seconds_per_sim_s * n_reps * groups
    return {"replicates": n_reps * groups, "sim_seconds_each": sim_s,
            "wall_hours_single_process": round(total / 3600, 1)}
