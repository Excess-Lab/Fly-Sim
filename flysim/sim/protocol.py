"""Experiment protocols, replication runner, statistics and power analysis.

Separation of roles (the organism never knows it is in an experiment):
  Organism    flysim.agent.fly.Fly            - senses, brain, body; only reacts to the world
  Environment flysim.world.*                   - apple, air, zones, odour sources, shocks
  Protocol    PROTOCOLS below                  - hypothesis, groups, measurements, replications
  Runner      run_series()                     - executes replicates (multi-process)
  Analyzer    analyze()                        - means, CIs, effect sizes, power
  Report      scripts/report.py                - human-readable report

Each replicate is reproducible from (protocol, group, seed, model version) -> experiment_id.
Groups use independent seed streams (group_seed); start pose is drawn from the seed.
"""
import json
import math
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np

from flysim.sim.experiments import ROOT

PROTOCOLS = {
    # Every protocol names its control group; analyze() compares each other group with it.
    # Learning protocols with a punishment (US) include `no_us_shortcut` (EXPERIMENT_DESIGN.md):
    # it removes the A24 physiology shortcut, so the US must reach the DANs through the
    # connectome. For flies with a VNC (all flies since 0.4.0) heat already travels through leg
    # nociceptors -> VNC -> brain, so in heat protocols `no_us_shortcut` ~ `experimental`; in
    # shock protocols (tmaze_choice, food_search) it removes the only path from shock to PPL1.
    "tmaze_choice": {
        "hypothesis": "After pairing odour CS+ with punishment, flies avoid CS+ relative to CS-; "
                      "the effect needs KC->MBON plasticity and decays over hours.",
        "function": "tmaze_choice",
        "control": "no_plasticity",
        "groups": {"experimental": {"cond": "full"}, "no_plasticity": {"cond": "no_plasticity"},
                   "no_us_shortcut": {"cond": "no_us_shortcut"},
                   "connectome_only": {"cond": "reflex_only"}},
        "metric": ("tests", 0, "PI"),
        # Tully & Quinn 1985: one N = the mean PI of two reciprocal half-experiments (CS+ = odour 1
        # in one half, odour 2 in the other). Records are paired by this key within a group.
        "reciprocal": "cs_plus",
        "kwargs": {"retention_h": [0.0]},
        "reference": "real flies: PI ~0.8-0.9 immediately (Tully & Quinn 1985)",
    },
    "place_avoidance": {
        "hypothesis": "Flies spend less time in a place previously paired with noxious heat.",
        "function": "place_avoidance",
        "control": "no_plasticity",
        "groups": {"experimental": {"cond": "full"}, "no_plasticity": {"cond": "no_plasticity"},
                   "no_us_shortcut": {"cond": "no_us_shortcut"}},
        "metric": ("avoidance_index",),
        "kwargs": {},
        "reference": "Ofstad et al. 2011: visual place learning in a heat arena",
    },
    "conflict": {
        "hypothesis": "Hungry flies tolerate a noxious hot ring to reach fermenting fruit more than fed flies.",
        "function": "conflict",
        "control": "fed",
        "groups": {"hungry_24h": {"starved_h": 24}, "fed": {"starved_h": 0}},
        "metric": ("frac_on_apple",),
        "kwargs": {},
        "reference": "Sayin et al. 2019; Lewis et al. 2015",
    },
    "food_search": {
        "hypothesis": "Punishment on the fruit delays the return to it (latency increases).",
        "function": "food_search_after_aversion",
        "control": "no_plasticity",
        "groups": {"experimental": {"cond": "full"}, "no_plasticity": {"cond": "no_plasticity"},
                   "no_us_shortcut": {"cond": "no_us_shortcut"}},
        "metric": ("after_aversion", "frac_on_apple"),
        "kwargs": {},
        "reference": "-",
    },
}


def _get(d, path):
    for p in path:
        d = d[p]
    return d


def group_seed(name: str, group: str, rep: int, seed0: int = 100) -> int:
    """Independent seed stream per (protocol, group): groups never share seeds, so the
    unpaired tests in analyze() are valid. Stable across runs (crc32, not Python's hash)."""
    return int(seed0 + (zlib.crc32(f"{name}/{group}".encode()) % 100_000) * 1000 + rep)


def _run_one(args):
    name, group, seed, gkw, kw, run_id = args
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    from flysim.sim import experiments as E
    t0 = time.time()
    fn = getattr(E, PROTOCOLS[name]["function"])
    res = fn(seed=seed, **gkw, **kw)
    res.update(protocol=name, group=group, seed=seed, run_id=run_id, wall_s=round(time.time() - t0))
    return res


def run_series(name, n_reps=5, processes=1, seed0=100, out=None, **override):
    """Run n_reps independent replicates of every group (independent seed streams per group).
    Each call gets a run_id; analyze() uses only the latest run unless told otherwise."""
    p = PROTOCOLS[name]
    if p.get("reciprocal") and n_reps % 2:
        n_reps += 1   # reciprocal halves come in pairs
    kw = {**p["kwargs"], **override}
    if "retention_h" in kw:
        kw["retention_h"] = tuple(kw["retention_h"])
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    jobs = [(name, g, group_seed(name, g, r, seed0), gkw, kw, run_id)
            for g, gkw in p["groups"].items() for r in range(n_reps)]
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
    from scipy import stats
    return float(stats.t.ppf(q, df))


def _select(rows, run_id="latest", model_version=None):
    """Keep records of one run (default: the latest run_id in the file) and/or one model version.
    run_id='all' keeps every run (only meaningful together with model_version)."""
    if model_version is not None:
        rows = [r for r in rows if r.get("model_version") == model_version]
    if run_id == "latest":
        ids = [r["run_id"] for r in rows if r.get("run_id")]
        run_id = ids[-1] if ids else None
        if run_id is None:
            return [], None
    if run_id != "all":
        rows = [r for r in rows if r.get("run_id") == run_id]
    return rows, run_id


def _holm(ps):
    """Holm-Bonferroni adjusted p-values (None stays None)."""
    idx = [i for i, p in enumerate(ps) if p is not None and math.isfinite(p)]
    adj = [None] * len(ps)
    m = len(idx)
    running = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: ps[i])):
        running = max(running, min(1.0, (m - k) * ps[i]))
        adj[i] = running
    return adj


def _values(rows, p):
    """Metric per group. Reciprocal protocols: one value = mean of a CS+ = odour-1 half and a
    CS+ = odour-2 half (paired in seed order within the group), as in Tully & Quinn 1985."""
    by, dropped = {}, {}
    key = p.get("reciprocal")
    halves = {}
    for r in sorted(rows, key=lambda r: r.get("seed", 0)):
        try:
            v = float(_get(r, p["metric"]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        if key:
            halves.setdefault(r["group"], {}).setdefault(r.get(key, "?"), []).append(v)
        else:
            by.setdefault(r["group"], []).append(v)
    for g, hs in halves.items():
        lists = list(hs.values())
        if len(lists) != 2:
            dropped[g] = sum(map(len, lists))
            continue
        n = min(map(len, lists))
        by[g] = [(a + b) / 2 for a, b in zip(lists[0][:n], lists[1][:n])]
        dropped[g] = sum(map(len, lists)) - 2 * n
    return by, dropped


def analyze(name, path=None, run_id="latest", model_version=None) -> dict:
    """Per-group mean, SD, 95 % CI; every group vs the protocol's control group: mean
    difference (group - control) with Welch CI, Cohen's d (sign = group - control), Welch t and
    Mann-Whitney p (raw and Holm-adjusted across comparisons), n per group for 80 % power."""
    from scipy import stats
    p = PROTOCOLS[name]
    path = Path(path or ROOT / f"series_{name}.jsonl")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("protocol", name) == name]
    rows, used_run = _select(rows, run_id, model_version)
    by, dropped = _values(rows, p)
    out = {"protocol": name, "hypothesis": p["hypothesis"], "reference": p["reference"],
           "metric": ".".join(map(str, p["metric"])), "run_id": used_run,
           "model_versions": sorted({str(r.get("model_version")) for r in rows}),
           "control": p["control"], "groups": {}, "comparisons": {}}
    if p.get("reciprocal"):
        out["unit"] = "pair of reciprocal half-experiments (Tully & Quinn 1985)"
        out["unpaired_records_dropped"] = dropped
    for g, xs in by.items():
        x = np.array(xs)
        n = len(x)
        sd = float(x.std(ddof=1)) if n > 1 else float("nan")
        half = _t_ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        out["groups"][g] = {"n": n, "mean": float(x.mean()), "sd": sd,
                            "ci95": [float(x.mean() - half), float(x.mean() + half)]}
    ctrl = p["control"]
    if ctrl not in by or len(by[ctrl]) < 2:
        out["note"] = f"control group {ctrl!r} missing or n<2: no comparisons"
        return out
    c = np.array(by[ctrl])
    comps = {}
    for g, xs in by.items():
        if g == ctrl:
            continue
        a = np.array(xs)
        if len(a) < 2:
            continue
        sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(c) - 1) * c.var(ddof=1)) / (len(a) + len(c) - 2))
        diff = float(a.mean() - c.mean())
        d = diff / sp if sp > 0 else float("nan")
        va, vc = a.var(ddof=1) / len(a), c.var(ddof=1) / len(c)
        se = math.sqrt(va + vc)
        dfw = (va + vc) ** 2 / ((va ** 2 / (len(a) - 1)) + (vc ** 2 / (len(c) - 1))) if se > 0 else 1
        hw = _t_ppf(0.975, dfw) * se if se > 0 else 0.0
        t = stats.ttest_ind(a, c, equal_var=False) if se > 0 else None
        u = stats.mannwhitneyu(a, c, alternative="two-sided") if len(set(a) | set(c)) > 1 else None
        comps[f"{g} vs {ctrl}"] = {
            "diff_mean": diff, "diff_ci95_welch": [diff - hw, diff + hw], "cohens_d": d,
            "welch_t_p": float(t.pvalue) if t is not None else None,
            "mannwhitney_p": float(u.pvalue) if u is not None else None,
            "n_per_group_for_80pct_power": power_n(d)}
    keys = list(comps)
    for test in ("welch_t_p", "mannwhitney_p"):
        adj = _holm([comps[k][test] for k in keys])
        for k, v in zip(keys, adj):
            comps[k][test + "_holm"] = v
    out["comparisons"] = comps
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
    sim_s = {"tmaze_choice": 30 + 45 + 30 + 30, "place_avoidance": 30 + 30 + 30, "conflict": 60,
             "food_search": 100}.get(name, 60)
    groups = len(PROTOCOLS[name]["groups"])
    total = sim_s * seconds_per_sim_s * n_reps * groups
    return {"replicates": n_reps * groups, "sim_seconds_each": sim_s,
            "wall_hours_single_process": round(total / 3600, 1)}
