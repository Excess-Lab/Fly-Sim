"""Replicated experiment series with statistics and power analysis.

  python scripts/series.py plan  tmaze_choice 10            # cost estimate on this machine
  python scripts/series.py run   tmaze_choice 10 --proc 4   # run 10 seeds per group
  python scripts/series.py stats tmaze_choice               # latest run: every group vs control
  python scripts/series.py stats tmaze_choice --run all --version <git describe>   # pool runs

Groups get independent seed streams (protocol.group_seed); each `run` call writes a run_id and
`stats` uses only the latest run unless --run / --version say otherwise (the jsonl file is
appended across runs and model versions). Comparisons: every group vs the protocol's control
group, Welch t and Mann-Whitney with Holm correction, Cohen's d (sign = group - control).
"""
import argparse
import json

from flysim.sim import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["plan", "run", "stats", "list"])
ap.add_argument("name", nargs="?")
ap.add_argument("n", nargs="?", type=int, default=5)
ap.add_argument("--proc", type=int, default=1)
ap.add_argument("--speed", type=float, default=10.0, help="wall s per simulated s per fly")
ap.add_argument("--run", default="latest", help="run_id to analyse, 'latest' (default) or 'all'")
ap.add_argument("--version", default=None, help="only records of this model_version")
a = ap.parse_args()
if a.action == "list":
    for k, v in P.PROTOCOLS.items():
        print(f"{k:16s} control={v['control']:14s} {v['hypothesis']}")
elif a.action == "plan":
    print(json.dumps(P.compute_budget(a.name, a.n, a.speed), indent=1))
elif a.action == "run":
    P.run_series(a.name, n_reps=a.n, processes=a.proc)
    print(json.dumps(P.analyze(a.name), indent=1, ensure_ascii=False))
else:
    print(json.dumps(P.analyze(a.name, run_id=a.run, model_version=a.version), indent=1,
                     ensure_ascii=False))
