"""Replicated experiment series with statistics and power analysis.

  python scripts/series.py plan  tmaze_choice 10            # cost estimate on this machine
  python scripts/series.py run   tmaze_choice 10 --proc 4   # run 10 seeds per group
  python scripts/series.py stats tmaze_choice               # means, 95% CI, d, p, power-based n
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
a = ap.parse_args()
if a.action == "list":
    for k, v in P.PROTOCOLS.items():
        print(f"{k:16s} {v['hypothesis']}")
elif a.action == "plan":
    print(json.dumps(P.compute_budget(a.name, a.n, a.speed), indent=1))
elif a.action == "run":
    P.run_series(a.name, n_reps=a.n, processes=a.proc)
    print(json.dumps(P.analyze(a.name), indent=1, ensure_ascii=False))
else:
    print(json.dumps(P.analyze(a.name), indent=1, ensure_ascii=False))
