"""Body-transplant analysis: does each brain re-tune to its new body?

  .venv/bin/python scripts/analyze_transplant.py results/runs/<run_id>

For every fly, in 10 s windows: correlation between the brain's turn command and the body's real
yaw rate (sensorimotor coupling), walking speed, punishment-DAN (PPL1) rate. Reports the value
before and after the transplant and the time until the coupling is back to >= 80 % of its
pre-transplant level. This measures sensorimotor adaptation, not "understanding".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WIN_S = 10.0


def windows(d, t0):
    d = d.assign(rel=(d.clock_h - t0) * 3600.0)
    d["w"] = np.floor(d.rel / WIN_S).astype(int)
    out = []
    for w, g in d.groupby("w"):
        if len(g) < 20 or g.cmd_turn.std() < 1e-6 or g.yaw_rate_deg_s.std() < 1e-6:
            r = np.nan
        else:
            r = float(np.corrcoef(g.cmd_turn, g.yaw_rate_deg_s)[0, 1])
        out.append({"t_s": w * WIN_S, "coupling_r": r, "speed": g.speed_mm_s.mean(),
                    "ppl1": g.PPL1_punish_Hz.mean()})
    return pd.DataFrame(out)


def main(run_dir):
    run_dir = Path(run_dir)
    ts = pd.read_csv(run_dir / "timeseries.csv")
    ev = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines() if x.strip()]
    tp = [e for e in ev if e["type"] == "transplant"]
    if not tp:
        print("no transplant event in this run")
        return
    t0 = tp[0]["clock_h"]
    print(f"transplant at world clock {t0 * 3600:.1f} s\n")
    for fly, d in ts.groupby("fly"):
        wdf = windows(d.sort_values("clock_h"), t0)
        pre, post = wdf[wdf.t_s < 0], wdf[wdf.t_s >= 0]
        base = pre.coupling_r.mean()
        rec = post[post.coupling_r >= 0.8 * base].t_s.min() if base == base and base > 0 else np.nan
        print(f"{fly}: coupling r before {base:.2f}, first 10 s after {post.coupling_r.iloc[0] if len(post) else np.nan:.2f}, "
              f"later {post.coupling_r.iloc[1:].mean():.2f}; recovered after {rec} s")
        print(f"   speed {pre.speed.mean():.1f} -> {post.speed.mean():.1f} mm/s, "
              f"PPL1 {pre.ppl1.mean():.1f} -> {post.ppl1.mean():.1f} Hz")
        print(wdf.round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main(sys.argv[1])
