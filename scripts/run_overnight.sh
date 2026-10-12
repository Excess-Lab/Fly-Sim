#!/bin/zsh
# Replicated series (3 seeds per group) + social pairs, one process at a time (8 GB RAM).
cd "$(dirname "$0")/.."
mkdir -p results/logs
export NUMBA_NUM_THREADS=6
for p in conflict tmaze_choice place_avoidance food_search; do
  echo "=== $(date +%H:%M) series $p"
  .venv/bin/python scripts/series.py run $p 3 > results/logs/series_$p.log 2>&1
  tail -25 results/logs/series_$p.log | head -40
done
for m in courtship fight; do
  echo "=== $(date +%H:%M) pair $m"
  .venv/bin/python scripts/run_experiment.py pair mode=$m seconds=60 > results/logs/pair_$m.log 2>&1
  tail -1 results/logs/pair_$m.log | cut -c1-600
done
echo "=== tracking"; .venv/bin/python scripts/tracking.py female > results/logs/tracking_female.log 2>&1
.venv/bin/python scripts/report.py
echo "=== done $(date +%H:%M)"
