#!/bin/zsh
# Sequential experiment queue (8 GB RAM: one at a time). Logs in results/logs/.
cd "$(dirname "$0")/.."
mkdir -p results/logs
export NUMBA_NUM_THREADS=6
run() { echo "=== $(date +%H:%M:%S) $*"; .venv/bin/python scripts/run_experiment.py "$@" > "results/logs/${1}_${2//[^a-zA-Z0-9_.=-]/}_$3.log" 2>&1; tail -1 "results/logs/${1}_${2//[^a-zA-Z0-9_.=-]/}_$3.log" | cut -c1-400; }
run pair mode=courtship seconds=60
run pair mode=fight seconds=60
run tmaze_choice cond=full retention_h='[0,0.5,1,3]'
run tmaze_choice cond=no_plasticity
run tmaze_choice cond=no_us_shortcut
run tmaze_choice cond=reflex_only
run place_avoidance cond=full
run place_avoidance cond=no_plasticity
run food_search_after_aversion cond=full
run food_search_after_aversion cond=no_plasticity
run conflict starved_h=24
run conflict starved_h=0
echo "=== tracking"; .venv/bin/python scripts/tracking.py female > results/logs/tracking_female.log 2>&1; .venv/bin/python scripts/tracking.py male > results/logs/tracking_male.log 2>&1
.venv/bin/python scripts/report.py
echo "=== done $(date +%H:%M:%S)"
