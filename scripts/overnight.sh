#!/bin/sh
# 24-hour test run inside the Docker container (CPU or GPU, chosen automatically):
#   docker compose exec flysim sh scripts/overnight.sh
# Everything is written to results/overnight/ in the project folder on your disk; every finished
# replicate is saved at once, so a shutdown loses only the replicate that was running.
cd "$(dirname "$0")/.."
OUT=results/overnight
mkdir -p "$OUT"
PROC=${PROC:-2}          # parallel replicates; each needs ~1.5 GB RAM
REPS=${REPS:-6}          # replicates per group (more = tighter confidence intervals)
log() { echo "$(date '+%d.%m %H:%M') $*" | tee -a "$OUT/progress.txt"; }

log "start: backend $(python -c 'from flysim.brain.lif import backend_info as b; print(b())')"
log "1/4 correctness checks"
python scripts/validate_all.py female_banc male --long 20 > "$OUT/validate.txt" 2>&1
log "    $(tail -1 "$OUT/validate.txt")"
log "2/4 three nervous systems on identical tests"
python scripts/compare_connectomes.py > "$OUT/compare_connectomes.txt" 2>&1
if python -c 'from flysim.brain.lif import backend_info as b; exit(b()["backend"] != "gpu")'; then
  log "3/4 GPU vs CPU check and speed"
  python scripts/check_cuda.py --bench female_banc > "$OUT/gpu_check.txt" 2>&1
  log "    $(grep -E 'per simulated second|OK|MISMATCH' "$OUT/gpu_check.txt" | tr '\n' ' ')"
else
  log "3/4 no GPU: skipped"
fi
log "4/4 experiment series ($REPS replicates per group, $PROC at a time)"
for p in conflict place_avoidance tmaze_choice food_search; do
  log "    series $p"
  python scripts/series.py run $p $REPS --proc $PROC > "$OUT/series_$p.txt" 2>&1
  log "    done $p"
done
log "finished"
