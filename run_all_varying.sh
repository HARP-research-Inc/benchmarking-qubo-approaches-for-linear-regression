#!/usr/bin/env bash
set -euo pipefail

REPS=40
DIMS=(4 8 16 32 64 128 256)
NOISE_SET=(0.01 0.05)
CORR_SET=(0.0 0.8)
MAX_ITER=30
NUM_SOLVES=1
TIMEOUT=60
TIMEOUT_ONE_SHOT=1000

ts=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/run_${ts}"
mkdir -p "$OUTDIR"

for noise in "${NOISE_SET[@]}"; do
  for corr in "${CORR_SET[@]}";  do
    for ((r=1;r<=REPS;r++)); do
      SEED=$((1000+r))

      # --------------- classical ---------------
      python main.py --mode classical \
        --dims "${DIMS[@]}" --noise "$noise" --corr "$corr" \
        --seed "$SEED" --out "$OUTDIR/classical_${noise}_${corr}_rep${r}.csv"

      # --------------- box-naive ---------------
      python main.py --mode box-naive \
        --dims "${DIMS[@]}" --noise "$noise" --corr "$corr" \
        --seed "$SEED" --max_iter "$MAX_ITER" \
        --num_solves "$NUM_SOLVES" --timeout_ms "$TIMEOUT" \
        --out "$OUTDIR/box_naive_${noise}_${corr}_rep${r}.csv"

      # --------------- box-opt ---------------
      python main.py --mode box-opt \
        --dims "${DIMS[@]}" --noise "$noise" --corr "$corr" \
        --seed "$SEED" --max_iter "$MAX_ITER" \
        --num_solves "$NUM_SOLVES" --timeout_ms "$TIMEOUT" \
        --out "$OUTDIR/box_opt_${noise}_${corr}_rep${r}.csv"

      # --------------- potok ---------------
      python main.py --mode potok \
        --dims "${DIMS[@]}" --noise "$noise" --corr "$corr" \
        --seed "$SEED" --prec_bits 2 3 \
        --timeout_ms "$TIMEOUT_ONE_SHOT" \
        --out "$OUTDIR/potok_${noise}_${corr}_rep${r}.csv"
    done
  done
done

