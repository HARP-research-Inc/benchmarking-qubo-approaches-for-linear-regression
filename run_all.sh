#!/usr/bin/env bash
#
# run_all.sh ─ repeat the full benchmark matrix several times
# Usage:  ./run_all.sh [reps]        # default reps = 3
# Requires:  AE_KEY env‑var already exported

set -euo pipefail

REPS="${1:-2}"                 # how many full passes
DIMS=(4 8 16 32 64 128 256)              # feature counts (n = 10*d in code)
NOISE=0.01
CORR=0.0
MAX_ITER=40
NUM_SOLVES=1
TIMEOUT=40                    # ms
TIMEOUT_ONE_SHOT=1000

ts="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/run_${ts}"
mkdir -p "${OUTDIR}"

echo "Writing CSVs under ${OUTDIR}"
echo "Running ${REPS} repetitions over dims: ${DIMS[*]}"

for r in $(seq 1 "${REPS}"); do
  echo "=== repetition ${r}/${REPS} ==="

  # ---------- classical ----------
  python main.py --mode classical \
    --dims "${DIMS[@]}"           \
    --noise "${NOISE}"            \
    --corr  "${CORR}"             \
    --seed  "${r}"                \
    --out   "${OUTDIR}/classical_rep${r}.csv"

  # ---------- box naive ----------
  python main.py --mode box-naive \
    --dims "${DIMS[@]}"           \
    --noise "${NOISE}"            \
    --corr  "${CORR}"             \
    --seed  "${r}"                \
    --max_iter "${MAX_ITER}"      \
    --num_solves "${NUM_SOLVES}"  \
    --timeout_ms "${TIMEOUT}"     \
    --out   "${OUTDIR}/box_naive_rep${r}.csv"

  # ---------- box optimised ----------
  python main.py --mode box-opt   \
    --dims "${DIMS[@]}"           \
    --noise "${NOISE}"            \
    --corr  "${CORR}"             \
    --seed  "${r}"                \
    --max_iter "${MAX_ITER}"      \
    --num_solves "${NUM_SOLVES}"  \
    --timeout_ms "${TIMEOUT}"     \
    --out   "${OUTDIR}/box_opt_rep${r}.csv"

  # ---------- potok QUBO ----------
  # sweep K = 2–4 precisions in a single call
  python main.py --mode potok     \
    --dims "${DIMS[@]}"           \
    --noise "${NOISE}"            \
    --corr  "${CORR}"             \
    --seed  "${r}"                \
    --prec_bits 2 3 4            \
    --timeout_ms "${TIMEOUT_ONE_SHOT}"     \
    --out   "${OUTDIR}/potok_rep${r}.csv"

done

echo "All done.  CSVs written to ${OUTDIR}"

