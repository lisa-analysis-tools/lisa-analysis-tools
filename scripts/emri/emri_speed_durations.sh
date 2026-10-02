#!/bin/bash
# EMRI direct-to-WDM vs production at 6, 12 and 24 months on one GPU.
#
# Per duration (FEW and the table load once per step):
#   1. emri_direct_stage_timing.py, dense response, both response grids (sparse = knots + 12 h
#      cap with the exact carrier, the default; pixels = one sample per 3600 s pixel), single
#      template plus a sweep of rows per response call (CHUNKS; the fit's in-model steps are 2,
#      its eigen sweeps EMRI_BATCH_MAX_SIZE) over ROWS rows: per-stage ms per template, the GPU
#      memory-pool footprint per call size, production ms per template, a summary table per
#      threshold, and one JSON line per run in ${OUT}/stages.jsonl.
#   2. emri_direct_sparse_check.py: mismatch sparse-vs-pixels and both vs production on the
#      grid of that duration, plus single and batched ms per template.
#
#   salloc ... (1 GPU), then from the repo root:
#       bash scripts/emri/emri_speed_durations.sh
#   knobs (env): BACKEND (cuda13x), DAYS ("180 365 730"), THRESH ("1e-3,1e-5"), SRC (1),
#   TABLE, CATALOG, ORBITS (equal-arm), CHUNKS ("1,2,4,8,16,32"), ROWS (32), GRIDS
#   ("sparse,pixels"), REPS (3), OUT (emri_speed_durations_<date>).
#
# Memory: a direct template holds a full (3, 1440, Nt) float64 accumulator while it is built:
# 150 MB at 6 months, 300 MB at 12, 605 MB at 24 -- a 32-row call at 24 months is ~20 GB plus
# the lookup's temporaries (the per-call GPU pool footprint is printed).
set -euo pipefail

BACKEND=${BACKEND:-cuda13x}
DAYS=${DAYS:-180 365 730}
THRESH=${THRESH:-1e-3,1e-5}
SRC=${SRC:-1}
TABLE=${TABLE:-wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5}
CATALOG=${CATALOG:-/shared/data/mojito_cache/catalogues/emri_cat_mojito_lite_processed_MT.hdf5}
ORBITS=${ORBITS:-equal-arm}
CHUNKS=${CHUNKS:-1,2,4,8,16,32}
ROWS=${ROWS:-32}
GRIDS=${GRIDS:-sparse,pixels}
REPS=${REPS:-3}
OUT=${OUT:-emri_speed_durations_$(date +%Y%m%d_%H%M)}
mkdir -p "${OUT}"
[ -f "${TABLE}" ] || { echo "[durations] lookup table ${TABLE} not found (set TABLE=...)"; exit 2; }

common=(--backend "${BACKEND}" --dt 2.5 --src "${SRC}" --direct-table "${TABLE}"
        --catalog "${CATALOG}" --orbits "${ORBITS}")
for days in ${DAYS}; do
  log="${OUT}/stages_${days}d.log"
  echo "[durations] ${days} d: grids ${GRIDS}, rows per call ${CHUNKS} over ${ROWS} rows -> ${log}"
  python scripts/emri/emri_direct_stage_timing.py "${common[@]}" --days "${days}" --thresh "${THRESH}" \
    --response dense --response-grid "${GRIDS}" --batch-rows "${ROWS}" --chunk-rows "${CHUNKS}" \
    --reps "${REPS}" --out "${OUT}/stages.jsonl" 2>&1 | grep -v "ModeSelector\|lisaconstants\|warnings.warn" \
    > "${log}" || echo "[durations] FAILED: ${log}"
  sed -n '/^\[summary\]/,/^$/p' "${log}" || true
  log="${OUT}/accuracy_${days}d.log"
  echo "[durations] ${days} d accuracy sparse vs pixels vs production -> ${log}"
  python scripts/emri/emri_direct_sparse_check.py "${common[@]}" --days "${days}" --thresh "${THRESH}" \
    --rows 16 2>&1 | grep -v "ModeSelector\|lisaconstants\|warnings.warn" > "${log}" \
    || echo "[durations] FAILED: ${log}"
  grep "^\[sparse\]" "${log}" || true
done
echo "[durations] done: ${OUT}/"
