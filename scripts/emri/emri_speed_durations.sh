#!/bin/bash
# EMRI direct-to-WDM vs production at 6, 12 and 24 months on one GPU.
#
# Per duration (FEW and the table load once per step):
#   1. emri_direct_stage_timing.py, dense response, both response grids (sparse = knots + 12 h
#      cap with the exact carrier, the default; pixels = one sample per 3600 s pixel), both
#      lookups (kernel = the fused C++/CUDA lookup sum, the default; python = tracer + table +
#      scatter-add), the batch built on the fit's active band (BAND, Hz), single
#      template plus a sweep of rows per response call (CHUNKS; the fit's in-model steps are 2,
#      its eigen sweeps EMRI_BATCH_MAX_SIZE) over ROWS rows: per-stage ms per template, the GPU
#      memory-pool footprint per call size, production ms per template, a summary table per
#      threshold, and one JSON line per run in ${OUT}/stages.jsonl.
#   2. emri_direct_sparse_check.py: mismatch sparse-vs-pixels and both vs production on the
#      grid of that duration, plus single and batched ms per template.
#   Both print the source's optimal SNR at that duration (production and direct templates) on
#   the fit's run box: 0.25-25 mHz, EDGE pixels cropped at each end, scirdv1 XYZ sensitivity
#   (no galactic foreground).
#
#   salloc ... (1 GPU), then from the repo root:
#       bash scripts/emri/emri_speed_durations.sh
#   knobs (env): BACKEND (cuda13x), DAYS ("180 365 730"), THRESH ("1e-3,1e-5"), SRC (1),
#   TABLE, CATALOG, ORBITS (equal-arm), CHUNKS ("1,2,4,8,16,32"), ROWS (32), GRIDS
#   ("sparse,pixels"), LOOKUPS ("kernel,python"), BAND ("2.5e-4,2.5e-2"; "" = all layers), EDGE (60),
#   REPS (3), OUT (emri_speed_durations_<date>).
#
# Memory: a direct template holds a (3, n_m, Nt) float64 accumulator while it is built: on the
# full grid (BAND="") 150 MB at 6 months, 300 MB at 12, 605 MB at 24; the 2.5e-4..2.5e-2 Hz
# band keeps 179 of 1440 layers (~8x less). The kernel lookup has no other per-pixel arrays;
# the python lookup adds its temporaries (the per-call GPU pool footprint is printed).
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
LOOKUPS=${LOOKUPS:-kernel,python}
BAND=${BAND-2.5e-4,2.5e-2}
EDGE=${EDGE:-60}
REPS=${REPS:-3}
OUT=${OUT:-emri_speed_durations_$(date +%Y%m%d_%H%M)}
mkdir -p "${OUT}"
[ -f "${TABLE}" ] || { echo "[durations] lookup table ${TABLE} not found (set TABLE=...)"; exit 2; }

common=(--backend "${BACKEND}" --dt 2.5 --src "${SRC}" --direct-table "${TABLE}"
        --catalog "${CATALOG}" --orbits "${ORBITS}" --edge "${EDGE}")
for days in ${DAYS}; do
  log="${OUT}/stages_${days}d.log"
  echo "[durations] ${days} d: grids ${GRIDS}, lookups ${LOOKUPS}, band ${BAND:-all}, rows per call ${CHUNKS} over ${ROWS} rows -> ${log}"
  python scripts/emri/emri_direct_stage_timing.py "${common[@]}" --days "${days}" --thresh "${THRESH}" \
    --response dense --response-grid "${GRIDS}" --lookup "${LOOKUPS}" --band-hz "${BAND}" \
    --batch-rows "${ROWS}" --chunk-rows "${CHUNKS}" \
    --reps "${REPS}" --out "${OUT}/stages.jsonl" 2>&1 | grep -v "ModeSelector\|lisaconstants\|warnings.warn" \
    > "${log}" || echo "[durations] FAILED: ${log}"
  grep "^\[snr\]" "${log}" || true
  sed -n '/^\[summary\]/,/^$/p' "${log}" || true
  log="${OUT}/accuracy_${days}d.log"
  echo "[durations] ${days} d accuracy sparse vs pixels vs production -> ${log}"
  python scripts/emri/emri_direct_sparse_check.py "${common[@]}" --days "${days}" --thresh "${THRESH}" \
    --rows 16 2>&1 | grep -v "ModeSelector\|lisaconstants\|warnings.warn" > "${log}" \
    || echo "[durations] FAILED: ${log}"
  grep "^\[sparse\]" "${log}" || true
done
echo "[durations] done: ${OUT}/"
