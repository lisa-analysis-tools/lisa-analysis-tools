#!/bin/bash
# EMRI direct-to-WDM vs production at 6, 12 and 24 months on one GPU (production convention:
# 15552000 s per 6 months -> 180 / 360 / 720 d; a 730 d window does not fit: the packaged
# equal-arm orbits end REF + 697.9 d and the 731-day mojito bricks' data and light-travel times
# REF + 730.5 d, while the window starts REF + START_OFFSET_S and the production wrapper reads
# 4e4 s past its end -- so 24 months needs the source's L1 brick, ORBITS=auto/l1).
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
#   plus the FITTED HYPERBOLIC-TANGENT GALACTIC FOREGROUND at that duration's Tobs
#   (FittedHyperbolicTangentGalacticForeground; FOREGROUND=off = instrument only). MOJITO DATA: with ORBITS=auto (default) the source's mojito L1
#   brick is looked up (L1_DIR if set, else MOJITO_LIGHT_PATH/data/EMRI/L1, then recursively
#   MOJITO_DATA_PATH, MOJITO_INFO_PATH and the catalogue's root); when found, the templates use
#   its orbits and window and each is scored against its source-only stream ([data] lines:
#   data SNR, mismatch, logL = -1/2 <d-h|d-h>, opt/data SNR ratio). The data holds every mode:
#   a template's mismatch includes its mode truncation. No brick: equal-arm orbits, no data.
#
#   salloc ... (1 GPU), then from the repo root:
#       bash scripts/emri/emri_speed_durations.sh
#   knobs (env): BACKEND (cuda13x), DAYS ("180 360 720"), THRESH ("1e-3,1e-5"), SRC (1),
#   TABLE, CATALOG, ORBITS (auto; equal-arm skips the data), L1_DIR (""), MOJITO_DATA_PATH
#   (/shared/data/mojito_cache), CHUNKS ("1,2,4,8,16,32"), ROWS (32), GRIDS
#   ("sparse,pixels"), LOOKUPS ("kernel,python"), BAND ("2.5e-4,2.5e-2"; "" = all layers), EDGE (60),
#   FOREGROUND (on),
#   REPS (3), OUT (emri_speed_durations_<date>).
#
# Memory: a direct template holds a (3, n_m, Nt) float64 accumulator while it is built: on the
# full grid (BAND="") 150 MB at 6 months, 300 MB at 12, 605 MB at 24; the 2.5e-4..2.5e-2 Hz
# band keeps 179 of 1440 layers (~8x less). The kernel lookup has no other per-pixel arrays;
# the python lookup adds its temporaries (the per-call GPU pool footprint is printed).
set -euo pipefail

BACKEND=${BACKEND:-cuda13x}
DAYS=${DAYS:-180 360 720}
THRESH=${THRESH:-1e-3,1e-5}
SRC=${SRC:-1}
TABLE=${TABLE:-wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5}
CATALOG=${CATALOG:-/shared/data/mojito_cache/catalogues/emri_cat_mojito_lite_processed_MT.hdf5}
ORBITS=${ORBITS:-auto}
L1_DIR=${L1_DIR:-}
export MOJITO_DATA_PATH=${MOJITO_DATA_PATH:-/shared/data/mojito_cache}
CHUNKS=${CHUNKS:-1,2,4,8,16,32}
ROWS=${ROWS:-32}
GRIDS=${GRIDS:-sparse,pixels}
LOOKUPS=${LOOKUPS:-kernel,python}
BAND=${BAND-2.5e-4,2.5e-2}
EDGE=${EDGE:-60}
FOREGROUND=${FOREGROUND:-on}
REPS=${REPS:-3}
OUT=${OUT:-emri_speed_durations_$(date +%Y%m%d_%H%M)}
mkdir -p "${OUT}"
[ -f "${TABLE}" ] || { echo "[durations] lookup table ${TABLE} not found (set TABLE=...)"; exit 2; }

common=(--backend "${BACKEND}" --dt 2.5 --src "${SRC}" --direct-table "${TABLE}"
        --catalog "${CATALOG}" --orbits "${ORBITS}" --edge "${EDGE}" --foreground "${FOREGROUND}")
if [ -n "${L1_DIR}" ]; then common+=(--l1-dir "${L1_DIR}"); fi
for days in ${DAYS}; do
  log="${OUT}/stages_${days}d.log"
  echo "[durations] ${days} d: grids ${GRIDS}, lookups ${LOOKUPS}, band ${BAND:-all}, rows per call ${CHUNKS} over ${ROWS} rows -> ${log}"
  python scripts/emri/emri_direct_stage_timing.py "${common[@]}" --days "${days}" --thresh "${THRESH}" \
    --response dense --response-grid "${GRIDS}" --lookup "${LOOKUPS}" --band-hz "${BAND}" \
    --batch-rows "${ROWS}" --chunk-rows "${CHUNKS}" \
    --reps "${REPS}" --out "${OUT}/stages.jsonl" 2>&1 | grep -v "ModeSelector\|lisaconstants\|warnings.warn" \
    > "${log}" || { echo "[durations] FAILED: ${log}"; tail -4 "${log}"; }
  grep "^\[snr\]\|^\[data\]\|^\[speed\] orbits" "${log}" || true
  sed -n '/^\[summary\]/,/^$/p' "${log}" || true
  log="${OUT}/accuracy_${days}d.log"
  echo "[durations] ${days} d accuracy sparse vs pixels vs production -> ${log}"
  python scripts/emri/emri_direct_sparse_check.py "${common[@]}" --days "${days}" --thresh "${THRESH}" \
    --rows 16 2>&1 | grep -v "ModeSelector\|lisaconstants\|warnings.warn" > "${log}" \
    || { echo "[durations] FAILED: ${log}"; tail -4 "${log}"; }
  grep "^\[sparse\]" "${log}" || true
done
echo "[durations] done: ${OUT}/"
